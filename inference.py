import sys
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

import utils
from dataset import create_dataset, create_loader
from models.vit import interpolate_pos_embed
from models.HAMMER import HAMMER
from transformers import BertTokenizerFast
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch
import ruamel_yaml as yaml
import argparse
import os
import warnings

from tools.multilabel_metrics import AveragePrecisionMeter

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


# img = "data/DGM4/manipulation/StyleCLIP/1096799-StyleCLIP.jpg"
# text = "The other day about six girls asked to kiss me on the cheek I feel like I ve earned it Sir Peter Blake"

def preprocess_image(image, size=224):
    """Preprocess image for model input"""
    transform = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(image)


def text_input_adjust(text_input, fake_word_pos, device):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids]) - 1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in
                                input_ids_remove_SEP]  # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device)

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    subword_idx_rm_CLSSEP_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []

        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[
            0].tolist()  # transfer fake_word_pos into numbers

        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP)  # get the sub-word position (token position)

        subword_idx_rm_CLSSEP_batch.append(subword_idx_rm_CLSSEP_array)

        # transfer the fake word position into fake token position
        for i in fake_word_pos_decimal:
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch, subword_idx_rm_CLSSEP_batch


def visualize_results(image_path, text, output_coord, logits_tok, tokenizer, image_size=224):
    if type(image_path) is list:
        image_path = image_path[0]
    if type(text) is list:
        text = text[0]
    pprint({
        'image_path': image_path,
        'text': text,
        'output_coord': output_coord,
        'logits_tok': logits_tok
    })
    # image_path, text = image_path, text
    """Visualize the detection results"""
    # Load and display image
    image = Image.open(image_path).convert('RGB')
    W, H = image.size

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw detection box
    if output_coord is not None:
        # Convert normalized coordinates to image coordinates
        cx, cy, w, h = output_coord[0].cpu().numpy()
        x = (cx - w / 2) * W
        y = (cy - h / 2) * H
        width = w * W
        height = h * H

        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Process text tokens
    tokens = tokenizer.tokenize(text)
    logits_tok = logits_tok.view(-1, 2)
    manipulated_tokens = logits_tok.argmax(1).cpu().numpy()

    # Print manipulated words
    print("\nManipulated Words:")
    for i, (token, is_manipulated) in enumerate(zip(tokens[:len(manipulated_tokens)], manipulated_tokens)):
        if is_manipulated:
            print(f"{token} (manipulated)")

    plt.axis('off')
    plt.show()
    output_path = "output.png"
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    # parser.add_argument('--image', required=True, help='Path to input image')
    # parser.add_argument('--text', required=True, help='Input text')

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.log = True
    # Initialize model
    device = torch.device(args.device)
    tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)
    model = HAMMER(args=args, config=config, text_encoder=args.text_encoder,
                   tokenizer=tokenizer, init_deit=True)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    pos_embed_reshaped = interpolate_pos_embed(
        state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    print('load checkpoint from %s' % args.checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # model = model.to(device)

    # Run inference
    # results = inference(args, model, args.image, args.text, device, config)

    _, val_dataset = create_dataset(config)
    samplers = [None]
    val_loader = create_loader([val_dataset],
                               samplers,
                               batch_size=[config['batch_size_val']],
                               num_workers=[4],
                               is_trains=[False],
                               collate_fns=[None])[0]
    model_without_ddp = model
    evaluation(args, model, val_loader, tokenizer, device, config)

    # Print results
    # print("\nInference Results:")
    # print(f"Image: {args.image}")
    # print(f"Text: {args.text}")
    # print(f"\nReal/Fake Classification:")
    # print(f"Is Fake: {results['is_fake']}")
    # print(f"Confidence: {results['real_fake_probs'][0][1]:.2%}")
    #
    # print(f"\nMulti-label Classification:")
    # label_names = ['Face Swap', 'Face Attribute',
    #                'Text Swap', 'Text Attribute']
    # for i, (label, prob) in enumerate(zip(results['multi_labels'][0], results['multi_label_probs'][0])):
    #     print(
    #         f"{label_names[i]}: {'Yes' if label else 'No'} (Confidence: {prob:.2%})")
    #
    # print(f"\nDetection Box (cx, cy, w, h):")
    # print(results['detection_box'][0])
    #
    # print(f"\nManipulated Words:")
    # tokens = tokenizer.tokenize(results['text'])
    # # Ensure tokens and manipulated_tokens have the same length
    # for i, (token, is_manipulated) in enumerate(zip(tokens[:len(results['manipulated_tokens'])],
    #                                                 results['manipulated_tokens'])):
    #     if is_manipulated:
    #         print(f"{token} (manipulated)")


@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    print_freq = 200

    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []
    cls_nums_all = 0
    cls_acc_all = 0

    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0

    TP_all_multicls = np.zeros(4, dtype=int)
    TN_all_multicls = np.zeros(4, dtype=int)
    FP_all_multicls = np.zeros(4, dtype=int)
    FN_all_multicls = np.zeros(4, dtype=int)
    F1_multicls = np.zeros(4)

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for i, (image_path, image, label, text, fake_image_box, fake_word_pos, W, H) in (
            enumerate(metric_logger.log_every(args, data_loader, print_freq, header))):
        image = image.to(device, non_blocking=True)

        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True,
                               return_attention_mask=True, return_token_type_ids=False)

        text_input, fake_token_pos, _ = text_input_adjust(text_input, fake_word_pos, device)

        logits_real_fake, logits_multicls, output_coord, logits_tok = model(
            image, label, text_input, fake_image_box,
            fake_token_pos, is_train=False)
        pprint({
            "logits_real_fake": logits_real_fake,
            "logits_multicls": logits_multicls,
            "output_coord": output_coord,
            "logits_tok": logits_tok
        })

        visualize_results(image_path[0], text, output_coord, logits_tok, tokenizer, )
    #     ##================= real/fake cls ========================##
    #     cls_label = torch.ones(len(label), dtype=torch.long).to(image.device)
    #     real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
    #     cls_label[real_label_pos] = 0
    #
    #     y_pred.extend(F.softmax(logits_real_fake, dim=1)[:, 1].cpu().flatten().tolist())
    #     y_true.extend(cls_label.cpu().flatten().tolist())
    #
    #     pred_acc = logits_real_fake.argmax(1)
    #     cls_nums_all += cls_label.shape[0]
    #     cls_acc_all += torch.sum(pred_acc == cls_label).item()
    #
    #     # ----- multi metrics -----
    #     target, _ = get_multi_label(label, image)
    #     multi_label_meter.add(logits_multicls, target)
    #
    #     for cls_idx in range(logits_multicls.shape[1]):
    #         cls_pred = logits_multicls[:, cls_idx]
    #         cls_pred[cls_pred >= 0] = 1
    #         cls_pred[cls_pred < 0] = 0
    #
    #         TP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 1)).item()
    #         TN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 0)).item()
    #         FP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 1)).item()
    #         FN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 0)).item()
    #
    #     ##================= bbox cls ========================##
    #     boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
    #     boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)
    #
    #     IOU, _ = box_ops.box_iou(boxes1, boxes2.to(device), test=True)
    #
    #     IOU_pred.extend(IOU.cpu().tolist())
    #
    #     IOU_50_bt = torch.zeros(IOU.shape, dtype=torch.long)
    #     IOU_75_bt = torch.zeros(IOU.shape, dtype=torch.long)
    #     IOU_95_bt = torch.zeros(IOU.shape, dtype=torch.long)
    #
    #     IOU_50_bt[IOU > 0.5] = 1
    #     IOU_75_bt[IOU > 0.75] = 1
    #     IOU_95_bt[IOU > 0.95] = 1
    #
    #     IOU_50.extend(IOU_50_bt.cpu().tolist())
    #     IOU_75.extend(IOU_75_bt.cpu().tolist())
    #     IOU_95.extend(IOU_95_bt.cpu().tolist())
    #
    #     ##================= token cls ========================##
    #     token_label = text_input.attention_mask[:, 1:].clone()  # [:,1:] for ingoring class token
    #     token_label[token_label == 0] = -100  # -100 index = padding token
    #     token_label[token_label == 1] = 0
    #
    #     for batch_idx in range(len(fake_token_pos)):
    #         fake_pos_sample = fake_token_pos[batch_idx]
    #         if fake_pos_sample:
    #             for pos in fake_pos_sample:
    #                 token_label[batch_idx, pos] = 1
    #
    #     logits_tok_reshape = logits_tok.view(-1, 2)
    #     logits_tok_pred = logits_tok_reshape.argmax(1)
    #     token_label_reshape = token_label.view(-1)
    #
    #     # F1
    #     TP_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
    #     TN_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
    #     FP_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
    #     FN_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()
    #
    # ##================= real/fake cls ========================##
    # y_true, y_pred = np.array(y_true), np.array(y_pred)
    # AUC_cls = roc_auc_score(y_true, y_pred)
    # ACC_cls = cls_acc_all / cls_nums_all
    # fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    # EER_cls = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    #
    # ##================= bbox cls ========================##
    # IOU_score = sum(IOU_pred) / len(IOU_pred)
    # IOU_ACC_50 = sum(IOU_50) / len(IOU_50)
    # IOU_ACC_75 = sum(IOU_75) / len(IOU_75)
    # IOU_ACC_95 = sum(IOU_95) / len(IOU_95)
    # # ##================= token cls========================##
    # ACC_tok = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
    # Precision_tok = TP_all / (TP_all + FP_all)
    # Recall_tok = TP_all / (TP_all + FN_all)
    # F1_tok = 2 * Precision_tok * Recall_tok / (Precision_tok + Recall_tok)
    # ##================= multi-label cls ========================##
    # MAP = multi_label_meter.value().mean()
    # OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()
    #
    # for cls_idx in range(logits_multicls.shape[1]):
    #     Precision_multicls = TP_all_multicls[cls_idx] / (TP_all_multicls[cls_idx] + FP_all_multicls[cls_idx])
    #     Recall_multicls = TP_all_multicls[cls_idx] / (TP_all_multicls[cls_idx] + FN_all_multicls[cls_idx])
    #     F1_multicls[cls_idx] = 2 * Precision_multicls * Recall_multicls / (Precision_multicls + Recall_multicls)
    #
    # return AUC_cls, ACC_cls, EER_cls, \
    #     MAP.item(), OP, OR, OF1, CP, CR, CF1, F1_multicls, \
    #     IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
    #     ACC_tok, Precision_tok, Recall_tok, F1_tok


# "image": "DGM4/origin/bbc/0498/511.jpg",
# "text": "Geoffrey Martin has run a butcher shop for 28 years",

if __name__ == '__main__':
    # image = "data/DGM4/origin/bbc/0498/511.jpg"
    # text = "Geoffrey Martin has run a butcher shop for 28 years"

    sys.argv = ['inference.py',
                '--config', './configs/my.yaml',
                '--checkpoint', './results/checkpoint/checkpoint_best.pth',
                '--text_encoder', 'bert-base-uncased',
                '--device', 'cuda',
                # '--image', image,
                # '--text', text]
                ]

    main()
