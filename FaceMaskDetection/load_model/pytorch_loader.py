import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + '/../models/')
sys.path.append('FaceMaskDetection/models/')


def load_pytorch_model(model_path):
    # torch.serialization.add_safe_globals(['MainModel.KitModel'])
    model = torch.load(model_path)
    return model


def pytorch_inference(model, img_arr):
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    device = torch.device(dev)
    model.to(device)
    input_tensor = torch.tensor(img_arr).float().to(device)
    y_bboxes, y_scores, = model.forward(input_tensor)
    return y_bboxes.detach().cpu().numpy(), y_scores.detach().cpu().numpy()
