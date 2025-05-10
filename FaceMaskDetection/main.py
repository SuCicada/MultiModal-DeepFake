# -*- coding:utf-8 -*-
from enum import Enum
import os
import cv2
import time
import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)

import argparse
import numpy as np
from PIL import Image
from FaceMaskDetection.utils.anchor_generator import generate_anchors
from FaceMaskDetection.utils.anchor_decode import decode_bbox
from FaceMaskDetection.utils.nms import single_class_non_max_suppression
from FaceMaskDetection.load_model.pytorch_loader import load_pytorch_model, pytorch_inference

# model = load_pytorch_model('models/face_mask_detection.pth');
current_dir = os.path.dirname(os.path.abspath(__file__))
model = load_pytorch_model(current_dir + '/models/model360.pth')
# anchor configuration
# feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11],
                [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}


def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )
    face_result = Result.NO_FACE
    result = {
        "face_result": face_result.value,
        "bbox": [],
        "conf": 0,
        "class_id": 0
    }
    print("keep_idxs: ", keep_idxs)
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        # print(
        # "conf: ", conf,
        # "class_id: ", class_id,
        # "xmin: ", xmin, "ymin: ", ymin, "xmax: ", xmax, "ymax: ", ymax)
        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        if class_id == 0:
            face_result = Result.MASK
        elif class_id == 1:
            face_result = Result.NO_MASK
        # output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
        result = {
            "face_result": face_result.value,
            "bbox": [xmin, ymin, xmax, ymax],
            "conf": conf,
            "class_id": class_id
        }

    if show_result:
        # Image.fromarray(image).show()
        cv2.imwrite("result.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return result


class Result(Enum):
    NO_FACE = 0
    MASK = 1
    NO_MASK = 2


def face_mask_detection(img_path):
    """
    @return 
        {
            "face_result": face_result.value, 
            "bbox": [xmin, ymin, xmax, ymax],
            "conf": conf,
            "class_id": class_id
        }
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = inference(img, show_result=True, target_shape=(360, 360))
    return result

# dir = "/home/peng/PROGRAM/GitHub/MultiModal-DeepFake/data/DGM4/mask"
# files = os.listdir(dir)
# files.sort()
# for file in files:
#     image_path = dir + "/" + file
#     print(f"{image_path}")
#     result = run_on_image(image_path)
#     print(result)


def main():
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--img-path', type=str,
                        default='img/demo2.jpg', help='path to your image.')
    args = parser.parse_args()
    images = args.img_path.split(",")
    res = {}
    for image in images:
        img_res = face_mask_detection(image)
        res[image] = img_res
    print(res)


if __name__ == "__main__":
    main()
