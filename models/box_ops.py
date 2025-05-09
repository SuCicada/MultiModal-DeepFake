# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
import math

def box_cxcywh_to_xyxy(x):  # 这个用了（从中心点转为角点）w宽，h高
    x_c, y_c, w, h = x.unbind(-1)#拆开最后一维
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)#拼接最后一维


def box_xyxy_to_cxcywh(x):#角点格式到中心点格式
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2, test=False):#IoU值为预测框与真实框的重合程度IoU=交集区域/并集区域
# torchvison 是用来训练图像类任务的，提供了图像预处理和增强操作
    area1 = box_area(boxes1)#计算面积
    area2 = box_area(boxes2)#计算面积

    # lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]左上角
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]右下角

    wh = (rb - lt).clamp(min=0)  # [N,2]防止为负
    # inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    inter = wh[:, 0] * wh[:, 1]  # [N]交集面积

    # union = area1[:, None] + area2 - inter
    union = area1 + area2 - inter #并集面积

    iou = inter / union #IoU值

    if test:#当 test=True 时，才执行这个逻辑。一般在测试集或伪数据检测时使用，避免因特殊样本导致计算异常。
        zero_lines = boxes2==torch.zeros_like(boxes2)#检测是否为空框
        zero_lines_idx = torch.where(zero_lines[:,0]==True)[0]

        for idx in zero_lines_idx:
            if all(boxes1[idx,:] < 1e-4):
                iou[idx]=1

    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    iou, union = box_iou(boxes1, boxes2)

    # lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    # rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # area = wh[:, :, 0] * wh[:, :, 1]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area
