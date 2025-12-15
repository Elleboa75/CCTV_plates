import numpy as np
import torchvision
import torch
import time
from typing import List, Dict, Any
from ultralytics.nn.tasks import DetectionModel
from types import SimpleNamespace

def load_model_from_file(path: str, device: str, cfg_file: str) -> torch.nn.Module:
    """
    Loads a trained YOLO model from a .pth file.
    """
    print(f"Loading model from {path}...")

    model = DetectionModel(cfg = cfg_file, nc = 1)

    state_dict = torch.load(path, map_location = device)
    model.load_state_dict(state_dict)

    model.to(device)

    model.args = SimpleNamespace(
        box = 7.5,
        cls = 0.5,
        dfl = 1.5,
        pose = 12.0,
        kobj = 1.0,
        overlap_mask = True,
        mask_ratio = 4,
        dropout = 0.0,
        val = False
    )

    model.eval()  # Set to evaluation mode by default
    return model

def preprocess_batch(images: torch.Tensor, targets: List[torch.Tensor], device) -> Dict[str, Any]:
    """
    :Prepares the batch for the YOLOv8/v11 Loss function.

    :param images: The batch of images (B, C, H, W)
    :param targets: A list where each element is a Tensor of shape (Num_Boxes, 5) for that image.
    :return: A dictionary matching the keys expected by v8DetectionLoss.
    """
    all_idx: List[torch.Tensor] = []
    all_classes: List[torch.Tensor] = []
    all_bboxes: List[torch.Tensor] = []

    for idx, target in enumerate(targets):
        if target is None or len(target) == 0:
            continue

        n_boxes: int = target.shape[0]

        batch_index: torch.Tensor = torch.full((n_boxes, 1), idx, dtype = torch.long, device = device)

        all_idx.append(batch_index)
        all_classes.append(target[:, 0:1].to(device))
        all_bboxes.append(target[:, 1:5].to(device))

    batch_idx: torch.Tensor
    batch_class: torch.Tensor
    batch_bboxes: torch.Tensor

    if len(all_idx) > 0:
        batch_idx = torch.cat(all_idx, dim = 0)
        batch_class = torch.cat(all_classes, dim = 0)
        batch_bboxes = torch.cat(all_bboxes, dim = 0)
    else:
        batch_idx = torch.zeros((0, 1), device = device)
        batch_class = torch.zeros((0, 1), device = device)
        batch_bboxes = torch.zeros((0, 4), device = device)

    return {
        "img": images.to(device),
        "batch_idx": batch_idx.view(-1),
        "cls": batch_class.view(-1, 1),
        "bboxes": batch_bboxes
    }

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, w, h) to (x1, y1, x2, y2).
    x, y are the center of the box.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(
        prediction,
        conf_thres = 0.25,
        iou_thres = 0.45,
        max_det = 300
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Args:
        prediction: (Batch_Size, 5 + nc, Num_Anchors)
                    For your 1-class model: (B, 5, 6400) -> [x, y, w, h, conf]
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum detections per image

    Returns:
        List of detections, one per image. Each is (Num_Dets, 6) -> [x1, y1, x2, y2, conf, cls]
    """

    # 1. Checks
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # Select only inference output

    device = prediction.device
    bs = prediction.shape[0]  # Batch size
    nc = prediction.shape[1] - 4  # Number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 5 + nc  # mask start index

    # 2. Settings
    # limit time to prevent hanging
    time_limit = 0.5 + 0.05 * bs
    t = time.time()

    output = [torch.zeros((0, 6), device = device)] * bs

    # 3. Process each image in the batch
    for xi, x in enumerate(prediction):  # image index, image inference

        # Transpose: (Channels, Anchors) -> (Anchors, Channels)
        # x shape becomes: (Num_Anchors, 5)
        x = x.transpose(0, -1)

        # If output is (5, N), splitting it:
        # box: x[:, :4] (x,y,w,h)
        # cls: x[:, 4:] (conf)

        # Filter by Confidence Score
        # For nc=1, x[:, 4] is the confidence
        xc = x[:, 4] > conf_thres
        x = x[xc]

        # If none remain, process next image
        if x.shape[0] == 0:
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Confidence
        conf = x[:, 4:5]

        # Class (Since you have 1 class, we just make a column of zeros)
        cls = torch.zeros_like(conf)

        # Concatenate: [x1, y1, x2, y2, conf, cls]
        x = torch.cat((box, conf, cls), 1)

        # Check limit
        if x.shape[0] > max_det:
            x = x[x[:, 4].argsort(descending = True)[:max_det]]

        # Batched NMS (Standard torchvision NMS)
        # We pass boxes and scores
        i = torchvision.ops.nms(x[:, :4], x[:, 4], iou_thres)

        # Limit detections
        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            print(f'WARNING NMS time limit {time_limit:.3f}s exceeded')
            break

    return output