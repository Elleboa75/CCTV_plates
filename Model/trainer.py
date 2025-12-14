import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torchgen import model
from tqdm import tqdm
import numpy as np
from YOLOV11 import training_pipeline
import math
from typing import Tuple, List, Dict, Any, Optional
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from types import SimpleNamespace

# HYPERPARAMETERS
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 50
WEIGHT_DECAY = 1e-4
CFG_FILE = "yolo11n.yaml"


def preprocess_batch(images: torch.Tensor, targets: List[torch.Tensor]) -> Dict[str, Any]:
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

        batch_index: torch.Tensor = torch.full((n_boxes, 1), idx, dtype = torch.long, device = DEVICE)

        all_idx.append(batch_index)
        all_classes.append(target[:, 0:1].to(DEVICE))
        all_bboxes.append(target[:, 1:5].to(DEVICE))

    batch_idx: torch.Tensor
    batch_class: torch.Tensor
    batch_bboxes: torch.Tensor

    if len(all_idx) > 0:
        batch_idx = torch.cat(all_idx, dim = 0)
        batch_class = torch.cat(all_classes, dim = 0)
        batch_bboxes = torch.cat(all_bboxes, dim = 0)
    else:
        batch_idx = torch.zeros((0, 1), device = DEVICE)
        batch_class = torch.zeros((0, 1), device = DEVICE)
        batch_bboxes = torch.zeros((0, 4), device = DEVICE)

    return {
        "img": images.to(DEVICE),
        "batch_idx": batch_idx.view(-1),
        "cls": batch_class.view(-1, 1),
        "bboxes": batch_bboxes
    }

def train_one_epoch(loader, model, optimizer, scaler, loss_fn):
    model.train()
    loop = tqdm(loader, leave = True)

    mean_loss: List[float] = []

    for batch_idx, (images, targets) in enumerate(loop):

        batch_dict = preprocess_batch(images, targets)
        optimizer.zero_grad()

        with autocast(device_type = 'cuda'):

            preds = model(batch_dict["img"])
            loss, loss_items = loss_fn(preds, batch_dict)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss.append(loss.item())
        loop.set_postfix(loss = loss.item())
    return sum(mean_loss) / len(mean_loss)

def evaluate(loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader, desc = "Validating", leave = True)
    mean_loss = []

    with torch.no_grad():
        for images, targets in loop:
            batch_dict = preprocess_batch(images, targets)

            with autocast(device_type = 'cuda'):
                preds = model(batch_dict["img"])
                loss, _ = loss_fn(preds, batch_dict)
            mean_loss.append(loss.item())

    return sum(mean_loss) / len(mean_loss)

def main():
    train_loader, val_loader, test_loader = training_pipeline(
        device = DEVICE,
        batch_size = BATCH_SIZE,
        workers = 16,
    )
    print(f"Initializing model {CFG_FILE}...")
    model = (DetectionModel(CFG_FILE, nc = 1))
    model.to(DEVICE)

    #Default Ultralytics values
    model.args = SimpleNamespace(
        box = 7.5,  # Box loss gain
        cls = 0.5,  # Class loss gain (scaled by pixel count)
        dfl = 1.5,  # Distribution Focal Loss gain
        pose = 12.0,  # (Not used for detection, but required to prevent crash)
        kobj = 1.0,  # (Not used, but required)

        # Training settings the loss might peek at
        overlap_mask = True,
        mask_ratio = 4,
        dropout = 0.0,
        val = False
    )

    criterion = v8DetectionLoss(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    )
    scaler = GradScaler()
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}\n")

        train_loss = train_one_epoch(train_loader, model, optimizer, scaler, criterion)
        val_loss = evaluate(val_loader, model, criterion)

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model.pth")


if __name__ == "__main__":
    main()