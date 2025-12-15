import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torchgen import model
from tqdm import tqdm
from YOLOV11 import training_pipeline
from typing import List, Dict, Any
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from types import SimpleNamespace
from yolo_helpers import non_max_suppression, xywh2xyxy, preprocess_batch, load_model_from_file
from ultralytics.utils.metrics import box_iou
import os

# HYPERPARAMETERS
LEARNING_RATE: float = 1e-4
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE: int = 16
EPOCHS: int = 50
WEIGHT_DECAY: float = 1e-4
CFG_FILE: str = "yolo11n.yaml"
TRAINING_MODE: bool = False
MODEL_PATH: str = "best_model.pth"

def train_one_epoch(loader, model, optimizer, scaler, loss_fn) -> float:
    model.train()
    loop = tqdm(loader, leave = True)

    mean_loss: List[float] = []

    for batch_idx, (images, targets) in enumerate(loop):
        batch_dict = preprocess_batch(images, targets, DEVICE)
        optimizer.zero_grad()

        with autocast(device_type = 'cuda'):
            preds = model(batch_dict["img"])
            loss, loss_items = loss_fn(preds, batch_dict)

        scaler.scale(loss.sum()).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.sum().item()
        mean_loss.append(loss_value)
        loop.set_postfix(loss = loss_value)

    return sum(mean_loss) / len(mean_loss)

def evaluate(loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader, desc = "Validating", leave = True)
    mean_loss = []

    with torch.no_grad():
        for images, targets in loop:
            batch_dict = preprocess_batch(images, targets, DEVICE)

            with autocast(device_type = 'cuda'):
                preds = model(batch_dict["img"])
                loss, _ = loss_fn(preds, batch_dict)
            loss_value = loss.sum().item()
            mean_loss.append(loss_value)

    return sum(mean_loss) / len(mean_loss)

def test(loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader, desc = "Testing", leave = True)

    mean_loss = []
    correct_detections = 0
    total_ground_truths = 0
    total_predictions = 0

    with torch.no_grad():
        for images, targets in loop:
            batch_dict = preprocess_batch(images, targets, DEVICE)

            with autocast(device_type = 'cuda'):
                preds = model(batch_dict["img"])
                loss, _ = loss_fn(preds, batch_dict)
            loss_value = loss.sum().item()
            mean_loss.append(loss_value)
            pred_list = non_max_suppression(preds, conf_thres = 0.25, iou_thres = 0.6)

            for idx, pred in enumerate(pred_list):

                mask = batch_dict['batch_idx'] == idx
                boxes = batch_dict['bboxes'][mask]

                if len(boxes) == 0:
                    total_predictions += len(pred)
                    continue

                boxes = xywh2xyxy(boxes)

                _, _, h, w = images.shape
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h

                total_ground_truths += len(boxes)
                total_predictions += len(pred)

                if len(pred) == 0:
                    continue

                iou_matrix = box_iou(pred[:, :4], boxes)

                # Check if any prediction overlaps a GT by > 50%
                # .max(dim=0) checks "For each GT, what was the best prediction IoU?"
                max_ious, _ = iou_matrix.max(dim = 0)

                # Count how many GTs were "found" (IoU > 0.5)
                detected_count = (max_ious > 0.5).sum().item()
                correct_detections += detected_count

    # Recall = Found / Total Real Plates
    recall = correct_detections / (total_ground_truths + 1e-6)

    # Precision = Found / Total Guesses
    precision = correct_detections / (total_predictions + 1e-6)

    avg_loss = sum(mean_loss) / len(mean_loss) if mean_loss else 0.0

    print(f"\nTest Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Recall (Accuracy): {recall:.2%}")
    print(f"  Precision: {precision:.2%}")

    return recall

def main() -> None:
    train_loader, val_loader, test_loader = training_pipeline(
        device = DEVICE,
        batch_size = BATCH_SIZE,
        workers = 16,
    )
    if TRAINING_MODE:
        print(f"Initializing model {CFG_FILE}...\n")
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
        best_model = None

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}\n")

            train_loss = train_one_epoch(train_loader, model, optimizer, scaler, criterion)
            val_loss = evaluate(val_loader, model, criterion)

            print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_PATH)
                best_model = model
    else:
        if os.path.exists(MODEL_PATH):
            best_model = load_model_from_file(MODEL_PATH, DEVICE, CFG_FILE)
            criterion = v8DetectionLoss(best_model)
        else:
            print(f"Error: Model file {MODEL_PATH} not found. Cannot test.")
            return

    if best_model is not None:
        print("\nRunning Final Test on Best Model...")
        try:
            best_model.eval()
            test_performance = test(test_loader, best_model, criterion)
            print(f"\nFinal Test Recall: {test_performance:.2%}")
        except Exception as e:
            print(f"\nError during testing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
