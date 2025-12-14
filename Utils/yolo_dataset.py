import os
import glob
import numpy as np
import cv2
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any, Optional
import torch

class YOLOFolderDataset(Dataset):
    def __init__(self, root: str, transform: None) -> None:
        """
        Pytorch Dataset for YOLO dataset fromat loading.
        :param root: Path to the folder containing the images and labels.
        :param transform: transformation applied to the images and labels, default function provided, can be overloaded.
        """
        self.root = root
        self.transform = transform

        self.image_paths = sorted(
            glob.glob(os.path.join(self.root, "images", "*.jpg")) +
            glob.glob(os.path.join(self.root, "images", "*.png")) +
            glob.glob(os.path.join(self.root, "images", "*.tif")) +
            glob.glob(os.path.join(self.root, "images", "*.jpeg"))
        )

        if len(self.image_paths) == 0:
            print(f"No images found in {self.root}")
            self.image_paths = sorted(
                glob.glob(os.path.join(self.root, "*.jpg")) +
                glob.glob(os.path.join(self.root, "*.png")) +
                glob.glob(os.path.join(self.root, "*.tif")) +
                glob.glob(os.path.join(self.root, "*.jpeg"))
            )
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        image_path: str = self.image_paths[index]
        image: np.ndarray = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image {image_path} could not be read.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_height: int
        original_width: int
        original_height, original_width = image.shape[:2]

        scale: float
        padding_information: Tuple[float, float]
        image, scale, padding_information = self.pad_image(image, target_size = (640, 640))
        padded_left, padded_top = padding_information

        label_path: str = image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"

        boxes: List[List[float]] = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    try:
                        data: List[float] = [float(x) for x in line.strip().split()]
                        if len(data) == 5:
                            #Recalculate Labels due to padding
                            class_id, x, y, width, height = data
                            # 1. Un-normalize
                            x_abs: float = x * original_width
                            y_abs: float = y * original_height
                            w_abs: float = width * original_width
                            h_abs: float = height * original_height

                            # 2. Scale
                            x_abs *= scale
                            y_abs *= scale
                            w_abs *= scale
                            h_abs *= scale

                            # 3. Shift (Padding)
                            x_abs += padded_left
                            y_abs += padded_top

                            # 4. Re-normalize to NEW size (640x640)
                            new_x: float = x_abs / 640.0
                            new_y: float = y_abs / 640.0
                            new_w: float = w_abs / 640.0
                            new_h: float = h_abs / 640.0

                            boxes.append([class_id, new_x, new_y, new_w, new_h])
                    except ValueError:
                        print(f"Line {line} is malformed.\n")

        target: torch.Tensor = torch.tensor(boxes, dtype = torch.float32)

        if self.transform is None:
            image: np.ndarray = image.astype(np.float32) / 255.0
            image: torch.Tensor = torch.from_numpy(image).permute(2, 0, 1)
        else:
            try:
                augmented: Dict[str, Any] = self.transform(image = image, bboxes = boxes)
                image: torch.Tensor = augmented["image"]
                target: torch.Tensor = torch.tensor(augmented["bboxes"], dtype = torch.float32)
            except Exception as e:
                raise RuntimeError(f"Transformation failed on {image_path}: {e}")

        return image, target

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        images, targets = zip(*batch)
        images = torch.stack(images, dim = 0)

        return images, list(targets)

    @staticmethod
    def pad_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
            Resizes an image to target_size while maintaining aspect ratio (padding with gray).

            Args:
                img: The source image (H, W, C).
                target_size: The desired (width, height).

            Returns:
                padded_img: The resized and padded image.
                scale: The resize scale factor used (e.g., 0.5 for half size).
                (pad_left, pad_top): The padding pixels added to the left and top.
            """
        h, w = image.shape[:2]
        target_w, target_h = target_size

        scale: float = min(target_h / h, target_w / w)
        new_width: int = int(w * scale)
        new_height: int = int(h * scale)

        resized_image: np.ndarray = cv2.resize(image, (new_width, new_height))

        # Padding calculations
        pad_width: float = (target_w - new_width) / 2
        pad_height: float = (target_h - new_height) / 2

        top: int = int(pad_height)
        bottom: int = int(pad_height + 0.5)
        left: int = int(pad_width)
        right: int = int(pad_width + 0.5)

        padded_image: np.ndarray = cv2.copyMakeBorder(
            resized_image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value = (110, 110, 110)
        )

        return padded_image, scale, (left, top)