import os
import glob
import numpy as np
import cv2
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image {image_path} could not be read.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    try:
                        data = [float(x) for x in line.strip().split()]
                        if len(data) == 5:
                            boxes.append(data)
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

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        images, targets = zip(*batch)
        images = torch.stack(images, dim = 0)

        return images, list(targets)