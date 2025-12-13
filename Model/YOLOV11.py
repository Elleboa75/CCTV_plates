import torch
from torch.utils.data import DataLoader, Dataset
from Utils.dataset_loader import RobustDatasetLoader
from Utils.yolo_dataset import YOLOFolderDataset
from typing import Any, Tuple

_path_to_train = "Data/Dataset/train"
_path_to_val = "Data/Dataset/valid"
_path_to_test = "Data/Dataset/test"

def training_pipeline(device: torch.device, batch_size: int = 16, workers: int = 16):
    """
    :param device: The target `torch.device` (e.g., 'cuda' or 'cpu') where tensors should be loaded.
    :param batch_size: The number of image samples to process in a single forward/backward pass.
    :param workers: The number of background subprocesses to spawn for parallel data loading.
    :return: A tuple containing the configured `train_loader` and `val_loader` ready for the training loop.
    """
    loader: RobustDatasetLoader = RobustDatasetLoader(
        source = [_path_to_train, _path_to_val, _path_to_test],
    )

    train_set, val_set, test_set = loader.load_data()

    print(f"Initializing Train Loader with {len(train_set)} images...")
    train_loader = DataLoader(
        dataset = train_set,
        batch_size = batch_size,
        shuffle = True,
        num_workers = workers,
        pin_memory = True,
        collate_fn = YOLOFolderDataset.collate_fn
    )

    print(f"Initializing Validation Loader with {len(val_set)} images...")
    val_loader = DataLoader(
        dataset = val_set,
        batch_size = batch_size,
        shuffle = False,
        num_workers = workers,
        pin_memory = True,
        collate_fn = YOLOFolderDataset.collate_fn
    )

    print(f"Initializing Test Loader with {len(test_set)} images...")
    test_loader = DataLoader(
        dataset = test_set,
        batch_size = batch_size,
        shuffle = False,
        num_workers = workers,
        pin_memory = True,
        collate_fn = YOLOFolderDataset.collate_fn
    )

    return train_loader,  val_loader, test_loader
