from Interfaces.dataset_loader_interface import DatasetLoader
from typing import List, Union, Any, Optional, Tuple
from torch.utils.data import random_split, Dataset
import os

from Utils.yolo_dataset import YOLOFolderDataset


class RobustDatasetLoader():
    def __init__(self,
                 source: Union[str, List[str]],
                 split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
        """
        :param source: Single path to the data, or multiple paths if data is already split
        :param split_ratios: Used if a single path is provided, train-val-test split
        """
        self.source = source
        self.split_ratios = split_ratios

    def load_data(self) -> List[Dataset]:
        """
        Data loader function, chooses how the to load the data based on the number of provided paths
        :return:
        """
        if isinstance(self.source, list):
            return self._load_multiple_paths(self.source)
        elif isinstance(self.source, str):
            return self._load_single_path(self.source)
        else:
            raise TypeError("Invalid source type")

    def _load_multiple_paths(self, paths: List[str]) -> List[Optional[Dataset]]:
        """
        Helper function to load from multiple paths
        :param paths: List of paths to the train, val, test folders
        :return: A list of the loaded files in train, val, optional(test) combination
        """
        print(f"Detected {len(paths)} folders.\n")

        if len(paths) == 3:
            train: Dataset = self._load_from_disk(paths[0])
            val: Dataset = self._load_from_disk(paths[1])
            test: Dataset = self._load_from_disk(paths[2])

            return [train, val, test]

        elif len(paths) == 2 and "val" in paths[1].lower():
            train: Dataset = self._load_from_disk(paths[0])
            val: Dataset = self._load_from_disk(paths[1])

            return [train, val, None]

        elif len(paths) == 2 and "test" in paths[1].lower():
            train: Dataset = self._load_from_disk(paths[0])
            test: Dataset = self._load_from_disk(paths[1])

            return [train, None, test]

        else:
            raise TypeError("Invalid number of paths")

    def _load_single_path(self, path: str) -> List[Optional[Dataset]]:
        """
        Helper function to load from a single path
        :param path: Path to the dataset
        :return:
        """
        print(f"Detected single source: {path}.\n")

        full_dataset: Dataset = self._load_from_disk(path)
        total_count: int = len(full_dataset)
        train_size: int = int(total_count * self.split_ratios[0])
        val_size: int = int(total_count * self.split_ratios[1])
        test_size: int = total_count - train_size - val_size

        return list(random_split(full_dataset, [train_size, val_size, test_size]))

    def _load_from_disk(self, path: str) -> Dataset:
        """
        Helper function to load from a single path
        :param path:
        :return:
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found.")
        return YOLOFolderDataset(root = path, transform = None)