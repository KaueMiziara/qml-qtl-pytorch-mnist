import os

from torch import save
from torchvision import datasets


class DataProcessor:
    """
    Handles the ETL process: downloads raw MNIST data, filters it to keep only
    digits 0 and 1, and saves the processed tensors to disk.
    """

    RAW_PATH = "./data/raw"
    PROCESSED_PATH = "./data/results"

    def __init__(self) -> None:
        """
        Initializes the processor and ensures the necessary directories exist.
        """
        os.makedirs(self.RAW_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_PATH, exist_ok=True)

    def process_and_save(self) -> None:
        """
        Downloads the raw dataset, processes both training and test subsets,
        and saves the filtered data to the processed directory.
        """
        self._process_subset(is_train=True)
        self._process_subset(is_train=False)

    def _process_subset(self, is_train: bool) -> None:
        """
        Loads the raw subset, filters for labels 0 and 1, and saves the result.

        Args:
            `is_train`: Boolean indicating whether to process the:
                - training set (True)
                - or the test set (False).
        """
        dataset = datasets.MNIST(
            root=self.RAW_PATH,
            train=is_train,
            download=True,
        )

        indices = (dataset.targets == 0) | (dataset.targets == 1)

        filtered_data = dataset.data[indices]
        filtered_targets = dataset.targets[indices]

        filename = "training.pt" if is_train else "test.pt"
        save(
            (filtered_data, filtered_targets),
            os.path.join(self.PROCESSED_PATH, filename),
        )
