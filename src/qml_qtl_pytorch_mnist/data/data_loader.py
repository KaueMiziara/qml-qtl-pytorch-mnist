import os

from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import MNISTDataset


class MNISTDataLoader:
    """
    Manages the creation of PyTorch DataLoaders for the processed MNIST dataset.
    Applies resizing and normalization transforms.
    """

    PROCESSED_PATH = "./data/results"

    def __init__(self, batch_size: int = 4) -> None:
        """
        Args:
            `batch_size`: The number of samples per batch to load.
        """
        self._batch_size = batch_size

    def get_data_loader(self, is_train: bool = True) -> DataLoader:
        """
        Creates and returns a DataLoader for the specified subset.

        Args:
            `is_train`: If True, loads the training set; otherwise, loads the test set.

        Returns:
            A PyTorch DataLoader configured with the dataset and transforms.

        Raises:
            `FileNotFoundError`: If the processed data files do not exist.
        """
        filename = "training.pt" if is_train else "test.pt"
        file_path = os.path.join(self.PROCESSED_PATH, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Processed data not found at {file_path}. "
                "Run MNISTDataProcessor().process_and_save() first."
            )

        dataset = MNISTDataset(
            file_path=file_path,
            transform=self._get_transform(),
        )

        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=is_train,
        )

    def _get_transform(self) -> transforms.Compose:
        """
        Defines the sequence of image transformations: resizing to 4x4,
        converting to tensor, and normalizing to range [-1, 1].

        Returns:
            A composed torchvision transform.
        """
        return transforms.Compose(
            [
                transforms.Resize((4, 4)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
