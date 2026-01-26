from collections.abc import Callable

from torch import Tensor, load
from torch.utils.data import Dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    """
    Custom PyTorch Dataset to load the pre-processed MSNIST tensors from disk.
    """

    def __init__(
        self,
        file_path: str,
        transform: Callable | None = None,
    ) -> None:
        """
        Args:
            `file_path`: Path to the .pt file containing the data and targets.
            `transform`: Optional callable transform to apply to the data.
        """
        self.data, self.targets = load(file_path)
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """
        Retrieves the image and label at the specified index.

        Args:
            `idx`: The index of the item to retrieve.

        Returns:
            A tuple containing the transformed image tensor and its label.
        """
        img = self.data[idx]
        target = int(self.targets[idx])

        img = transforms.ToPILImage()(img)

        if self.transform:
            img = self.transform(img)

        return img, target

    pass
