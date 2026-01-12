import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Data:
    def __init__(self, batch_size=64) -> None:
        self._batch_size = batch_size

    def get_mnist_loader(self, is_train=False) -> DataLoader:
        return DataLoader(
            self._get_dataset(is_train=is_train),
            batch_size=self._batch_size,
            shuffle=True,
        )

    def _get_dataset(self, is_train=False):
        return torchvision.datasets.MNIST(
            root="./data",
            train=is_train,
            download=True,
            transform=self._get_transform(),
        )

    def _get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((4, 4)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    pass
