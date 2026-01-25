from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Data:
    DATA_PATH = "./data"

    def __init__(self, batch_size=64) -> None:
        self._batch_size = batch_size

    def get_mnist_loader(self, is_train=False) -> DataLoader:
        return DataLoader(
            self._get_dataset(is_train=is_train),
            batch_size=self._batch_size,
            shuffle=True,
        )

    def _get_dataset(self, is_train=False):
        return datasets.MNIST(
            root=self.DATA_PATH,
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
