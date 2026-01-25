import os

from torch import save
from torchvision import datasets


class DataProcessor:
    """
    Responsible for downloading RAW data, filtering specific classes (0 and 1),
    and saving the processed dataset to disk.
    """

    RAW_PATH = "./data/raw"
    PROCESSED_PATH = "./data/results"

    def __init__(self, batch_size=4) -> None:
        os.makedirs(self.RAW_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_PATH, exist_ok=True)

        self._batch_size = batch_size

    # def get_mnist_loader(self, is_train=False) -> DataLoader:
    #     dataset = self._get_dataset(is_train=is_train)
    #     return DataLoader(
    #         dataset,
    #         batch_size=self._batch_size,
    #         shuffle=True,
    #     )

    def process_and_save(self):
        self._process_subset(is_train=True)
        self._process_subset(is_train=False)

    def _process_subset(self, is_train: bool):
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

    # def _get_transform(self):
    #     return transforms.Compose(
    #         [
    #             transforms.Resize((4, 4)),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5,), (0.5,)),
    #         ]
    #     )
