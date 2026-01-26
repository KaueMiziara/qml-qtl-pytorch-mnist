import os

from torch import stack
from torchvision import transforms

from qml_qtl_pytorch_mnist.data import DataLoader, Dataset
from qml_qtl_pytorch_mnist.visualization import Visualizer

if __name__ == "__main__":
    batch_size = 4
    loader_factory = DataLoader(batch_size=batch_size)
    visualizer = Visualizer()

    test_loader = loader_factory.get_data_loader(is_train=False)

    reduced_imgs, reduced_labels = next(iter(test_loader))

    test_file_path = os.path.join(DataLoader.PROCESSED_PATH, "test.pt")

    raw_transform = transforms.ToTensor()
    raw_dataset = Dataset(file_path=test_file_path, transform=raw_transform)

    original_imgs = []

    for i in range(batch_size):
        img, _ = raw_dataset[i]
        original_imgs.append(img)

    original_imgs_tensor = stack(original_imgs)

    print(f"Original Shape: {original_imgs_tensor.shape}")
    print(f"Reduced Shape:  {reduced_imgs.shape}")

    visualizer.save_data_comparison(
        original_imgs=original_imgs_tensor,
        reduced_imgs=reduced_imgs,
        labels=reduced_labels,
        filename="input_resolution_comparison.png",
    )
