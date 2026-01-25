from qml_qtl_pytorch_mnist.data import MNISTDataLoader


if __name__ == "__main__":
    dataloader = MNISTDataLoader()
    train_loader = dataloader.get_mnist_loader()
    test_loader = dataloader.get_mnist_loader(is_train=True)

    images, labels = next(iter(test_loader))
    print(f"Image batch size: {images.shape}")
