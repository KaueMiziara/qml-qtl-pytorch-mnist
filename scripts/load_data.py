from qml_qtl_pytorch_mnist.data import DataProcessor

if __name__ == "__main__":
    dataloader = DataProcessor()
    train_loader = dataloader.get_mnist_loader()
    test_loader = dataloader.get_mnist_loader(is_train=True)

    images, labels = next(iter(test_loader))
    print(f"Image batch size: {images.shape}")
