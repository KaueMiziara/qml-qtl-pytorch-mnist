from qml_qtl_pytorch_mnist.classical.data import Data


if __name__ == "__main__":
    dataset = Data()
    train_loader = dataset.get_mnist_loader()
    test_loader = dataset.get_mnist_loader(is_train=True)

    images, labels = next(iter(test_loader))
    print(f"Image batch size: {images.shape}")
