from qml_qtl_pytorch_mnist.data import DataLoader, DataProcessor

if __name__ == "__main__":
    # ETL
    processor = DataProcessor()

    print("Processing data...")
    processor.process_and_save()
    print(f"Data processed and saved to {processor.PROCESSED_PATH}")

    loader_factory = DataLoader(batch_size=4)
    train_loader = loader_factory.get_data_loader(is_train=True)

    images, labels = next(iter(train_loader))

    print("\n--- Data Verification ---")
    print(f"Batch Tensor Shape: {images.shape}")
    print(f"Labels in batch: {labels}")
    print(f"Value Range: min={images.min():.2f}, max={images.max():.2f}")
