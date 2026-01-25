from qml_qtl_pytorch_mnist.data import DataProcessor

if __name__ == "__main__":
    # ETL
    processor = DataProcessor()

    print("Processing data...")
    processor.process_and_save()
    print(f"Data processed and saved to {processor.PROCESSED_PATH}")
