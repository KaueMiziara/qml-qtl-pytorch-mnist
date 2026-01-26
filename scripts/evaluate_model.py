import os

import torch

from qml_qtl_pytorch_mnist.data import DataLoader
from qml_qtl_pytorch_mnist.metrics import Evaluator
from qml_qtl_pytorch_mnist.model.classifier import ClassicalClassifier

if __name__ == "__main__":
    BATCH_SIZE = 4
    MODEL_PATH = "./data/results/classical_model.pth"

    loader_factory = DataLoader(batch_size=BATCH_SIZE)
    test_loader = loader_factory.get_data_loader(is_train=False)

    model = ClassicalClassifier()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded model weights from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Modelfile not found at {MODEL_PATH}. Train first.")

    evaluator = Evaluator(model)
    metrics = evaluator.evaluate(test_loader)

    LINE_LEN = 40
    print("\n" + "=" * LINE_LEN)
    print("\tMODEL EVALUATION")
    print("=" * LINE_LEN)
    print(f"Accuracy:\t{metrics.accuracy:.2%}")
    print(f"Average Loss:\t{metrics.loss:.4f}")
    print("-" * LINE_LEN)
    print("Confusion Matrix:")
    print(f"True Positives  (1 as 1):\t{metrics.tp}")
    print(f"True Negatives  (0 as 0):\t{metrics.tn}")
    print(f"False Positives (0 as 1):\t{metrics.fp}")
    print(f"False Negatives (1 as 0):\t{metrics.fn}")
    print("=" * LINE_LEN + "\n")
