import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Evaluator:
    """
    Handles the evaluation of a PyTorch model on a given dataset.
    Calculates accuracy and confusion matrix metrics (TP, TN, FP, FN).
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Args:
            `model`: The trained PyTorch model to evaluate.
        """
        self.model = model

    def evaluate(
        self,
        data_loader: DataLoader,
    ) -> dict[str, float]:
        """
        Runs inference on the provided data loader and computes metrics.

        Args:
            `data_loader`: The DataLoader containing test data.

        Returns:
            A dictionary containing:
            - accuracy: The fraction of correct predictions.
            - loss: The average loss over the dataset.
            - TP, TN, FP, FN: Confusion matrix counts.
        """
        self.model.eval()
        criterion = nn.BCELoss()

        total_loss = 0.0
        correct = 0
        total = 0

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        with torch.no_grad():
            for images, labels in data_loader:
                outputs = self.model(images)

                labels = labels.float().view(-1, 1)

                loss = criterion(outputs, labels)
                total_loss += loss.items()

                predicted = (outputs > 0.5).float()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                true_positive += ((predicted == 1) & (labels == 1)).sum().item()
                true_negative += ((predicted == 0) & (labels == 0)).sum().item()
                false_positive += ((predicted == 1) & (labels == 0)).sum().item()
                false_negative += ((predicted == 0) & (labels == 1)).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total

        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "TP": true_positive,
            "TN": true_negative,
            "FP": false_positive,
            "FN": false_negative,
        }
