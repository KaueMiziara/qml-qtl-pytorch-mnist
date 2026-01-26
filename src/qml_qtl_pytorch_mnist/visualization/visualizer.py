import os

import matplotlib.pyplot as plt
from torch import Tensor


class Visualizer:
    """
    Handles the generation and saving of visual artifacts for the project.
    """

    PLOTS_PATH: str = "./plots"

    def __init__(self) -> None:
        """
        Initializes the Visualizer and ensures the output directories exist.
        """
        os.makedirs(self.PLOTS_PATH, exist_ok=True)

    def save_data_comparison(
        self,
        original_imgs: Tensor,
        reduced_imgs: Tensor,
        labels: Tensor,
        filename: str = "data_comparison.png",
    ) -> None:
        """
        Generates a side-by-side comparison of the original (28x28) images
        and the reduced (4x4) images to validate distinguishability.

        Args:
            `original_imgs`: Batch of original images (Batch, C, H, W).
            `reduced_imgs`: Batch of transformed images (Batch, C, H, W).
            `labels`: Batch of corresponding labels.
            `filename`: The name of the file to save within the plots directory.
        """
        batch_size = len(labels)
        fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 2, 5))

        if batch_size == 1:
            axes = axes.reshape(2, 1)

        for i in range(batch_size):
            ax_orig = axes[0, i]
            img_orig = original_imgs[i].squeeze().numpy()
            ax_orig.imshow(img_orig, cmap="gray")
            ax_orig.set_title(f"Orig: {labels[i].item()}")
            ax_orig.axis("off")

            ax_red = axes[1, i]
            img_red = reduced_imgs[i].squeeze().numpy()
            ax_red.imshow(img_red, cmap="gray")
            ax_red.set_title(f"4x4: {labels[i].item()}")
            ax_red.axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.PLOTS_PATH, filename)
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Comparison plot saved to: {save_path}")

    def save_loss_curve(
        self,
        epochs: list[int],
        losses: list[float],
        filename: str = "training_loss.png",
    ) -> None:
        """
        Plots the training loss curve and saves it to the plots directory.

        Args:
            `epochs`: List of epoch numbers.
            `losses`: List of loss values corresponding to the epochs.
            `filename`: The name of the file to save.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(
            epochs,
            losses,
            marker="o",
            linestyle="-",
            color="b",
            label="Training Loss",
        )
        plt.title("Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        saved_path = os.path.join(self.PLOTS_PATH, filename)
        plt.savefig(saved_path)
        plt.close()
        print(f"Loss curve saved to: {saved_path}")
