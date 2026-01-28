import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from qml_qtl_pytorch_mnist.data.data_loader import MNISTDataLoader
from qml_qtl_pytorch_mnist.metrics.evaluation import Evaluator
from qml_qtl_pytorch_mnist.models.backbone import (
    BaseBackbone,
    ClassicalBackbone,
    QuantumBackbonePennyLane,
    QuantumBackboneQiskit,
)
from qml_qtl_pytorch_mnist.models.classifier import Classifier

RESULTS_FILE = "./data/results/benchmark_data.csv"
PLOTS_DIR = "./plots"
sns.set_theme(style="whitegrid")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_log(
    model_name: str,
    backbone_cls: type[BaseBackbone],
    learning_rate: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    n_layers: int,
) -> tuple[list[dict], float, float]:
    print(f"\n--- Starting Benchmark: {model_name} ---")

    set_seed(42)

    model = Classifier(backbone_cls=backbone_cls, n_quantum_layers=n_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    history = []
    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.float().view(-1, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        history.append(
            {
                "Model": model_name,
                "Epoch": epoch + 1,
                "Loss": avg_loss,
                "Time": epoch_time,
            }
        )

    total_time = time.time() - start_time

    evaluator = Evaluator(model)
    metrics = evaluator.evaluate(test_loader)
    print(
        f"Finished {model_name}. Accuracy: {metrics.accuracy:.2%} | "
        f"Total Time: {total_time:.1f}s"
    )

    return history, metrics.accuracy, total_time


def generate_plots(df_history: pd.DataFrame, df_results: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_history,
        x="Epoch",
        y="Loss",
        hue="Model",
        marker="o",
        linewidth=2,
    )
    plt.title("Convergence Comparison: Classical vs Quantum Backbones", fontsize=14)
    plt.ylabel("Binary Cross Entropy Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOTS_DIR}/benchmark_loss_curve.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(
        data=df_results,
        x="Model",
        y="Total Training Time (s)",
        ax=axes[0],
        hue="Model",
        palette="viridis",
    )
    axes[0].set_title("Computational Cost (Training Time)", fontsize=12)
    axes[0].set_ylabel("Seconds (Lower is Better)")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f s")

    sns.barplot(
        data=df_results,
        x="Model",
        y="Final Accuracy",
        ax=axes[1],
        hue="Model",
        palette="magma",
    )
    axes[1].set_title("Model Effectiveness (Accuracy)", fontsize=12)
    axes[1].set_ylabel("Accuracy (Higher is Better)")
    axes[1].set_ylim(0, 1.1)  # Scale 0 to 100%
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/benchmark_metrics.png", dpi=300)
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    EPOCHS = 3
    SUBSET_SIZE = 200  # Set to None for full run
    BATCH_SIZE = 4

    loader_factory = MNISTDataLoader(batch_size=BATCH_SIZE)

    train_loader = loader_factory.get_data_loader(is_train=True)
    test_loader = loader_factory.get_data_loader(is_train=False)

    if SUBSET_SIZE:
        print(f"WARNING: Running on a subset of {SUBSET_SIZE} samples for speed.")

        full_train_ds = train_loader.dataset
        full_test_ds = test_loader.dataset

        indices = list(range(SUBSET_SIZE))

        g = torch.Generator()
        g.manual_seed(42)

        train_loader = DataLoader(
            Subset(full_train_ds, indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=g,
        )
        test_loader = DataLoader(
            Subset(full_test_ds, indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

    models_config = [
        ("Classical (Baseline)", ClassicalBackbone),
        ("PennyLane (Qubit)", QuantumBackbonePennyLane),
        ("Qiskit (Estimator)", QuantumBackboneQiskit),
    ]

    all_history = []
    final_results = []

    for name, backbone in models_config:
        try:
            hist, acc, duration = train_and_log(
                model_name=name,
                backbone_cls=backbone,
                learning_rate=0.01,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=EPOCHS,
                n_layers=2,
            )
            all_history.extend(hist)
            final_results.append(
                {
                    "Model": name,
                    "Final Accuracy": acc,
                    "Total Training Time (s)": duration,
                }
            )
        except Exception as e:
            print(f"ERROR running {name}: {e}")

    df_history = pd.DataFrame(all_history)
    df_results = pd.DataFrame(final_results)

    df_history.to_csv(RESULTS_FILE, index=False)
    print(f"\nRaw data saved to {RESULTS_FILE}")

    generate_plots(df_history, df_results)
