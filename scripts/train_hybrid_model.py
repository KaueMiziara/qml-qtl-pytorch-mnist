import torch
import torch.nn as nn
import torch.optim as optim

from qml_qtl_pytorch_mnist.data import DataLoader
from qml_qtl_pytorch_mnist.models.backbone import QuantumBackbonePennyLane
from qml_qtl_pytorch_mnist.models.classifier import Classifier
from qml_qtl_pytorch_mnist.visualization import Visualizer

if __name__ == "__main__":
    LEARNING_RATE = 0.001
    EPOCHS = 15
    BATCH_SIZE = 4

    loader_factory = DataLoader(batch_size=BATCH_SIZE)
    train_loader = loader_factory.get_data_loader(is_train=True)

    model = Classifier(n_quantum_layers=2, backbone_cls=QuantumBackbonePennyLane)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    print(f"Starting HYBRID training for {EPOCHS} epochs...")
    epoch_losses = []

    model.train()
    for epoch in range(EPOCHS):
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
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    visualizer = Visualizer()
    visualizer.save_loss_curve(
        epochs=list(range(1, EPOCHS + 1)),
        losses=epoch_losses,
        filename="hybrid_training_loss.png",
    )

    save_path = "./data/results/hybrid_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Hybrid model saved to {save_path}")

    print("\n--- Model Architecture ---")
    print(model)
