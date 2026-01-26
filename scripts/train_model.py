import torch
import torch.nn as nn
import torch.optim as optim

from qml_qtl_pytorch_mnist.data import DataLoader
from qml_qtl_pytorch_mnist.model.classifier import ClassicalClassifier
from qml_qtl_pytorch_mnist.visualization import Visualizer

if __name__ == "__main__":
    LEARNING_RATE = 0.01
    EPOCHS = 20
    BATCH_SIZE = 4

    loader_factory = DataLoader(batch_size=BATCH_SIZE)
    train_loader = loader_factory.get_data_loader(is_train=True)

    model = ClassicalClassifier()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    print(f"Starting training for {EPOCHS} epochs...")
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
        filename="classical_training_loss.png",
    )

    saved_path = "./data/results/classical_podel.pth"
    torch.save(model.state_dict(), saved_path)
    print(f"Model saved to {saved_path}")

    print("\n--- Model Architecture ---")
    print(model)
