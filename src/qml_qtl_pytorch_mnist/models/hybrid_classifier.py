import torch
import torch.nn as nn

from .backbone.quantum_backbone_pennylane import QuantumBackbonePennyLane
from .post_net import ClassicalPostNet
from .pre_net import ClassicalPreNet


class HybridClassifier(nn.Module):
    """
    The Hybrid Quantum-Classical Classifier.

    Structure:
    1. Pre-Net (Classical): Compresses 4x4 image (16 pixels) to 4 features
       and scales them to the range [-pi/2, pi/2].
    2. Backbone (Quantum): Encodes the 4 features into 4 qubits,
       applies a variational circuit, and measures expectation values.
    3. Post-Net (Classical): Maps the quantum measurements to a final
       classification probability (0 or 1).
    """

    def __init__(self, n_quantum_layers: int = 2) -> None:
        """
        Initializes the hybrid model components.

        Args:
            `n_quantum_layers`: Depth of the quantum ansatz (StronglyEntanglingLayers).
        """
        super().__init__()

        self.pre_net = ClassicalPreNet(input_dim=16, output_dim=4)

        self.backbone = QuantumBackbonePennyLane(
            input_dim=4,
            output_dim=2,
            n_layers=n_quantum_layers,
        )

        self.post_net = ClassicalPostNet(input_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the hybrid model.

        Args:
            `x`: Input tensor of images with shape (batch_size, 1, 4, 4).

        Returns:
            Output probability tensor of shape (batch_size, 1).
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        x = self.pre_net(x)
        x = self.backbone(x)
        x = self.post_net(x)

        return x
