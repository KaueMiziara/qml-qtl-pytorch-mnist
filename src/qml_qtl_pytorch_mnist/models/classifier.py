import torch
import torch.nn as nn

from .backbone import BaseBackbone
from .post_net import ClassicalPostNet
from .pre_net import ClassicalPreNet


class Classifier(nn.Module):
    """
    A modular Classifier for the MIST dataset that can switch between
    Classical and Quantum backbones via dependency injection.
    """

    def __init__(
        self,
        backbone_cls: type[BaseBackbone],
        n_quantum_layers: int = 2,
    ) -> None:
        """
        Initializes the hybrid model components.

        Args:
            `n_quantum_layers`: Depth of the quantum backbone ansatz.
            `backbone_cls`: The class definition of the backbone to use.
        """
        super().__init__()

        self.pre_net = ClassicalPreNet(input_dim=16, output_dim=4)

        self.backbone = backbone_cls(
            input_dim=4,
            output_dim=2,
            n_layers=n_quantum_layers,
        )

        self.post_net = ClassicalPostNet(input_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the hybrid model.

        1. Flattens the 2D image batch (B, 1, 4, 4) to (B, 16).
        2. Encodes and scales data via Pre-Net.
        3. Processes features via Backbone.
        4. Classifies result via Post-Net.

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
