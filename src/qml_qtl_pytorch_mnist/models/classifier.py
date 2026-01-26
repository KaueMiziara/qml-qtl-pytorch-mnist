import torch.nn as nn
from torch import Tensor

from .backbone.classical_backbone import ClassicalBackbone
from .post_net import ClassicalPostNet
from .pre_net import ClassicalPreNet


class ClassicalClassifier(nn.Module):
    """
    The main classification model that orchestrates the data flow through
    the Pre-Net, Backbone, and Post-Net.

    This architecture is designed to be modular: the ClassicalBackbone can
    be swapped for a QuantumBackbone in future iterations without changing
    the input processing or final decision logic.
    """

    def __init__(self) -> None:
        """
        Initializes the classifier by composing its three main sub-modules.
        """
        super().__init__()
        self.pre_net = ClassicalPreNet(input_dim=16, output_dim=4)
        self.backbone = ClassicalBackbone(input_dim=4, output_dim=2)
        self.post_net = ClassicalPostNet(input_dim=2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the model.

        1. Flattens the 2D image batch (B, 1, 4, 4) to (B, 16).
        2. Encodes and scales data via Pre-Net.
        3. Processes features via Backbone (Simulated Circuit).
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
