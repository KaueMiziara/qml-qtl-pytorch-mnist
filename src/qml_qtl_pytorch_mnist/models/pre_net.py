import numpy as np
import torch
import torch.nn as nn


class ClassicalPreNet(nn.Module):
    """
    The Pre-Net layer responsible for compressing the input data and rescaling
    it to a range suitable for the subsequent layers.

    In the final hybrid architecture, this prepares data for Angle Embedding
    (scaling to -pi to pi). For the classical baseline, it acts as a standard
    feature extraction and normalization layer.
    """

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 4,
    ) -> None:
        """
        Initializes the linear transformation layer.

        Args:
            `input_dim`: The number of input features (pixels).
            Defaults to 16 (4x4 flattened).
            `output_dim`: The number of output features (or qubits). Defaults to 4.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation followed by a Tanh activation and
        scaling by pi/2.

        Args:
            `x`: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim) with values
            scaled approximately between -pi/2 and pi/2.
        """
        x = self.linear(x)
        x = torch.tanh(x)
        x = x * (np.pi / 2.0)

        return x
