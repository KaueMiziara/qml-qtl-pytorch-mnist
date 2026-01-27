import torch
import torch.nn as nn

from .base_backbone import BaseBackbone


class ClassicalBackbone(BaseBackbone):
    """
    Simulates the variational quantum circuit (Ansatz) using a classical linear layer.

    In the hybrid model, this component will be replaced by a parameterized quantum
    circuit that performs rotations and entanglements, followed by a measurement
    that produces real-valued outputs.
    """

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 2,
        n_layers: int = 1,
    ) -> None:
        """
        Initializes the backbone layer.

        Args:
            `input_dim`: The number of input features (corresponding to input qubits).
            `output_dim`: The number of output measurements.
            `n_layers`: Accepted for compatibility but ignored.
        """
        super().__init__(input_dim, output_dim, n_layers)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a linear transformation to simulate the parameterized circuit
        and measurement process.

        Args:
            `x`: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.linear(x)
