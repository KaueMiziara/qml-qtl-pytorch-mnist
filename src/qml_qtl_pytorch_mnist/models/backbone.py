import torch
import torch.nn as nn


class ClassicalBackbone(nn.Module):
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
    ) -> None:
        """
        Initializes the backbone layer.

        Args:
            `input_dim`: The number of input features (corresponding to input qubits).
            `output_dim`: The number of output measurements.
        """
        super().__init__()
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
