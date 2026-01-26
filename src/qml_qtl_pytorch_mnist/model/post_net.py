import torch
import torch.nn as nn


class ClassicalPostNet(nn.Module):
    """
    The final decision layer (Post-Net).

    It takes the output from the backbone (simulating quantum measurements)
    and maps it to a single probability value using a linear transformation
    followed by a Sigmoid activation.
    """

    def __init__(self, input_dim: int = 2) -> None:
        """
        Initializes the post-net layer.

        Args:
            `input_dim`: The number of inputs coming from the backbone.
            Defaults to 2 (e.g., expectation values from 2 qubits).
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the classification probability.

        Args:
            `x`: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, 1) containing probabilities
            in the range [0, 1].
        """
        x = self.linear(x)
        return self.sigmoid(x)
