from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseBackbone(nn.Module, ABC):
    """
    Abstract Base Class for all backbones (Classical & Quantum).
    Enforces a strict interface so the Classifier wrapper can swap them seamlessly.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Standardizes initialization.

        Args:
            `input_dim`: Number of input features/qubits.
            `output_dim`: Number of output features/measurements.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Must return a Tensor of shape (batch_size, output_dim).
        """
        pass
