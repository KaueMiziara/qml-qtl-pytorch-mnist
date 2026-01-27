import pennylane as qml
import torch
from pennylane.measurements import MeasurementProcess

from .base_backbone import BaseBackbone


class QuantumBackbonePennyLane(BaseBackbone):
    """
    A Quantum Neural Network (QNN) backbone using PennyLane.

    Structure:
    1. Input: 4 features (from Pre-Net).
    2. Encoding: Angle Embedding (features -> qubit rotations).
    3. Ansatz: StronglyEntanglingLayers (trainable rotations + CNOTs).
    4. Measurement: Expectation value of PauliZ on the first 2 qubits.
    """

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 2,
        n_layers: int = 2,
    ) -> None:
        """
        Initializes the PennyLane quantum layer.

        Args:
            `input_dim`: Number of input features (must match number of qubits).
            `output_dim`: Number of measured outputs (qubits to measure).
            `n_layers`: Number of layers in the StronglyEntanglingLayers ansatz.
        """
        super().__init__(input_dim, output_dim, n_layers)

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

        weight_shapes: dict[str, tuple[int, int, int]] = {
            "weights": (self.n_layers, self.input_dim, 3)
        }

        self.qnn = qml.qnn.torch.TorchLayer(self.qnode, weight_shapes)

    def _circuit(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> list[MeasurementProcess]:
        """
        The internal quantum circuit logic.

        Args:
            `inputs`: Classical data to embed (Batch, n_qubits).
            `weights`: Variational parameters (n_layers, n_qubits, 3).

        Returns:
            Measurement processes, which the QNode converts to Tensors upon execution.
        """
        qml.AngleEmbedding(inputs, wires=range(self.input_dim), rotation="X")
        qml.StronglyEntanglingLayers(weights, wires=range(self.input_dim))

        return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the quantum circuit.

        Args:
            `x`: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.qnn(x)
