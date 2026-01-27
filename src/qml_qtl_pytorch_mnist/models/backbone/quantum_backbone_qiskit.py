from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import Tensor

from .base_backbone import BaseBackbone


class QuantumBackboneQiskit(BaseBackbone):
    """
    Qiskit implementation of the Quantum Backbone using Primitives V2.
    """

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 2,
        n_layers: int = 2,
    ) -> None:
        """
        Initializes the Qiskit quantum layer.

        Args:
            `input_dim`: Number of input features (qubits).
            `output_dim`: Number of measured outputs (first k qubits).
            `n_layers`: Depth of the EfficientSU2 ansatz.
        """
        super().__init__(input_dim, output_dim, n_layers)

        qc = QuantumCircuit(self.input_dim)

        input_params = ParameterVector("x", self.input_dim)
        for i in range(self.input_dim):
            qc.rx(input_params[i], i)

        ansatz = efficient_su2(
            num_qubits=self.input_dim,
            su2_gates=["ry", "rz"],
            entanglement="circular",
            reps=self.n_layers,
            parameter_prefix="theta",
        )
        qc.compose(ansatz, inplace=True)

        observables = [
            SparsePauliOp.from_list(
                [("I" * (self.input_dim - 1 - i) + "Z" + "I" * i, 1.0)]
            )
            for i in range(self.output_dim)
        ]

        estimator = StatevectorEstimator()

        self.qnn_net = EstimatorQNN(
            circuit=qc,
            estimator=estimator,
            observables=observables,
            input_params=list(input_params),
            weight_params=list(ansatz.parameters),
            input_gradients=True,
        )

        self.qnn = TorchConnector(self.qnn_net)

    def forward(self, x: Tensor) -> Tensor:
        """
        Passes the input through the quantum circuit.

        Args:
            `x`: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.qnn(x)
