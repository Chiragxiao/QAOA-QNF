import pennylane as qml
import torch
from torch import nn

n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits)

# Define a quantum circuit
@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.RY(weights[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Quantum layer in PyTorch
# Quantum layer in PyTorch
class QuantumLayer(nn.Module):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(n_qubits))

    def forward(self, x):
        quantum_out = quantum_circuit(x, self.weights)
        return torch.tensor(quantum_out, dtype=torch.float32)

