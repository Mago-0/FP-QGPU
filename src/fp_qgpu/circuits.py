from qiskit import QuantumCircuit


def simple00() -> QuantumCircuit:
    # einfacher Test-Circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def simple01() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(0)
    qc.z(1)
    qc.y(1)
    qc.y(0)
    return qc
