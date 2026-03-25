from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from fp_qgpu.simulator_mock import simulator_mock


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


def ghz_test(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n):
        if i < n - 1:
            qc.cx(i, i + 1)
    print(qc)
    transpiled_qc = transpile(qc, basis_gates=["u", "cx"])
    return transpiled_qc


def ghz(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n):
        if i < n - 1:
            qc.cx(i, i + 1)
    qc.measure_all()
    print(qc)
    transpiled_qc = transpile(qc, basis_gates=["u", "cx"])
    print(transpiled_qc.num_qubits)

    # Transpile for simulator
    simulator = AerSimulator()
    circ = transpile(transpiled_qc, simulator)

    # Run and get counts
    result = simulator.run(circ).result()
    counts = result.get_counts(circ)
    fig, ax = plt.subplots()
    plot_histogram(counts, ax=ax)
    plt.show()

    return transpiled_qc


def qft():
    n = 4
    qc = QuantumCircuit(n)
    qc.z(0)
    qc.append(
        QFTGate(n), range(n)
    )  # erstes arg: über wie viele qubits wird das QFT angewendet, zweites arg: über welche qbits wird gate angewendet
    qc.measure_all()
    print(qc)
    transpiled_qc = transpile(qc, basis_gates=["u", "cx"])
    print(transpiled_qc)

    # Transpile for simulator
    simulator = AerSimulator()
    circ = transpile(transpiled_qc, simulator)

    # Run and get counts
    result = simulator.run(circ).result()
    counts = result.get_counts(circ)
    fig, ax = plt.subplots()
    plot_histogram(counts, ax=ax)
    plt.show()
    return


def qft_superpos(n):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    qc.append(
        QFTGate(n), range(n)
    )  # erstes arg: über wie viele qubits wird das QFT angewendet, zweites arg: über welche qbits wird gate angewendet
    qc.measure_all()
    print(qc)
    transpiled_qc = transpile(qc, basis_gates=["u", "cx"])
    print(transpiled_qc)

    # Transpile for simulator
    simulator = AerSimulator()
    circ = transpile(transpiled_qc, simulator)

    # Run and get counts
    result = simulator.run(circ).result()
    counts = result.get_counts(circ)
    fig, ax = plt.subplots()
    plot_histogram(counts, ax=ax)
    plt.show()
    return


def ghz_example(n=3):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n):
        if i < n - 1:
            qc.cx(i, i + 1)
    qc.measure_all()
    print(qc)
    transpiled_qc = transpile(qc, basis_gates=["u", "cx"])
    sim_result = simulator_mock(transpiled_qc)
    print(sim_result)
    fig, ax = plt.subplots()
    plot_histogram(sim_result, ax=ax)
    plt.show()
