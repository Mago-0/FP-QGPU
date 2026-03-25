from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
from fp_qgpu.simulator_mock import simulator_mock
from qiskit.circuit.random import random_circuit
from fp_qgpu.simulator import simulator_own
import numpy as np


def create_test_circuit(n: int = 4) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n):
        if i < n - 1:
            qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def test_simulator() -> None:
    # Test with a simple circuit
    qc = create_test_circuit()

    # Mock simulator
    test_values = {"qc": qc, "shots": 1024, "seed": 20}
    result_mock = simulator_mock(**test_values)

    # Aer simulator
    simulator = AerSimulator(seed_simulator=test_values["seed"])
    circ = transpile(qc, simulator)

    result = simulator.run(circ, shots=test_values["shots"]).result()
    result_Aer = result.get_counts(circ)

    assert result_mock[0] == result_Aer


def test_random_circuit() -> None:
    # Test with a random circuit
    n = 4
    depth = 10
    qc = random_circuit(n, depth, measure=True)

    # Mock simulator
    test_values = {"qc": qc, "shots": 1024, "seed": 20}
    result_mock = simulator_mock(**test_values)

    # Aer simulator
    simulator = AerSimulator(seed_simulator=test_values["seed"])
    circ = transpile(qc, simulator)

    result = simulator.run(circ, shots=test_values["shots"]).result()
    result_Aer = result.get_counts(circ)

    assert result_mock[0] == result_Aer


def test_random_circuit_statevector() -> None:
    # Test with a random circuit and compare statevectors
    n = 4
    depth = 10
    qc = random_circuit(n, depth, measure=False)

    # Mock simulator
    test_values = {"qc": qc, "shots": 1024, "seed": 20}
    result_mock = simulator_mock(**test_values)

    # Aer simulator
    qc_st = qc.copy()
    qc_st.save_statevector()

    simulator = AerSimulator(seed_simulator=test_values["seed"])
    circ = transpile(qc_st, simulator)

    result = simulator.run(circ, shots=test_values["shots"]).result()
    state_vector_Aer = result.get_statevector(circ)

    assert result_mock[1] == state_vector_Aer


def test_own_simulator() -> None:

    # --- Random Circuit ---
    n = 4
    depth = 10
    qc = random_circuit(n, depth, measure=False)

    qc_trans = transpile(qc, basis_gates=["u", "cx"])

    # --- Dein Simulator ---
    result_own = simulator_own(qc_trans)

    # --- Aer Simulator ---
    simulator = AerSimulator()
    qc_st = qc.copy()
    qc_st.save_statevector()
    circ = transpile(qc_st, simulator)
    result = simulator.run(circ, shots=1024).result()
    state_vector_Aer = result.get_statevector(circ)

    # --- globale Phase bestimmen ---
    idx = int(np.argmax(np.abs(state_vector_Aer)))

    phase = state_vector_Aer[idx] / result_own[idx]
    assert np.allclose(state_vector_Aer, result_own * phase, atol=1e-12)
