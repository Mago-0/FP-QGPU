import numpy as np
import pytest
import sys
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "fp_qgpu"))
from fp_qgpu.gatter_operationen_numba import cx_gate_numba, u_gate_numba


def _assert_equivalent_up_to_global_phase(
    reference: np.ndarray, candidate: np.ndarray, atol: float = 1e-12
) -> None:
    idx = int(np.argmax(np.abs(reference)))
    phase = reference[idx] / candidate[idx]
    assert np.allclose(reference, candidate * phase, atol=atol)


def _run_aer_statevector(circuit: QuantumCircuit) -> np.ndarray:
    simulator = AerSimulator(method="statevector")
    qc_st = circuit.copy()
    qc_st.save_statevector()
    qc_st = transpile(qc_st, simulator, optimization_level=0)
    result = simulator.run(qc_st, shots=1).result()
    return np.asarray(result.get_statevector(qc_st), dtype=complex)


def _random_normalized_statevector(num_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = 2**num_qubits
    vec = rng.normal(size=size) + 1j * rng.normal(size=size)
    return vec / np.linalg.norm(vec)


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
def test_u_gate_numba_matches_qiskit_statevector(num_qubits: int) -> None:
    depth = max(4, num_qubits * 2)
    qc_random = random_circuit(
        num_qubits=num_qubits,
        depth=depth,
        max_operands=1,
        measure=False,
        seed=100 + num_qubits,
    )
    qc_u_only = transpile(qc_random, basis_gates=["u"], optimization_level=0)

    psi_flat = np.zeros(2**num_qubits, dtype=complex)
    psi_flat[0] = 1.0 + 0.0j
    psi = psi_flat.reshape([2] * num_qubits)

    for instruction in qc_u_only.data:
        name = instruction.operation.name
        if name != "u":
            raise ValueError(f"Expected only 'u' gates, found '{name}'.")

        qubit = qc_u_only.find_bit(instruction.qubits[0]).index
        axis = num_qubits - 1 - qubit
        psi = u_gate_numba(num_qubits, axis, instruction.operation.to_matrix(), psi)

    state_numba = psi.reshape(2**num_qubits)
    state_qiskit = _run_aer_statevector(qc_u_only)

    _assert_equivalent_up_to_global_phase(state_qiskit, state_numba)


@pytest.mark.parametrize(
    "num_qubits,control_qubit,target_qubit",
    [
        (2, 0, 1),
        (2, 1, 0),
        (3, 0, 2),
        (3, 2, 1),
        (4, 0, 3),
        (4, 3, 0),
        (4, 1, 2),
    ],
)
def test_cx_gate_numba_matches_qiskit_statevector(
    num_qubits: int, control_qubit: int, target_qubit: int
) -> None:
    initial_state = _random_normalized_statevector(
        num_qubits, seed=2000 + 17 * num_qubits + 5 * control_qubit + target_qubit
    )

    psi = initial_state.reshape([2] * num_qubits)
    control_axis = num_qubits - 1 - control_qubit
    target_axis = num_qubits - 1 - target_qubit
    state_numba = cx_gate_numba(num_qubits, control_axis, target_axis, psi).reshape(
        2**num_qubits
    )

    qc = QuantumCircuit(num_qubits)
    qc.initialize(initial_state, list(range(num_qubits)))
    qc.cx(control_qubit, target_qubit)
    state_qiskit = _run_aer_statevector(qc)

    _assert_equivalent_up_to_global_phase(state_qiskit, state_numba)
