from qiskit import transpile
from qiskit.circuit.random import random_circuit
import numpy as np

from fp_qgpu.gatter_operationen import get_circuit, u_gate
from fp_qgpu.gatter_operationen_numba import u_gate_numba


def _simulate_u_only_with_original(transpiled_qc) -> np.ndarray:
    """Apply only U gates using the existing (einsum-based) implementation."""
    num_qubits = transpiled_qc.num_qubits
    psi_flat = np.zeros(2**num_qubits, dtype=complex)
    psi_flat[0] = 1.0 + 0.0j
    psi = psi_flat.reshape([2] * num_qubits)

    for gate in get_circuit(transpiled_qc):
        name = gate[0]
        acting_on = gate[1]
        matrix = gate[2]

        if name != "u":
            raise ValueError(f"Expected only 'u' gates, found gate '{name}'.")

        axis_index = num_qubits - 1 - acting_on[0]
        psi = u_gate(num_qubits, axis_index, matrix, psi)

    return psi.reshape(2**num_qubits)


def _simulate_u_only_with_numba_compatible(transpiled_qc) -> np.ndarray:
    """Apply only U gates using the explicit numba-compatible implementation."""
    num_qubits = transpiled_qc.num_qubits
    psi_flat = np.zeros(2**num_qubits, dtype=complex)
    psi_flat[0] = 1.0 + 0.0j
    psi = psi_flat.reshape([2] * num_qubits)

    for gate in get_circuit(transpiled_qc):
        name = gate[0]
        acting_on = gate[1]
        matrix = gate[2]

        if name != "u":
            raise ValueError(f"Expected only 'u' gates, found gate '{name}'.")

        axis_index = num_qubits - 1 - acting_on[0]
        psi = u_gate_numba(num_qubits, axis_index, matrix, psi)

    return psi.reshape(2**num_qubits)


def run_u_only_comparison(
    num_qubits: int = 4, depth: int = 8, seed: int = 123
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a transpiled random U-only circuit and return both statevectors:
    (original implementation result, numba-compatible result).
    """
    # max_operands=1 restricts random gates to single-qubit gates.
    qc_random = random_circuit(
        num_qubits=num_qubits,
        depth=depth,
        max_operands=1,
        measure=False,
        seed=seed,
    )

    # Transpile to the U basis so that only "u" operations remain.
    qc_u_only = transpile(qc_random, basis_gates=["u"], optimization_level=0)

    gate_names = [instruction.operation.name for instruction in qc_u_only.data]
    invalid_gates = [name for name in gate_names if name != "u"]
    if invalid_gates:
        raise ValueError(
            f"Transpiled circuit is not U-only. Found non-U gates: {sorted(set(invalid_gates))}"
        )

    original_result = _simulate_u_only_with_original(qc_u_only)
    numba_compatible_result = _simulate_u_only_with_numba_compatible(qc_u_only)
    return original_result, numba_compatible_result


if __name__ == "__main__":
    original, numba_compatible = run_u_only_comparison()
    print("Original implementation statevector:")
    print(original)
    print("\nNumba-compatible implementation statevector:")
    print(numba_compatible)
    print("\nAllclose:", np.allclose(original, numba_compatible, atol=1e-12))
