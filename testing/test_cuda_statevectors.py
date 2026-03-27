import numpy as np
import pytest
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator

from fp_qgpu.simulator import CUDA_IMPORT_AVAILABLE, simulator_own_numba


def _assert_equivalent_up_to_global_phase(
    reference: np.ndarray, candidate: np.ndarray, atol: float = 1e-12
) -> None:
    idx = int(np.argmax(np.abs(reference)))
    if np.abs(candidate[idx]) < 1e-16:
        idx = int(np.argmax(np.abs(candidate)))
    phase = reference[idx] / candidate[idx]
    assert np.allclose(reference, candidate * phase, atol=atol)


def _cuda_ready() -> bool:
    if not CUDA_IMPORT_AVAILABLE:
        return False
    from numba import cuda

    return cuda.is_available()


@pytest.mark.parametrize("num_qubits", [2, 4, 6])
def test_cuda_statevector_matches_aer(num_qubits: int) -> None:
    if not _cuda_ready():
        pytest.skip("CUDA tests requested, but CUDA is unavailable on this machine.")

    depth = 10
    qc = random_circuit(num_qubits, depth, measure=False, seed=500 + num_qubits)
    qc_trans = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)

    # Warm up CUDA kernels to avoid one-time compile effects in the test run.
    simulator_own_numba(qc_trans, use_cuda=True)
    cuda_state = simulator_own_numba(qc_trans, use_cuda=True)

    simulator = AerSimulator(method="statevector")
    qc_aer = qc_trans.copy()
    qc_aer.save_statevector()
    qc_aer = transpile(qc_aer, simulator, optimization_level=0)
    aer_state = np.asarray(simulator.run(qc_aer, shots=1).result().get_statevector(qc_aer))

    _assert_equivalent_up_to_global_phase(aer_state, cuda_state)
