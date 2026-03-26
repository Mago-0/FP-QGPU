import time

import numpy as np
import pytest
from fp_qgpu.simulator import simulator_own
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator


def _assert_equivalent_up_to_global_phase(
    reference: np.ndarray, candidate: np.ndarray, atol: float = 1e-12
) -> None:
    idx = int(np.argmax(np.abs(reference)))
    phase = reference[idx] / candidate[idx]
    assert np.allclose(reference, candidate * phase, atol=atol)


def _run_aer_statevector(simulator: AerSimulator, circuit) -> np.ndarray:
    result = simulator.run(circuit, shots=1).result()
    return np.asarray(result.get_statevector(circuit), dtype=complex)


@pytest.mark.parametrize("num_qubits", [2, 4, 6, 8])
def test_statevector_runtime_ratio_vs_aer(benchmark, num_qubits: int):
    depth = max(8, num_qubits * 3)
    qc = random_circuit(num_qubits, depth, measure=False, seed=200 + num_qubits)
    qc_trans = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)

    simulator = AerSimulator(method="statevector")
    qc_aer = qc_trans.copy()
    qc_aer.save_statevector()
    qc_aer = transpile(qc_aer, simulator, optimization_level=0)

    state_own = simulator_own(qc_trans)
    state_aer = _run_aer_statevector(simulator, qc_aer)
    _assert_equivalent_up_to_global_phase(state_aer, state_own)

    own_times: list[float] = []
    aer_times: list[float] = []
    ratios: list[float] = []

    def run_both() -> None:
        t0 = time.perf_counter()
        simulator_own(qc_trans)
        own_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        _run_aer_statevector(simulator, qc_aer)
        aer_time = time.perf_counter() - t0

        own_times.append(own_time)
        aer_times.append(aer_time)
        ratios.append(own_time / aer_time)

    benchmark.pedantic(run_both, rounds=20, iterations=1, warmup_rounds=2)

    mean_own = float(np.mean(own_times))
    mean_aer = float(np.mean(aer_times))
    mean_ratio = float(np.mean(ratios))

    benchmark.extra_info["mean_own_s"] = mean_own
    benchmark.extra_info["mean_aer_s"] = mean_aer
    benchmark.extra_info["mean_ratio_own_div_aer"] = mean_ratio
    print(f"[ratio][{num_qubits}q] own/aer={mean_ratio:.4f}")
    assert mean_ratio > 0
