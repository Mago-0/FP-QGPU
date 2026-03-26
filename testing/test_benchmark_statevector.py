import time
import numpy as np
import pytest
from fp_qgpu.gatter_operationen import cx, u_gate
from fp_qgpu.gatter_operationen_numba import (
    cx_numba_compatible,
    cx_numba_compatible_three_loops,
    u_gate_numba_compatible,
    u_gate_numba_compatible_three_loops,
    u_gate_numba_compatible_two_loops,
)
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


def _simulate_with_impls(transpiled_qc, u_impl, cx_impl) -> np.ndarray:
    num_qubits = transpiled_qc.num_qubits
    psi_flat = np.zeros(2**num_qubits, dtype=complex)
    psi_flat[0] = 1.0 + 0.0j
    psi = psi_flat.reshape([2] * num_qubits)

    for instruction in transpiled_qc.data:
        name = instruction.operation.name

        if name == "u":
            qubit = transpiled_qc.find_bit(instruction.qubits[0]).index
            axis = num_qubits - 1 - qubit
            psi = u_impl(num_qubits, axis, instruction.operation.to_matrix(), psi)
            continue

        if name == "cx":
            control_qubit = transpiled_qc.find_bit(instruction.qubits[0]).index
            target_qubit = transpiled_qc.find_bit(instruction.qubits[1]).index
            control_axis = num_qubits - 1 - control_qubit
            target_axis = num_qubits - 1 - target_qubit
            psi = cx_impl(num_qubits, control_axis, target_axis, psi)
            continue

        raise ValueError(f"Unexpected gate '{name}' in benchmark circuit.")

    return psi.reshape(2**num_qubits)


def _run_variant_statevector(variant_name: str, qc_trans):
    if variant_name == "simulator_own":
        return simulator_own(qc_trans)

    if variant_name == "u_base__cx_base":
        return _simulate_with_impls(qc_trans, u_gate_numba_compatible, cx_numba_compatible)

    if variant_name == "u_base__cx_three":
        return _simulate_with_impls(
            qc_trans, u_gate_numba_compatible, cx_numba_compatible_three_loops
        )

    if variant_name == "u_two__cx_base":
        return _simulate_with_impls(
            qc_trans, u_gate_numba_compatible_two_loops, cx_numba_compatible
        )

    if variant_name == "u_two__cx_three":
        return _simulate_with_impls(
            qc_trans, u_gate_numba_compatible_two_loops, cx_numba_compatible_three_loops
        )

    if variant_name == "u_three__cx_base":
        return _simulate_with_impls(
            qc_trans, u_gate_numba_compatible_three_loops, cx_numba_compatible
        )

    if variant_name == "u_three__cx_three":
        return _simulate_with_impls(
            qc_trans,
            u_gate_numba_compatible_three_loops,
            cx_numba_compatible_three_loops,
        )

    raise ValueError(f"Unknown variant '{variant_name}'.")


@pytest.mark.parametrize("num_qubits", [2, 4, 6, 8])
@pytest.mark.parametrize(
    "variant_name",
    [
        "simulator_own",
        "u_base__cx_base",
        "u_base__cx_three",
        "u_two__cx_base",
        "u_two__cx_three",
        "u_three__cx_base",
        "u_three__cx_three",
    ],
)
def test_statevector_runtime_ratio_vs_aer(benchmark, num_qubits: int, variant_name: str):
    depth = max(8, num_qubits * 3)
    qc = random_circuit(num_qubits, depth, measure=False, seed=200 + num_qubits)
    qc_trans = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)

    simulator = AerSimulator(method="statevector")
    qc_aer = qc_trans.copy()
    qc_aer.save_statevector()
    qc_aer = transpile(qc_aer, simulator, optimization_level=0)

    state_own = _run_variant_statevector(variant_name, qc_trans)
    state_aer = _run_aer_statevector(simulator, qc_aer)
    _assert_equivalent_up_to_global_phase(state_aer, state_own)

    variant_times: list[float] = []
    aer_times: list[float] = []
    ratios: list[float] = []

    def run_both() -> None:
        t0 = time.perf_counter()
        _run_variant_statevector(variant_name, qc_trans)
        variant_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        _run_aer_statevector(simulator, qc_aer)
        aer_time = time.perf_counter() - t0

        variant_times.append(variant_time)
        aer_times.append(aer_time)
        ratios.append(variant_time / aer_time)

    benchmark.pedantic(run_both, rounds=12, iterations=1, warmup_rounds=2)

    mean_variant = float(np.mean(variant_times))
    mean_aer = float(np.mean(aer_times))
    mean_ratio = float(np.mean(ratios))

    benchmark.extra_info["variant"] = variant_name
    benchmark.extra_info["mean_variant_s"] = mean_variant
    benchmark.extra_info["mean_aer_s"] = mean_aer
    benchmark.extra_info["mean_ratio_variant_div_aer"] = mean_ratio
    print(f"[ratio][{variant_name}][{num_qubits}q] variant/aer={mean_ratio:.4f}")
    assert mean_ratio > 0
