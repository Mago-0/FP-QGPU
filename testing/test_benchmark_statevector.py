import time
import numpy as np
import pytest
from fp_qgpu.gatter_operationen_numba import (
    simulate_circuit_numba_compiled,
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


def _compile_numba_workload(transpiled_qc) -> tuple[np.ndarray, ...]:
    num_qubits = transpiled_qc.num_qubits
    gate_kinds: list[int] = []
    u_axes: list[int] = []
    u_mats: list[np.ndarray] = []
    cx_controls: list[int] = []
    cx_targets: list[int] = []

    for instruction in transpiled_qc.data:
        name = instruction.operation.name

        if name == "u":
            qubit = transpiled_qc.find_bit(instruction.qubits[0]).index
            axis = num_qubits - 1 - qubit
            gate_kinds.append(0)
            u_axes.append(axis)
            u_mats.append(
                np.asarray(instruction.operation.to_matrix(), dtype=np.complex128)
            )
            continue

        if name == "cx":
            control_qubit = transpiled_qc.find_bit(instruction.qubits[0]).index
            target_qubit = transpiled_qc.find_bit(instruction.qubits[1]).index
            control_axis = num_qubits - 1 - control_qubit
            target_axis = num_qubits - 1 - target_qubit
            gate_kinds.append(1)
            cx_controls.append(control_axis)
            cx_targets.append(target_axis)
            continue

        raise ValueError(f"Unexpected gate '{name}' in benchmark circuit.")

    if len(u_mats) > 0:
        u_mats_arr = np.asarray(u_mats, dtype=np.complex128)
    else:
        u_mats_arr = np.zeros((0, 2, 2), dtype=np.complex128)

    return (
        np.asarray(gate_kinds, dtype=np.int8),
        np.asarray(u_axes, dtype=np.int64),
        u_mats_arr,
        np.asarray(cx_controls, dtype=np.int64),
        np.asarray(cx_targets, dtype=np.int64),
    )


def _run_numba_compiled_statevector(
    num_qubits: int, compiled_workload: tuple[np.ndarray, ...]
) -> np.ndarray:
    gate_kinds, u_axes, u_mats, cx_controls, cx_targets = compiled_workload
    return simulate_circuit_numba_compiled(
        num_qubits,
        gate_kinds,
        u_axes,
        u_mats,
        cx_controls,
        cx_targets,
    )


def _run_variant_statevector(variant_name: str, qc_trans):
    if variant_name == "simulator_own":
        return simulator_own(qc_trans)

    if variant_name == "numba_compiled":
        compiled_workload = _compile_numba_workload(qc_trans)
        return _run_numba_compiled_statevector(qc_trans.num_qubits, compiled_workload)

    raise ValueError(f"Unknown variant '{variant_name}'.")


def _collect_call_times(
    callable_fn, rounds: int, warmup_rounds: int = 0
) -> list[float]:
    for _ in range(warmup_rounds):
        callable_fn()

    times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        callable_fn()
        times.append(time.perf_counter() - t0)
    return times


@pytest.mark.parametrize("num_qubits", [2, 4, 6, 8])
@pytest.mark.parametrize(
    "variant_name",
    [
        "simulator_own",
        "numba_compiled",
    ],
)
def test_statevector_runtime_ratio_vs_aer(
    benchmark, num_qubits: int, variant_name: str
):
    depth = max(8, num_qubits * 3)
    qc = random_circuit(num_qubits, depth, measure=False, seed=200 + num_qubits)
    qc_trans = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)

    simulator = AerSimulator(method="statevector")
    qc_aer = qc_trans.copy()
    qc_aer.save_statevector()
    qc_aer = transpile(qc_aer, simulator, optimization_level=0)

    compiled_workload = _compile_numba_workload(qc_trans)

    # Trigger JIT compilation before timing to avoid first-run bias.
    _run_numba_compiled_statevector(num_qubits, compiled_workload)
    _run_aer_statevector(simulator, qc_aer)

    if variant_name == "numba_compiled":
        state_own = _run_numba_compiled_statevector(num_qubits, compiled_workload)
    else:
        state_own = _run_variant_statevector(variant_name, qc_trans)

    state_aer = _run_aer_statevector(simulator, qc_aer)
    _assert_equivalent_up_to_global_phase(state_aer, state_own)

    if variant_name == "numba_compiled":

        def run_variant() -> np.ndarray:
            return _run_numba_compiled_statevector(num_qubits, compiled_workload)

    else:

        def run_variant() -> np.ndarray:
            return _run_variant_statevector(variant_name, qc_trans)

    rounds = 12
    warmup_rounds = 2
    variant_times = _collect_call_times(
        run_variant, rounds=rounds, warmup_rounds=warmup_rounds
    )
    aer_times = _collect_call_times(
        lambda: _run_aer_statevector(simulator, qc_aer),
        rounds=rounds,
        warmup_rounds=warmup_rounds,
    )
    ratios = [v / a for v, a in zip(variant_times, aer_times)]

    benchmark.pedantic(
        run_variant, rounds=rounds, iterations=1, warmup_rounds=warmup_rounds
    )

    mean_variant = float(np.mean(variant_times))
    mean_aer = float(np.mean(aer_times))
    mean_ratio = float(np.mean(ratios))

    benchmark.extra_info["variant"] = variant_name
    benchmark.extra_info["mean_variant_s"] = mean_variant
    benchmark.extra_info["mean_aer_s"] = mean_aer
    benchmark.extra_info["mean_ratio_variant_div_aer"] = mean_ratio
    print(f"[ratio][{variant_name}][{num_qubits}q] variant/aer={mean_ratio:.4f}")
    assert mean_ratio > 0
