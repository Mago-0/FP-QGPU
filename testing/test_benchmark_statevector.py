import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from fp_qgpu.simulator import simulator_own
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator

BENCHMARK_OUTPUT_DIR = Path(__file__).parent / ".benchmarks"
RUNTIME_PLOT_PATH = BENCHMARK_OUTPUT_DIR / "runtime_ratio_vs_qubits.png"
_RUNTIME_POINTS: dict[int, tuple[float, float, float]] = {}


def _write_runtime_ratio_plot() -> None:
    if not _RUNTIME_POINTS:
        return

    BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    qubits = sorted(_RUNTIME_POINTS)
    own_means = [_RUNTIME_POINTS[q][0] for q in qubits]
    aer_means = [_RUNTIME_POINTS[q][1] for q in qubits]
    ratios = [_RUNTIME_POINTS[q][2] for q in qubits]

    fig, (ax_runtime, ax_ratio) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 7), sharex=True
    )
    ax_runtime.plot(qubits, own_means, marker="o", label="Own implementation")
    ax_runtime.plot(qubits, aer_means, marker="s", label="Qiskit Aer")
    ax_runtime.set_ylabel("Runtime [s]")
    ax_runtime.set_title("Statevector runtime comparison over qubit count")
    ax_runtime.grid(True, alpha=0.3)
    ax_runtime.legend()

    ax_ratio.plot(qubits, ratios, marker="^", color="tab:green", label="Own / Aer")
    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax_ratio.set_xlabel("Number of qubits")
    ax_ratio.set_ylabel("Runtime ratio")
    ax_ratio.set_title("Runtime ratio over qubit count")
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend()

    fig.tight_layout()
    fig.savefig(RUNTIME_PLOT_PATH, dpi=150)
    plt.close(fig)


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
    _RUNTIME_POINTS[num_qubits] = (mean_own, mean_aer, mean_ratio)
    _write_runtime_ratio_plot()
    print(f"[ratio][{num_qubits}q] own/aer={mean_ratio:.4f}")
    print(f"[plot] saved {RUNTIME_PLOT_PATH}")
    assert mean_ratio > 0
