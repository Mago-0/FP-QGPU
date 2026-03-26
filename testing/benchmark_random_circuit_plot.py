from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator

from fp_qgpu.simulator import simulator_own, simulator_own_numba


@dataclass
class BenchmarkSeries:
    name: str
    means: list[float]
    stds: list[float]


def _run_aer_statevector(
    simulator: AerSimulator, circuit: QuantumCircuit
) -> np.ndarray:
    result = simulator.run(circuit, shots=1).result()
    return np.asarray(result.get_statevector(circuit), dtype=complex)


def _bench_callable(func, repeats: int) -> tuple[float, float]:
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def _build_circuits(
    num_qubits: int, seed: int
) -> tuple[QuantumCircuit, QuantumCircuit]:
    depth = max(8, num_qubits * 3)
    qc = random_circuit(num_qubits, depth, measure=False, seed=seed)
    qc_trans = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)

    aer_simulator = AerSimulator(
        method="statevector", fusion_enable=False, max_parallel_threads=1
    )
    qc_aer = qc_trans.copy()
    qc_aer.save_statevector()
    qc_aer = transpile(qc_aer, aer_simulator, optimization_level=0)

    return qc_trans, qc_aer


def main() -> None:
    output_dir = Path("testing") / ".benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    qubit_counts = list(range(1, 21, 2))
    repeats = 7

    own = BenchmarkSeries("simulator_own", [], [])
    own_numba = BenchmarkSeries("simulator_own_numba", [], [])
    aer = BenchmarkSeries("qiskit_aer", [], [])

    # Use a fixed Aer configuration requested by the user.
    aer_simulator = AerSimulator(
        method="statevector", fusion_enable=False, max_parallel_threads=1
    )

    for n in qubit_counts:
        qc_trans, qc_aer = _build_circuits(num_qubits=n, seed=1234 + n)
        repeats_for_n = 3 if n >= 16 else repeats

        # Warmup pass to avoid including one-time setup/JIT costs in timings.
        simulator_own(qc_trans)
        simulator_own_numba(qc_trans)
        _run_aer_statevector(aer_simulator, qc_aer)

        mean_own, std_own = _bench_callable(
            lambda: simulator_own(qc_trans), repeats_for_n
        )
        mean_own_numba, std_own_numba = _bench_callable(
            lambda: simulator_own_numba(qc_trans), repeats_for_n
        )
        mean_aer, std_aer = _bench_callable(
            lambda: _run_aer_statevector(aer_simulator, qc_aer), repeats_for_n
        )

        own.means.append(mean_own)
        own.stds.append(std_own)
        own_numba.means.append(mean_own_numba)
        own_numba.stds.append(std_own_numba)
        aer.means.append(mean_aer)
        aer.stds.append(std_aer)

        print(
            f"{n:2d}q | own={mean_own:.6f}s | own_numba={mean_own_numba:.6f}s | "
            f"aer={mean_aer:.6f}s | rounds={repeats_for_n}"
        )

    ratio_own = np.array(own.means) / np.array(aer.means)
    ratio_own_numba = np.array(own_numba.means) / np.array(aer.means)

    fig, (ax_runtime, ax_ratio) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    ax_runtime.errorbar(
        qubit_counts,
        own.means,
        yerr=own.stds,
        marker="o",
        capsize=3,
        linewidth=2,
        label=own.name,
    )
    ax_runtime.errorbar(
        qubit_counts,
        own_numba.means,
        yerr=own_numba.stds,
        marker="s",
        capsize=3,
        linewidth=2,
        label=own_numba.name,
    )
    ax_runtime.errorbar(
        qubit_counts,
        aer.means,
        yerr=aer.stds,
        marker="^",
        capsize=3,
        linewidth=2,
        label=aer.name,
    )

    ax_runtime.set_title("Random Circuit Benchmark: Aer vs simulator_own variants")
    ax_runtime.set_ylabel("Runtime [s]")
    ax_runtime.set_yscale("log")
    ax_runtime.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_runtime.legend()

    ax_ratio.plot(
        qubit_counts,
        ratio_own,
        marker="o",
        linewidth=2,
        label="simulator_own / qiskit_aer",
    )
    ax_ratio.plot(
        qubit_counts,
        ratio_own_numba,
        marker="s",
        linewidth=2,
        label="simulator_own_numba / qiskit_aer",
    )
    ax_ratio.axhline(1.0, color="black", linestyle=":", linewidth=1.5, label="parity")
    ax_ratio.set_xlabel("Number of Qubits")
    ax_ratio.set_ylabel("Runtime Ratio")
    ax_ratio.set_yscale("log")
    ax_ratio.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_ratio.legend()

    fig.tight_layout()

    output_file = output_dir / "random_circuit_benchmark.png"
    fig.savefig(output_file, dpi=180)
    print(f"Saved plot to: {output_file}")


if __name__ == "__main__":
    main()
