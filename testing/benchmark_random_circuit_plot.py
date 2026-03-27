from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator

from fp_qgpu.gatter_operationen_numba import simulate_circuit_numba_compiled
from fp_qgpu.simulator import simulator_own


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


def _compile_numba_workload(transpiled_qc: QuantumCircuit) -> tuple[np.ndarray, ...]:
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


def _save_benchmark_data(
    output_dir: Path,
    qubit_counts: list[int],
    own: BenchmarkSeries,
    numba_compiled: BenchmarkSeries,
    aer: BenchmarkSeries,
) -> tuple[Path, Path]:
    csv_path = output_dir / "random_circuit_benchmark_times.csv"
    json_path = output_dir / "random_circuit_benchmark_times.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "num_qubits",
            "simulator_own_mean_s",
            "simulator_own_std_s",
            "numba_compiled_mean_s",
            "numba_compiled_std_s",
            "qiskit_aer_mean_s",
            "qiskit_aer_std_s",
            "ratio_own_over_aer",
            "ratio_numba_compiled_over_aer",
        ])
        for i, num_qubits in enumerate(qubit_counts):
            ratio_own = own.means[i] / aer.means[i]
            ratio_numba = numba_compiled.means[i] / aer.means[i]
            writer.writerow([
                num_qubits,
                own.means[i],
                own.stds[i],
                numba_compiled.means[i],
                numba_compiled.stds[i],
                aer.means[i],
                aer.stds[i],
                ratio_own,
                ratio_numba,
            ])

    rows = []
    for i, num_qubits in enumerate(qubit_counts):
        rows.append({
            "num_qubits": num_qubits,
            "simulator_own_mean_s": own.means[i],
            "simulator_own_std_s": own.stds[i],
            "numba_compiled_mean_s": numba_compiled.means[i],
            "numba_compiled_std_s": numba_compiled.stds[i],
            "qiskit_aer_mean_s": aer.means[i],
            "qiskit_aer_std_s": aer.stds[i],
            "ratio_own_over_aer": own.means[i] / aer.means[i],
            "ratio_numba_compiled_over_aer": numba_compiled.means[i] / aer.means[i],
        })

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return csv_path, json_path


def _write_generated_docs(
    docs_generated_dir: Path,
    qubit_counts: list[int],
    own: BenchmarkSeries,
    numba_compiled: BenchmarkSeries,
    aer: BenchmarkSeries,
) -> Path:
    docs_generated_dir.mkdir(parents=True, exist_ok=True)
    generated_rst_path = docs_generated_dir / "benchmark_random_circuit_results.rst"

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append("Auto-generated benchmark report")
    lines.append("-------------------------------")
    lines.append("")
    lines.append(f"Generated at: {generated_at}")
    lines.append("")
    lines.append(".. csv-table:: Runtime comparison across all methods")
    lines.append(
        '   :header: "Qubits", "simulator_own [s]", "numba_compiled [s]", "qiskit_aer [s]", "own/aer", "numba_compiled/aer"'
    )
    lines.append("")

    for i, num_qubits in enumerate(qubit_counts):
        ratio_own = own.means[i] / aer.means[i]
        ratio_numba = numba_compiled.means[i] / aer.means[i]
        lines.append(
            f'   "{num_qubits}", "{own.means[i]:.6f}", "{numba_compiled.means[i]:.6f}", "{aer.means[i]:.6f}", "{ratio_own:.4f}", "{ratio_numba:.4f}"'
        )

    lines.append("")
    lines.append("The generated benchmark plot:")
    lines.append("")
    lines.append(".. image:: /_static/random_circuit_benchmark.png")
    lines.append("   :alt: Benchmark runtime and ratio over qubits")
    lines.append("   :width: 90%")
    lines.append("")

    generated_rst_path.write_text("\n".join(lines), encoding="utf-8")
    return generated_rst_path


def main() -> None:
    output_dir = Path("testing") / ".benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_static_dir = Path("docs") / "_static"
    docs_generated_dir = Path("docs") / "_generated"
    docs_static_dir.mkdir(parents=True, exist_ok=True)

    qubit_counts = list(range(1, 21, 2))
    repeats = 7

    own = BenchmarkSeries("simulator_own", [], [])
    numba_compiled = BenchmarkSeries("numba_compiled", [], [])
    aer = BenchmarkSeries("qiskit_aer", [], [])

    aer_simulator = AerSimulator(
        method="statevector", fusion_enable=False, max_parallel_threads=1
    )

    for n in qubit_counts:
        qc_trans, qc_aer = _build_circuits(num_qubits=n, seed=1234 + n)
        compiled_workload = _compile_numba_workload(qc_trans)
        repeats_for_n = 3 if n >= 16 else repeats

        # Warmup pass to avoid including one-time setup/JIT costs in timings.
        simulator_own(qc_trans)
        _run_numba_compiled_statevector(n, compiled_workload)
        _run_aer_statevector(aer_simulator, qc_aer)

        mean_own, std_own = _bench_callable(
            lambda: simulator_own(qc_trans), repeats_for_n
        )
        mean_numba_compiled, std_numba_compiled = _bench_callable(
            lambda: _run_numba_compiled_statevector(n, compiled_workload), repeats_for_n
        )
        mean_aer, std_aer = _bench_callable(
            lambda: _run_aer_statevector(aer_simulator, qc_aer), repeats_for_n
        )

        own.means.append(mean_own)
        own.stds.append(std_own)
        numba_compiled.means.append(mean_numba_compiled)
        numba_compiled.stds.append(std_numba_compiled)
        aer.means.append(mean_aer)
        aer.stds.append(std_aer)

        print(
            f"{n:2d}q | own={mean_own:.6f}s | numba_compiled={mean_numba_compiled:.6f}s | "
            f"aer={mean_aer:.6f}s | rounds={repeats_for_n}"
        )

    ratio_own = np.array(own.means) / np.array(aer.means)
    ratio_numba_compiled = np.array(numba_compiled.means) / np.array(aer.means)

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
        numba_compiled.means,
        yerr=numba_compiled.stds,
        marker="s",
        capsize=3,
        linewidth=2,
        label=numba_compiled.name,
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
        ratio_numba_compiled,
        marker="s",
        linewidth=2,
        label="numba_compiled / qiskit_aer",
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
    docs_plot_file = docs_static_dir / "random_circuit_benchmark.png"
    shutil.copy2(output_file, docs_plot_file)

    csv_path, json_path = _save_benchmark_data(
        output_dir, qubit_counts, own, numba_compiled, aer
    )
    generated_rst_path = _write_generated_docs(
        docs_generated_dir, qubit_counts, own, numba_compiled, aer
    )

    print(f"Saved plot to: {output_file}")
    print(f"Saved docs plot to: {docs_plot_file}")
    print(f"Saved benchmark CSV to: {csv_path}")
    print(f"Saved benchmark JSON to: {json_path}")
    print(f"Updated generated docs at: {generated_rst_path}")


if __name__ == "__main__":
    main()
