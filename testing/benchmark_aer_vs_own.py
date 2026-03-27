from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator

from fp_qgpu.gatter_operationen_numba import cx_gate_numba, u_gate_numba
from fp_qgpu.simulator import simulator_own


@dataclass
class BenchmarkRow:
    qubits: int
    aer_s: float
    own_s: float
    numba_s: float

    @property
    def own_ratio(self) -> float:
        return self.own_s / self.aer_s

    @property
    def numba_ratio(self) -> float:
        return self.numba_s / self.aer_s


def _assert_equivalent_up_to_global_phase(
    reference: np.ndarray, candidate: np.ndarray, atol: float = 1e-10
) -> None:
    idx = int(np.argmax(np.abs(reference)))
    if np.abs(candidate[idx]) < 1e-16:
        idx = int(np.argmax(np.abs(candidate)))
    phase = reference[idx] / candidate[idx]
    if not np.allclose(reference, candidate * phase, atol=atol):
        raise AssertionError("Statevectors differ beyond global phase.")


def _run_aer_statevector(
    simulator: AerSimulator, circuit: QuantumCircuit
) -> np.ndarray:
    result = simulator.run(circuit, shots=1).result()
    return np.asarray(result.get_statevector(circuit), dtype=complex)


def _simulate_with_numba_gates(transpiled_qc: QuantumCircuit) -> np.ndarray:
    num_qubits = transpiled_qc.num_qubits
    psi_flat = np.zeros(2**num_qubits, dtype=complex)
    psi_flat[0] = 1.0 + 0.0j
    psi = psi_flat.reshape([2] * num_qubits)

    for instruction in transpiled_qc.data:
        gate_name = instruction.operation.name

        if gate_name == "u":
            qubit = transpiled_qc.find_bit(instruction.qubits[0]).index
            axis = num_qubits - 1 - qubit
            psi = u_gate_numba(num_qubits, axis, instruction.operation.to_matrix(), psi)
            continue

        if gate_name == "cx":
            control_qubit = transpiled_qc.find_bit(instruction.qubits[0]).index
            target_qubit = transpiled_qc.find_bit(instruction.qubits[1]).index
            control_axis = num_qubits - 1 - control_qubit
            target_axis = num_qubits - 1 - target_qubit
            psi = cx_gate_numba(num_qubits, control_axis, target_axis, psi)
            continue

        raise ValueError(f"Unexpected gate '{gate_name}' in transpiled circuit.")

    return psi.reshape(2**num_qubits)


def _timed_mean(func, repeats: int, warmups: int = 1) -> float:
    for _ in range(warmups):
        func()

    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        durations.append(time.perf_counter() - start)

    return float(np.mean(durations))


def run_benchmark(
    qubit_list: tuple[int, ...] = (2, 4, 6, 8),
    depth_factor: int = 3,
    repeats: int = 5,
    seed: int = 100,
    output_png: str | Path = "testing/benchmark_aer_vs_own.png",
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []

    for num_qubits in qubit_list:
        depth = 10
        qc = random_circuit(num_qubits, depth, measure=False, seed=seed + num_qubits)
        qc_trans = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)

        aer_sim = AerSimulator(method="statevector")
        qc_aer = qc_trans.copy()
        qc_aer.save_statevector()
        qc_aer = transpile(qc_aer, aer_sim, optimization_level=0)

        state_aer = _run_aer_statevector(aer_sim, qc_aer)
        state_own = simulator_own(qc_trans)
        state_numba = _simulate_with_numba_gates(qc_trans)

        _assert_equivalent_up_to_global_phase(state_aer, state_own)
        _assert_equivalent_up_to_global_phase(state_aer, state_numba)

        aer_time = _timed_mean(lambda: _run_aer_statevector(aer_sim, qc_aer), repeats)
        own_time = _timed_mean(lambda: simulator_own(qc_trans), repeats)
        numba_time = _timed_mean(lambda: _simulate_with_numba_gates(qc_trans), repeats)

        rows.append(
            BenchmarkRow(
                qubits=num_qubits,
                aer_s=aer_time,
                own_s=own_time,
                numba_s=numba_time,
            )
        )

    _plot_results(rows, output_png)
    return rows


def _plot_results(rows: list[BenchmarkRow], output_png: str | Path) -> None:
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qubits = [row.qubits for row in rows]
    aer_times = [row.aer_s for row in rows]
    own_times = [row.own_s for row in rows]
    numba_times = [row.numba_s for row in rows]

    own_ratios = [row.own_ratio for row in rows]
    numba_ratios = [row.numba_ratio for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].plot(qubits, aer_times, marker="o", label="Qiskit Aer")
    axes[0].plot(qubits, own_times, marker="s", label="simulator_own (u_gate/cx)")
    axes[0].plot(
        qubits,
        numba_times,
        marker="^",
        label="numba gates (u_gate_numba/cx_gate_numba)",
    )
    axes[0].set_xlabel("Number of qubits")
    axes[0].set_ylabel("Mean runtime [s]")
    axes[0].set_title("Runtime vs qubits")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(qubits, own_ratios, marker="s", label="simulator_own / Aer")
    axes[1].plot(qubits, numba_ratios, marker="^", label="numba gates / Aer")
    axes[1].axhline(1.0, linestyle="--", linewidth=1.0, color="gray")
    axes[1].set_xlabel("Number of qubits")
    axes[1].set_ylabel("Runtime ratio")
    axes[1].set_title("Runtime ratio vs qubits")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Benchmark: Aer vs simulator_own vs numba gates")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _print_table(rows: list[BenchmarkRow]) -> None:
    print("qubits | aer_s | own_s | numba_s | own/aer | numba/aer")
    for row in rows:
        print(
            f"{row.qubits:>6} | {row.aer_s:>8.6f} | {row.own_s:>8.6f} | {row.numba_s:>8.6f} | {row.own_ratio:>7.3f} | {row.numba_ratio:>9.3f}"
        )


if __name__ == "__main__":
    benchmark_rows = run_benchmark()
    _print_table(benchmark_rows)
    print("Plot written to testing/benchmark_aer_vs_own.png")
