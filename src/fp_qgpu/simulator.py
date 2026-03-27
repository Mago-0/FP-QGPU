from fp_qgpu.gatter_operationen import get_circuit
from qiskit import QuantumCircuit
from fp_qgpu.gatter_operationen import u_gate, cx
from fp_qgpu.gatter_operationen_numba import u_gate_numba, cx_gate_numba
import numpy as np
from typing import Callable

try:
    from fp_qgpu.gatter_operationen_cuda import (
        cx_gate_cuda,
        simulate_circuit_cuda,
        u_gate_cuda,
    )
    from numba import cuda

    CUDA_IMPORT_AVAILABLE = True
except ImportError:
    CUDA_IMPORT_AVAILABLE = False


def simulator_own(transpiled_circuite: QuantumCircuit) -> np.ndarray:
    num = transpiled_circuite.num_qubits
    circuit = get_circuit(transpiled_circuite)
    psi_vec = np.zeros(2**num, dtype=complex)
    psi_vec[0] = 1
    psi = np.reshape(psi_vec, [2] * num)

    for gate in circuit:
        name = gate[0]
        acting_on = gate[1]
        # acting_on = num - 1 - acting_on[0] # da die qubits in umgekehrter reihenfolge angeordnet sind, muss das acting_on angepasst werden
        matrix = gate[2]

        if name == "u":
            acting_on = num - 1 - acting_on[0]
            psi = u_gate(num, acting_on, matrix, psi)

        if name == "cx":
            control = num - 1 - acting_on[0]
            target = num - 1 - acting_on[1]
            psi = cx(num, control, target, psi)

        if name == "barrier":
            continue

        if name == "measure":
            continue

    psi = np.reshape(psi, 2**num)
    return psi


def _simulate_circuit_with_ops(
    transpiled_circuite: QuantumCircuit,
    u_op: Callable[[int, int, np.ndarray, np.ndarray], np.ndarray],
    cx_op: Callable[[int, int, int, np.ndarray], np.ndarray],
) -> np.ndarray:
    num = transpiled_circuite.num_qubits
    circuit = get_circuit(transpiled_circuite)
    psi_vec = np.zeros(2**num, dtype=complex)
    psi_vec[0] = 1
    psi = np.reshape(psi_vec, [2] * num)

    for gate in circuit:
        name = gate[0]
        acting_on = gate[1]
        # acting_on = num - 1 - acting_on[0] # da die qubits in umgekehrter reihenfolge angeordnet sind, muss das acting_on angepasst werden
        matrix = gate[2]

        if name == "u":
            acting_on = num - 1 - acting_on[0]
            psi = u_op(num, acting_on, matrix, psi)

        if name == "cx":
            control = num - 1 - acting_on[0]
            target = num - 1 - acting_on[1]
            psi = cx_op(num, control, target, psi)

        if name == "barrier":
            continue

        if name == "measure":
            continue

    psi = np.reshape(psi, 2**num)
    return psi


def simulator_own_numba(
    transpiled_circuite: QuantumCircuit, use_cuda: bool = False
) -> np.ndarray:
    if not use_cuda:
        return _simulate_circuit_with_ops(transpiled_circuite, u_gate_numba, cx_gate_numba)

    if not CUDA_IMPORT_AVAILABLE:
        raise RuntimeError(
            "CUDA path requested, but CUDA modules are unavailable. "
            "Install CUDA-enabled Numba and ensure fp_qgpu.gatter_operationen_cuda exists."
        )

    if not cuda.is_available():
        raise RuntimeError("CUDA path requested, but no CUDA-capable GPU is available.")

    num = transpiled_circuite.num_qubits
    circuit = get_circuit(transpiled_circuite)
    return simulate_circuit_cuda(num, circuit)
