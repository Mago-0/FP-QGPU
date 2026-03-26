import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit

from fp_qgpu.gatter_operationen import cx, u_gate
from fp_qgpu.gatter_operationen_numba import (
    cx_numba_compatible,
    cx_numba_compatible_three_loops,
    u_gate_numba_compatible,
    u_gate_numba_compatible_three_loops,
    u_gate_numba_compatible_two_loops,
)


def _simulate_with_gate_impls(
    transpiled_qc: QuantumCircuit, u_impl, cx_impl
) -> np.ndarray:
    """
    Simulate a transpiled circuit with injected U/CX gate implementations.
    """
    num_qubits = transpiled_qc.num_qubits
    psi_flat = np.zeros(2**num_qubits, dtype=complex)
    psi_flat[0] = 1.0 + 0.0j
    psi = psi_flat.reshape([2] * num_qubits)

    for instruction in transpiled_qc.data:
        name = instruction.operation.name

        if name == "u":
            q = transpiled_qc.find_bit(instruction.qubits[0]).index
            axis = num_qubits - 1 - q
            matrix = instruction.operation.to_matrix()
            psi = u_impl(num_qubits, axis, matrix, psi)
            continue

        if name == "cx":
            control_q = transpiled_qc.find_bit(instruction.qubits[0]).index
            target_q = transpiled_qc.find_bit(instruction.qubits[1]).index
            control_axis = num_qubits - 1 - control_q
            target_axis = num_qubits - 1 - target_q
            psi = cx_impl(num_qubits, control_axis, target_axis, psi)
            continue

        raise ValueError(f"Unexpected gate '{name}' in transpiled circuit.")

    return psi.reshape(2**num_qubits)


def _build_random_u_only_circuit(
    num_qubits: int, depth: int, seed: int
) -> QuantumCircuit:
    """
    Build random circuit and transpile so only U gates remain.
    """
    qc_random = random_circuit(
        num_qubits=num_qubits,
        depth=depth,
        max_operands=1,
        measure=False,
        seed=seed,
    )
    qc_u_only = transpile(qc_random, basis_gates=["u"], optimization_level=0)
    assert all(instr.operation.name == "u" for instr in qc_u_only.data)
    return qc_u_only


def _build_random_cx_only_circuit(
    num_qubits: int, depth: int, seed: int
) -> QuantumCircuit:
    """
    Build random circuit containing only CX gates.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)

    for _ in range(depth):
        control = int(rng.integers(0, num_qubits))
        target = int(rng.integers(0, num_qubits - 1))
        if target >= control:
            target += 1
        qc.cx(control, target)

    qc_cx_only = transpile(qc, basis_gates=["cx"], optimization_level=0)
    assert all(instr.operation.name == "cx" for instr in qc_cx_only.data)
    return qc_cx_only


def _build_random_ucx_circuit(
    num_qubits: int, depth: int, seed: int
) -> QuantumCircuit:
    """
    Build random circuit and transpile to U/CX basis.
    """
    qc_random = random_circuit(
        num_qubits=num_qubits,
        depth=depth,
        measure=False,
        seed=seed,
    )
    qc_ucx = transpile(qc_random, basis_gates=["u", "cx"], optimization_level=0)
    assert all(instr.operation.name in {"u", "cx"} for instr in qc_ucx.data)
    return qc_ucx


def test_u_implementations_on_random_u_only_circuits() -> None:
    for seed in (11, 12, 13):
        circuit = _build_random_u_only_circuit(num_qubits=5, depth=14, seed=seed)
        expected = _simulate_with_gate_impls(circuit, u_gate, cx)

        results = [
            _simulate_with_gate_impls(circuit, u_gate_numba_compatible, cx),
            _simulate_with_gate_impls(circuit, u_gate_numba_compatible_two_loops, cx),
            _simulate_with_gate_impls(circuit, u_gate_numba_compatible_three_loops, cx),
        ]

        for result in results:
            assert np.allclose(result, expected, atol=1e-12)


def test_cx_implementations_on_random_cx_only_circuits() -> None:
    for seed in (21, 22, 23):
        circuit = _build_random_cx_only_circuit(num_qubits=5, depth=30, seed=seed)
        expected = _simulate_with_gate_impls(circuit, u_gate, cx)

        results = [
            _simulate_with_gate_impls(circuit, u_gate, cx_numba_compatible),
            _simulate_with_gate_impls(circuit, u_gate, cx_numba_compatible_three_loops),
        ]

        for result in results:
            assert np.allclose(result, expected, atol=1e-12)


def test_all_implementations_on_random_transpiled_ucx_circuits() -> None:
    for seed in (31, 32, 33):
        circuit = _build_random_ucx_circuit(num_qubits=5, depth=12, seed=seed)
        expected = _simulate_with_gate_impls(circuit, u_gate, cx)

        implementation_pairs = [
            (u_gate_numba_compatible, cx_numba_compatible),
            (u_gate_numba_compatible, cx_numba_compatible_three_loops),
            (u_gate_numba_compatible_two_loops, cx_numba_compatible),
            (u_gate_numba_compatible_two_loops, cx_numba_compatible_three_loops),
            (u_gate_numba_compatible_three_loops, cx_numba_compatible),
            (u_gate_numba_compatible_three_loops, cx_numba_compatible_three_loops),
        ]

        for u_impl, cx_impl in implementation_pairs:
            result = _simulate_with_gate_impls(circuit, u_impl, cx_impl)
            assert np.allclose(result, expected, atol=1e-12)
