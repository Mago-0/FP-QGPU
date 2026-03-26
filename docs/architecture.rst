Architecture
============

Package Layout
--------------

* ``fp_qgpu.circuits``: Circuit construction utilities and demonstration helpers.
* ``fp_qgpu.gatter_operationen``: Gate-level tensor operations for custom simulation.
* ``fp_qgpu.gatter_operationen_numba``: Numba-compiled gate kernels for ``u`` and
	``cx`` with loop-based index traversal.
* ``fp_qgpu.simulator``: Custom statevector simulator based on tensor contractions.
* ``fp_qgpu.simulator_mock``: Qiskit Aer-backed simulator wrapper for reference behavior.
* ``fp_qgpu.pauli_matricies``: Pauli matrix helper functions.

Execution Flow
--------------

1. A Qiskit circuit is built or provided.
2. ``get_circuit`` extracts gate names, targets, and gate matrices.
3. ``simulator_own`` iteratively applies ``u_gate`` and ``cx`` to a tensor state.
4. The final state tensor is flattened into a statevector.

The repository also contains ``simulator_own_numba`` which applies
``u_gate_numba`` and ``cx_gate_numba`` for a JIT-compiled execution path.

For the Numba ``cx`` kernel, the state is traversed in three nested loops
(upper, middle, lower) over 4-state blocks defined by control/target bit
positions. Inside each block, only the affected pair is swapped in-place using
explicit source and target indices, avoiding full output-state allocation.
