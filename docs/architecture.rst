Architecture
============

Package Layout
--------------

* ``fp_qgpu.circuits``: Circuit construction utilities and demonstration helpers.
* ``fp_qgpu.gatter_operationen``: Gate-level tensor operations for custom simulation.
* ``fp_qgpu.simulator``: Custom statevector simulator based on tensor contractions.
* ``fp_qgpu.simulator_mock``: Qiskit Aer-backed simulator wrapper for reference behavior.
* ``fp_qgpu.pauli_matricies``: Pauli matrix helper functions.

Execution Flow
--------------

1. A Qiskit circuit is built or provided.
2. ``get_circuit`` extracts gate names, targets, and gate matrices.
3. ``simulator_own`` iteratively applies ``u_gate`` and ``cx`` to a tensor state.
4. The final state tensor is flattened into a statevector.

Design Notes
------------

The custom simulator currently focuses on ``u`` and ``cx`` gate handling in transpiled circuits.
