Architecture
============

Package Layout
--------------

* ``fp_qgpu.circuits``: Circuit construction utilities and demonstration helpers.
* ``fp_qgpu.gatter_operationen``: Gate-level tensor operations for custom simulation.
* ``fp_qgpu.gatter_operationen_numba``: Numba-compiled gate kernels for ``u`` and
	``cx`` with loop-based index traversal.
* ``fp_qgpu.gatter_operationen_cuda``: CUDA kernels and host launchers for GPU
  execution of ``u`` and ``cx`` and a GPU-resident full-circuit loop.
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

CUDA Execution Path
-------------------

When ``simulator_own_numba(..., use_cuda=True)`` is requested:

1. CUDA module imports are validated during ``fp_qgpu.simulator`` import.
2. Runtime availability is validated using ``numba.cuda.is_available()``.
3. ``simulate_circuit_cuda`` initializes the state on device memory.
4. ``u`` gates are applied with ``app_u_kernel`` into a second device buffer.
5. Buffers are swapped after each ``u`` gate to avoid host copies.
6. ``cx`` gates are applied in-place with ``app_cx_kernel``.
7. Final state is copied to host once at the end.

CUDA kernels in ``fp_qgpu.gatter_operationen_cuda`` use grid-stride loops and an
occupancy-aware launch helper so launches can oversubscribe work safely while
keeping streaming multiprocessors active.

For the Numba ``cx`` kernel, the state is traversed in three nested loops
(upper, middle, lower) over 4-state blocks defined by control/target bit
positions. Inside each block, only the affected pair is swapped in-place using
explicit source and target indices, avoiding full output-state allocation.

For the CUDA ``cx`` kernel, a thread swaps amplitudes only when
``control=1`` and ``target=0``. This one-direction condition ensures each pair
is swapped exactly once in-place.
