Benchmarking
============

This guide documents the random-circuit benchmark used to compare FP-QGPU
simulator variants against Qiskit Aer for final statevector computation.

Compared Implementations
------------------------

The benchmark compares three implementations:

* ``simulator_own``: baseline implementation using ``u_gate`` and ``cx`` from
   ``fp_qgpu.gatter_operationen``.
* ``numba_compiled``: Numba-compiled full-circuit path using
   ``simulate_circuit_numba_compiled`` from ``fp_qgpu.gatter_operationen_numba``.
* ``qiskit_aer``: Aer statevector simulator reference backend.

The Numba ``cx_gate_numba`` implementation uses structured three-loop block
traversal and performs in-place source/target swaps on the flattened statevector,
avoiding a full output-buffer allocation for CX updates.

Benchmark Cases
---------------

The benchmark script is defined in ``testing/benchmark_random_circuit_plot.py``.
It runs for odd qubit counts from 1 to 19:

* ``[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]``

For each case:

* Circuit depth is set to ``max(8, num_qubits * 3)``.
* A random circuit is generated with ``seed=1234 + num_qubits``.
* Aer is configured with ``method='statevector'``, ``fusion_enable=False``, and
   ``max_parallel_threads=1``.
* Warmup runs are executed before measurement for all three implementations.
* Repeats are ``7`` up to 15 qubits and ``3`` for 17 and 19 qubits.

Run the Benchmark
-----------------

From the repository root:

.. code-block:: bash

   python testing/benchmark_random_circuit_plot.py

Saved Outputs
-------------

Running the benchmark script updates all of the following automatically:

* ``testing/.benchmarks/random_circuit_benchmark.png``
* ``testing/.benchmarks/random_circuit_benchmark_times.csv``
* ``testing/.benchmarks/random_circuit_benchmark_times.json``
* ``docs/_static/random_circuit_benchmark.png``
* ``docs/_generated/benchmark_random_circuit_results.rst``

Latest Generated Results
------------------------

This section is auto-generated from the saved benchmark data:

.. include:: _generated/benchmark_random_circuit_results.rst
