Benchmarking
============

This guide documents the random-circuit benchmark used to compare FP-QGPU
simulator variants against Qiskit Aer for final statevector computation.

Data Source
-----------

The current benchmark snapshot in this page is sourced from benchmark CSV/JSON
artifacts in ``testing/.benchmarks`` (or ``testing/testing/.benchmarks`` when
the benchmark is executed from inside the ``testing`` directory):

* ``random_circuit_benchmark_times.json``
* ``random_circuit_benchmark_times.csv``
* ``random_circuit_benchmark_simtime.csv``
* ``random_circuit_benchmark.png``
* ``statevector_runtime_vs_qubits.png``

Compared Implementations
------------------------

The benchmark compares up to four implementations:

* ``simulator_own``: baseline implementation using ``u_gate`` and ``cx`` from
   ``fp_qgpu.gatter_operationen``.
* ``numba_compiled``: Numba-compiled full-circuit path using
   ``simulate_circuit_numba_compiled`` from ``fp_qgpu.gatter_operationen_numba``.
* ``numba_cuda``: CUDA-backed path via ``simulator_own_numba(..., use_cuda=True)``
   from ``fp_qgpu.simulator`` (only when CUDA is available).
* ``qiskit_aer``: Aer statevector simulator reference backend.

The latest stored data includes ``simulator_own``, ``numba_compiled``, and
``qiskit_aer``. The ``numba_cuda`` series is only present when the benchmark is
executed on a CUDA-capable machine.

The Numba ``cx_gate_numba`` implementation uses structured three-loop block
traversal and performs in-place source/target swaps on the flattened statevector,
avoiding a full output-buffer allocation for CX updates.

Benchmark Cases
---------------

The benchmark script is defined in ``testing/benchmark_random_circuit_plot.py``.
The current stored dataset runs for odd qubit counts from 1 to 21:

* ``[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]``

For each case:

* Circuit depth is set to ``10``.
* A random circuit is generated with ``seed=1234 + num_qubits``.
* Aer is configured with ``method='statevector'``, ``fusion_enable=False``, and
   ``max_parallel_threads=1``.
* Warmup runs are executed before measurement for all active implementations.
* Repeats are ``7`` up to 15 qubits and ``3`` for 17 and higher.

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
* ``testing/.benchmarks/random_circuit_benchmark_simtime.csv``
* ``testing/.benchmarks/random_circuit_benchmark_times.json``
* ``testing/.benchmarks/statevector_runtime_vs_qubits.png``
* ``docs/_static/random_circuit_benchmark.png``
* ``docs/_generated/benchmark_random_circuit_results.rst``

Snapshot From ``testing/.benchmarks``
-------------------------------------

The runtime table below is generated from the benchmark CSV file, not hardcoded
values.

Latest Generated Results
------------------------

This section is auto-generated from the saved benchmark data:

.. include:: _generated/benchmark_random_circuit_results.rst
