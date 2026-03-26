How to generate benchmark runtime plots
=======================================

This project includes a benchmark test that compares the custom statevector
simulator against Qiskit Aer over increasing qubit counts.

Run the benchmark
-----------------

From the repository root, run:

.. code-block:: bash

   PYTHONPATH=src pytest testing/test_benchmark_statevector.py --benchmark-disable-gc

If you use ``uv``, run:

.. code-block:: bash

   PYTHONPATH=src uv run pytest testing/test_benchmark_statevector.py --benchmark-disable-gc

Output artifact
---------------

The benchmark writes the runtime comparison figure to:

- ``testing/.benchmarks/runtime_ratio_vs_qubits.png``

What the plot shows
-------------------

- Top panel: absolute runtime in seconds for both implementations.
- Bottom panel: ratio ``own / aer`` across qubit counts.

A ratio below ``1`` means the custom simulator is faster for that point; above
``1`` means Qiskit Aer is faster.
