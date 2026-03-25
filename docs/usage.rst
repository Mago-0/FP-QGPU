Usage
=====

Basic import
------------

.. code-block:: python

   from fp_qgpu import simple00, simulator_mock, simulator_own
   from qiskit import transpile

Reference simulation with Aer
-----------------------------

Use ``simulator_mock`` to run a circuit with Qiskit Aer and retrieve both
counts (if measurements are present) and the statevector:

.. code-block:: python

   qc = simple00()
   qc.measure_all()
   counts, statevector = simulator_mock(qc, shots=1024, seed=42)

Custom simulator path
---------------------

The custom simulator currently expects a circuit transpiled to ``u`` and ``cx``
gates:

.. code-block:: python

   qc = simple00()
   qc_transpiled = transpile(qc, basis_gates=["u", "cx"])
   psi = simulator_own(qc_transpiled)

Available example circuits
--------------------------

- ``simple00()``: Bell-state style 2-qubit example
- ``simple01()``: small hand-written 2-qubit gate sequence
- ``ghz_test(n)``: GHZ-style circuit without measurements
- ``ghz(n)``: GHZ-style circuit with measurements and histogram visualization
- ``ghz_example(n=3)``: GHZ-style example using the mock simulator
