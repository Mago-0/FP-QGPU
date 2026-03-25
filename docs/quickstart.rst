Quickstart
==========

This page walks through a minimal end-to-end FP-QGPU workflow.

1. Build a circuit.
2. Simulate with Aer-based helper.
3. Compare with the custom simulator.

Create a Circuit
----------------

.. code-block:: python

   from fp_qgpu.circuits import ghz_test

   qc = ghz_test(3)

Run the Mock Simulator
----------------------

.. code-block:: python

   from fp_qgpu.simulator_mock import simulator_mock

   counts, statevector = simulator_mock(qc, shots=1024, seed=20)
   print(counts)

Run the Custom Simulator
------------------------

.. code-block:: python

   from fp_qgpu.simulator import simulator_own

   psi = simulator_own(qc)
   print(psi)

What to Read Next
-----------------

* Go to :doc:`tutorials` for concrete usage patterns.
* Go to :doc:`architecture` to understand internals.
* Go to :doc:`api/index` for full API details.
