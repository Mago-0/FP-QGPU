Tutorials
=========

GHZ State Workflow
------------------

Use :func:`fp_qgpu.circuits.ghz_test` to produce a transpiled circuit and then compare simulator outputs.

.. code-block:: python

   import numpy as np
   from fp_qgpu.circuits import ghz_test
   from fp_qgpu.simulator import simulator_own
   from fp_qgpu.simulator_mock import simulator_mock

   qc = ghz_test(4)
   counts, state_aer = simulator_mock(qc, shots=2048, seed=42)
   state_custom = simulator_own(qc)

   i = int(np.argmax(np.abs(state_aer)))
   phase = state_aer[i] / state_custom[i]
   print(np.allclose(state_aer, state_custom * phase, atol=1e-12))

QFT Experiment
--------------

Use :func:`fp_qgpu.circuits.qft` and :func:`fp_qgpu.circuits.qft_superpos` for exploratory experiments with the quantum Fourier transform.

Notes
-----

These functions are visualization-focused and display histograms directly.
