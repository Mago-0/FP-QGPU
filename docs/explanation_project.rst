Why FP-QGPU is structured this way
==================================

FP-QGPU combines two complementary simulation paths:

- A **reference path** via Qiskit Aer (``simulator_mock``) to provide trusted
  behavior and baseline outputs.
- A **custom NumPy-based path** (``simulator_own``) to explore gate-application
  mechanics and internal statevector evolution.

This split supports both correctness checking and experimentation. The Aer path
answers "what output should I expect?", while the custom path answers "how is
the state transformed step by step?"

Why transpilation to ``u`` and ``cx``?
--------------------------------------

The custom simulator currently applies a restricted gate basis. Transpiling to
``u`` and ``cx`` provides a predictable representation for gate extraction and
tensor-based application logic.

Why notebooks and static circuit plots?
---------------------------------------

Notebooks in ``examples/`` provide practical, executable demonstrations.
Generated static plots in ``docs/_static/`` keep Sphinx pages stable and easy
to browse without requiring runtime execution during docs builds.
