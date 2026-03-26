How to run example notebooks
============================

FP-QGPU includes Jupyter notebooks in the ``examples/`` directory.

Available notebooks
-------------------

- ``examples/quickstart.ipynb``: Basic import and reference simulation flow
- ``examples/custom_simulator.ipynb``: Use ``simulator_own`` with transpiled circuits
- ``examples/ghz_example.ipynb``: GHZ workflow and simulator comparison
- ``examples/circuit_plots.ipynb``: Generate circuit images for documentation

Run notebooks
-------------

From the repository root:

.. code-block:: bash

   uv run jupyter notebook

Open the notebooks from ``examples/`` in Jupyter.
