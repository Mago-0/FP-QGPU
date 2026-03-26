How to install FP-QGPU
======================

Requirements
------------

- Python 3.13 or newer
- A working C/C++ toolchain compatible with NumPy/Qiskit dependencies

Install from repository
-----------------------

.. code-block:: bash

   uv sync

For development (tests + docs):

.. code-block:: bash

   uv sync --group dev

Verify installation
-------------------

Run tests:

.. code-block:: bash

   uv run pytest -q

Build documentation:

.. code-block:: bash

   cd docs
   uv run make html
