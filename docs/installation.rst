Installation
============

Requirements
------------

- Python 3.13 or newer
- A working C/C++ toolchain compatible with NumPy/Qiskit dependencies

Install from repository
-----------------------

Clone the repository and install dependencies with ``uv``:

.. code-block:: bash

   uv sync

For development (tests + docs):

.. code-block:: bash

   uv sync --group dev

Quick check
-----------

Run tests:

.. code-block:: bash

   uv run pytest -q

Build documentation:

.. code-block:: bash

   cd docs
   uv run make html
