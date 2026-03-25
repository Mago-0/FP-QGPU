Installation
============

Requirements
------------

* Python 3.13+
* A virtual environment tool (recommended)

Create Environment
------------------

.. code-block:: powershell

	python -m venv .venv
	.\.venv\Scripts\Activate.ps1

Install Package
---------------

From the repository root:

.. code-block:: powershell

	pip install -e .

Install Development Dependencies
--------------------------------

.. code-block:: powershell

	pip install -e .[dev]

Verify Installation
-------------------

.. code-block:: powershell

	python -m pytest -q

Troubleshooting
---------------

If Qiskit Aer is not available, reinstall dependencies inside your active virtual environment.

.. code-block:: powershell

	pip install --upgrade qiskit qiskit-aer
