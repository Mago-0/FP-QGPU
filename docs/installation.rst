Installation
============

Requirements
------------

* Python 3.13+
* uv

Create Environment And Sync Dependencies
----------------------------------------

.. code-block:: powershell

	uv venv
	.\.venv\Scripts\Activate.ps1
	uv sync --all-extras

Verify Installation
-------------------

.. code-block:: powershell

	uv run pytest -q

Troubleshooting
---------------

If Qiskit Aer is not available, re-sync dependencies:

.. code-block:: powershell

	uv sync --all-extras --reinstall
