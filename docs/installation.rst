Installation
============

Requirements
------------

* Python 3.13+
* uv

Create Environment And Sync Dependencies
----------------------------------------
Install uv

.. code-block:: powershell

	pip install uv

Clone the repository and create a new uv environment:

.. code-block:: powershell

	git clone https://github.com/Mago-0/FP-QGPU.git

Sync dependencies and create a virtual environment:

.. code-block:: powershell

	cd FP-QGPU
	uv sync

Verify Installation
-------------------

.. code-block:: powershell

	uv run pytest -q
