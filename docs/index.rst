FP-QGPU Documentation
=====================

.. raw:: html

    <section class="hero">
       <p class="hero-kicker">Quantum Circuit Simulation Toolkit</p>
       <h1>FP-QGPU</h1>
       <p class="hero-lead">
          Build, transpile, and simulate quantum circuits with a compact Python API.
          This documentation follows a practical, guide-first structure inspired by scientific tooling docs.
       </p>
    </section>

.. raw:: html

    <section class="card-grid">
       <a class="doc-card" href="installation.html">
          <h3>Installation</h3>
          <p>Set up a local environment and install dependencies.</p>
       </a>
       <a class="doc-card" href="quickstart.html">
          <h3>Quickstart</h3>
          <p>Run your first GHZ simulation in minutes.</p>
       </a>
       <a class="doc-card" href="tutorials.html">
          <h3>Tutorials</h3>
          <p>Explore circuit construction and statevector workflows.</p>
       </a>
       <a class="doc-card" href="benchmarking.html">
          <h3>Benchmarking</h3>
          <p>Compare FP-QGPU and Qiskit Aer runtime over benchmark cases.</p>
       </a>
       <a class="doc-card" href="architecture.html">
          <h3>Architecture</h3>
          <p>Understand the package layout and execution model.</p>
       </a>
       <a class="doc-card" href="api/index.html">
          <h3>API Reference</h3>
          <p>Detailed reference for every public module and function.</p>
       </a>
       <a class="doc-card" href="notebooks/simulator_comparison.html">
          <h3>Notebook Demo</h3>
          <p>Visualize circuits and verify simulator parity in Jupyter format.</p>
       </a>
       <a class="doc-card" href="diataxis.html">
          <h3>Diataxis Map</h3>
          <p>Navigate tutorials, how-to guides, reference, and explanation.</p>
       </a>
       <a class="doc-card" href="references.html">
          <h3>References</h3>
          <p>External sources and technical background links.</p>
       </a>
    </section>

Getting Oriented
----------------

FP-QGPU provides:

* circuit builders (for example GHZ helpers)
* a Qiskit-based simulator wrapper
* a custom tensor-based simulator implementation
* low-level gate utilities used by the simulator core

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   quickstart
   notebooks/simulator_comparison

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: How-to Guides

   installation
   tutorials
   benchmarking

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api/index
   references

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Explanation

   architecture
   diataxis
