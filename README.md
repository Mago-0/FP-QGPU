# FP-QGPU
Repo zum fortgeschrittenen Praktrikum an der Universität Stuttgart zur Simulation von Quantenschaltungen auf GPUs

Author: Marco Gerhardt, Vincent Beguin

## Examples

Jupyter notebook examples are available in `examples/`:

- `examples/quickstart.ipynb`
- `examples/custom_simulator.ipynb`
- `examples/ghz_example.ipynb`
- `examples/circuit_plots.ipynb` (generates `docs/_static/circuit_*.png`)

## Benchmark runtime plot

To generate a runtime plot over qubit count (both implementations + ratio), run:

```bash
PYTHONPATH=src pytest testing/test_benchmark_statevector.py --benchmark-disable-gc
```

Or with `uv`:

```bash
PYTHONPATH=src uv run pytest testing/test_benchmark_statevector.py --benchmark-disable-gc
```

The plot is written to:

- `testing/.benchmarks/runtime_ratio_vs_qubits.png`
