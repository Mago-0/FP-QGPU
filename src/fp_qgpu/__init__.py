"""Public package exports for FP-QGPU."""

def hello() -> str:
    """Return a minimal package health-check string."""
    return "Hello from fp-qgpu!"

from fp_qgpu.circuits import ghz, ghz_example, ghz_test, simple00, simple01
from fp_qgpu.gatter_operationen import u_gate, cx
from fp_qgpu.simulator import simulator_own
from fp_qgpu.simulator_mock import simulator_mock

__all__ = [
    "ghz",
    "ghz_example",
    "ghz_test",
    "simple00",
    "simple01",
    "u_gate",
    "cx",
    "simulator_own",
    "simulator_mock",
]
