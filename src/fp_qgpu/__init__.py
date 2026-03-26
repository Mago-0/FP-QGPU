from fp_qgpu.circuits import ghz, ghz_example, ghz_test, simple00, simple01
from fp_qgpu.gatter_operationen import u_gate, cx
from fp_qgpu.gatter_operationen_numba import cx_gate_numba, u_gate_numba
from fp_qgpu.simulator import simulator_own
from fp_qgpu.simulator_mock import simulator_mock


def hello() -> str:
    return "Hello from fp-qgpu!"


__all__ = [
    "ghz",
    "ghz_example",
    "ghz_test",
    "simple00",
    "simple01",
    "u_gate",
    "cx",
    "u_gate_numba",
    "cx_gate_numba",
    "simulator_own",
    "simulator_mock",
]
