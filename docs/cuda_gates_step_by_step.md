# CUDA Gate Implementation (Step by Step)

This document explains how the CUDA path is implemented in
`src/fp_qgpu/gatter_operationen_cuda.py`.

## 1) Imports and setup

```python
import math

import numpy as np
from numba import cuda
```

- `numba.cuda` provides the CUDA JIT compiler and GPU API.
- `numpy` is used for host-side array preparation.
- `math` is used for launch-size calculations.

---

## 2) Launch helper: `_launch_config()`

```python
def _launch_config(total_work_items: int, threads_per_block: int) -> tuple[int, int]:
    threads = max(1, min(threads_per_block, 1024))
    required_blocks = max(1, math.ceil(total_work_items / threads))
    min_occupancy_blocks = max(1, 2 * cuda.get_current_device().MULTIPROCESSOR_COUNT)
    blocks = max(required_blocks, min_occupancy_blocks)
    return blocks, threads
```

- Bounds threads per block to CUDA limits.
- Computes enough blocks for the workload.
- Enforces a minimum block count based on SM count for better occupancy.

---

## 3) U gate kernel: `app_u_kernel()`

```python
@cuda.jit
def app_u_kernel(
    bit_position: int,
    input_state: np.ndarray,
    output_state: np.ndarray,
    u00: complex,
    u01: complex,
    u10: complex,
    u11: complex,
) -> None:
    pair_index = cuda.grid(1)
    stride = cuda.gridsize(1)
    total_pairs = input_state.size // 2

    while pair_index < total_pairs:
        lower_mask = (1 << bit_position) - 1
        upper = pair_index >> bit_position
        lower = pair_index & lower_mask

        idx0 = (upper << (bit_position + 1)) | lower
        idx1 = idx0 | (1 << bit_position)

        amp0 = input_state[idx0]
        amp1 = input_state[idx1]

        output_state[idx0] = u00 * amp0 + u01 * amp1
        output_state[idx1] = u10 * amp0 + u11 * amp1
        pair_index += stride
```

What this does:

- Uses one thread per amplitude pair (`|...0...>`, `|...1...>`).
- Reconstructs the paired indices with bit operations.
- Uses a grid-stride loop (`pair_index += stride`) so launches can safely use
  more blocks than strictly required.
- Accepts matrix entries as scalar args (`u00..u11`) instead of passing a
  matrix array to the kernel.

---

## 4) CX gate kernel: `app_cx_kernel()`

```python
@cuda.jit
def app_cx_kernel(
    control_bit_position: int,
    target_bit_position: int,
    state: np.ndarray,
) -> None:
    i = cuda.grid(1)
    stride = cuda.gridsize(1)

    if control_bit_position == target_bit_position:
        return

    control_mask = 1 << control_bit_position
    target_mask = 1 << target_bit_position

    while i < state.size:
        if (i & control_mask) != 0 and (i & target_mask) == 0:
            j = i | target_mask
            tmp = state[i]
            state[i] = state[j]
            state[j] = tmp
        i += stride
```

Why this is correct:

- The swap is applied only for indices where control is `1` and target is `0`.
- That one-direction condition guarantees each pair is swapped exactly once.
- The operation is in-place, so no second buffer is needed for CX.

---

## 5) Host launcher for U: `u_gate_cuda()`

```python
def u_gate_cuda(
    number_of_qubits: int,
    acting_on: int,
    u: np.ndarray,
    vec: np.ndarray,
    threads_per_block: int = 256,
) -> np.ndarray:
```

Main steps:

```python
bit_position = _axis_to_bit_position(number_of_qubits, acting_on)
original_shape = vec.shape
input_state = np.ascontiguousarray(vec.reshape(-1), dtype=np.complex128)
u_local = np.ascontiguousarray(u, dtype=np.complex128)

d_input = cuda.device_array(input_state.shape, dtype=np.complex128)
d_output = cuda.device_array(input_state.shape, dtype=np.complex128)
d_input.copy_to_device(input_state)

blocks_per_grid, threads = _launch_config(input_state.size // 2, threads_per_block)
app_u_kernel[blocks_per_grid, threads](
    bit_position,
    d_input,
    d_output,
    u_local[0, 0],
    u_local[0, 1],
    u_local[1, 0],
    u_local[1, 1],
)

return d_output.copy_to_host().reshape(original_shape)
```

---

## 6) Host launcher for CX: `cx_gate_cuda()`

```python
def cx_gate_cuda(
    number_of_qubits: int,
    control: int,
    target: int,
    vec: np.ndarray,
    threads_per_block: int = 256,
) -> np.ndarray:
```

Main steps:

```python
control_bit_position = _axis_to_bit_position(number_of_qubits, control)
target_bit_position = _axis_to_bit_position(number_of_qubits, target)
original_shape = vec.shape
state = np.ascontiguousarray(vec.reshape(-1), dtype=np.complex128)

d_state = cuda.device_array(state.shape, dtype=np.complex128)
d_state.copy_to_device(state)

blocks_per_grid, threads = _launch_config(state.size, threads_per_block)
app_cx_kernel[blocks_per_grid, threads](
    control_bit_position,
    target_bit_position,
    d_state,
)

return d_state.copy_to_host().reshape(original_shape)
```

---

## 7) Full-circuit CUDA path: `simulate_circuit_cuda()`

`simulate_circuit_cuda()` keeps state on device through the full gate loop:

- Initializes `d_state_a` and `d_state_b` on GPU.
- For each `u` gate, writes from `d_state_a` to `d_state_b`, then swaps buffers.
- For each `cx` gate, updates `d_state_a` in place.
- Copies back to host once after all gates are complete.

This reduces host-device transfer overhead compared to per-gate copies.

---

## 8) Integration in simulator

In `src/fp_qgpu/simulator.py`, `simulator_own_numba(..., use_cuda=False)` can
switch between:

- CPU Numba path (`u_gate_numba`, `cx_gate_numba`)
- CUDA full-circuit path (`simulate_circuit_cuda`) when `use_cuda=True`

The simulator also checks CUDA availability and raises clear errors if CUDA is
requested but unavailable.
