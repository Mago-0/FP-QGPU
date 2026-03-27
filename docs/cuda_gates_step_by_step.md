# CUDA Gate Implementation (Step by Step)

This document explains how the CUDA versions of the `u` and `cx` gates are implemented in `src/fp_qgpu/gatter_operationen_cuda.py`.

## 1) Imports and CUDA kernel setup

```python
import math

import numpy as np
from numba import cuda
```

- `numba.cuda` provides the CUDA JIT compiler and GPU API.
- `numpy` is used for host-side array preparation.
- `math.ceil` is used to compute the number of CUDA blocks.

---

## 2) U gate kernel: `app_u_kernel()`

```python
@cuda.jit
def app_u_kernel(
    number_of_qubits: int,
    bit_position: int,
    u: np.ndarray,
    input_state: np.ndarray,
    output_state: np.ndarray,
) -> None:
    pair_index = cuda.grid(1)
    total_pairs = input_state.size // 2

    if pair_index >= total_pairs:
        return
```

### What happens here

- `@cuda.jit` compiles this function as a GPU kernel.
- `pair_index = cuda.grid(1)` gives each thread a unique linear index.
- For a single-qubit gate, amplitudes are updated in pairs (`|...0...>` and `|...1...>`), so there are `N/2` pairs for `N` amplitudes.
- Threads outside valid range return immediately.

### Index reconstruction (core idea)

```python
    lower_mask = (1 << bit_position) - 1
    upper = pair_index >> bit_position
    lower = pair_index & lower_mask

    idx0 = (upper << (bit_position + 1)) | lower
    idx1 = idx0 | (1 << bit_position)
```

- `idx0` and `idx1` differ only in `bit_position`.
- `idx0` corresponds to target bit `0`.
- `idx1` corresponds to target bit `1`.
- This avoids expensive per-thread loops over all qubits.

### Apply the 2x2 U matrix

```python
    amp0 = input_state[idx0]
    amp1 = input_state[idx1]

    u00 = u[0, 0]
    u01 = u[0, 1]
    u10 = u[1, 0]
    u11 = u[1, 1]

    output_state[idx0] = u00 * amp0 + u01 * amp1
    output_state[idx1] = u10 * amp0 + u11 * amp1
```

- Each thread reads one amplitude pair, applies the matrix multiply, and writes both results.
- Output is written to a separate array (`output_state`) to avoid write conflicts.

---

## 3) CX gate kernel: `app_cx_kernel()`

```python
@cuda.jit
def app_cx_kernel(
    control_bit_position: int,
    target_bit_position: int,
    state: np.ndarray,
) -> None:
    i = cuda.grid(1)

    if i >= state.size or control_bit_position == target_bit_position:
        return
```

### Why in-place works for CX

- A CX gate only swaps amplitudes between basis states when:
  - control bit is `1`, and
  - target bit flips (`0 <-> 1`).
- We can do this in place if each swap pair is handled exactly once.

### Bit-mask condition and one-direction swap

```python
    control_mask = 1 << control_bit_position
    target_mask = 1 << target_bit_position

    if (i & control_mask) != 0 and (i & target_mask) == 0:
        j = i | target_mask
        tmp = state[i]
        state[i] = state[j]
        state[j] = tmp
```

- `control_mask` checks whether control bit is `1`.
- `target_mask` checks whether target bit is `0`.
- Only indices with `target=0` perform swaps, so each pair is swapped once.

---

## 4) Host launcher for U: `u_gate_cuda()`

```python
def u_gate_cuda(
    number_of_qubits: int,
    bit_position: int,
    u: np.ndarray,
    vec: np.ndarray,
    threads_per_block: int = 256,
) -> np.ndarray:
```

### Steps

```python
    input_state = np.ascontiguousarray(vec.reshape(-1), dtype=np.complex128)
    output_state = np.empty_like(input_state)
    u_local = np.ascontiguousarray(u, dtype=np.complex128)
```

- Flatten state to 1D for efficient GPU indexing.
- Ensure contiguous memory and stable dtype (`complex128`).

```python
    d_input = cuda.to_device(input_state)
    d_output = cuda.device_array_like(output_state)
    d_u = cuda.to_device(u_local)
```

- Copy host arrays to device memory.

```python
    total_pairs = input_state.size // 2
    blocks_per_grid = math.ceil(total_pairs / threads_per_block)
```

- Grid size is computed from required number of threads.

```python
    app_u_kernel[blocks_per_grid, threads_per_block](
        number_of_qubits,
        bit_position,
        d_u,
        d_input,
        d_output,
    )

    return d_output.copy_to_host().reshape(vec.shape)
```

- Launch kernel.
- Copy result back to CPU and restore original tensor shape.

---

## 5) Host launcher for CX: `cx_gate_cuda()`

```python
def cx_gate_cuda(
    number_of_qubits: int,
    control_bit_position: int,
    target_bit_position: int,
    vec: np.ndarray,
    threads_per_block: int = 256,
) -> np.ndarray:
```

### Steps

```python
    _ = number_of_qubits  # Kept for API compatibility with other backends.
    state = np.ascontiguousarray(vec.reshape(-1), dtype=np.complex128)
    d_state = cuda.to_device(state)
```

- Flatten and copy state to GPU.
- `number_of_qubits` is currently unused but kept for signature compatibility.

```python
    blocks_per_grid = math.ceil(state.size / threads_per_block)

    app_cx_kernel[blocks_per_grid, threads_per_block](
        control_bit_position,
        target_bit_position,
        d_state,
    )

    return d_state.copy_to_host().reshape(vec.shape)
```

- One thread per amplitude index.
- Kernel applies conditional in-place swaps.
- Result is copied back and reshaped.

---

## 6) Integration in simulator

In `src/fp_qgpu/simulator.py`, `simulator_own_numba(..., use_cuda=False)` can now switch between:

- CPU Numba path (`u_gate_numba`, `cx_gate_numba`)
- CUDA path (`u_gate_cuda`, `cx_gate_cuda`) when `use_cuda=True`

It also checks CUDA availability and raises clear errors if CUDA is requested but unavailable.

