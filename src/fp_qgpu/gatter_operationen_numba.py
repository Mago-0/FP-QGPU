import numpy as np
import numba


@numba.njit(cache=True)
def _axis_to_bit_position(number_of_qubits: int, axis_index: int) -> int:
    """
    Convert a tensor axis index to the corresponding bit position in the flat index.

    Tensor state shape is [2] * number_of_qubits in C-order.
    Axis 0 is the most significant bit in the flattened basis index.
    """
    return number_of_qubits - 1 - axis_index


@numba.njit(cache=True)
def _u_gate_flat_inplace(
    number_of_qubits: int,
    bit_position: int,
    u: np.ndarray,
    input_state: np.ndarray,
    output_state: np.ndarray,
) -> None:
    two_pow_q = 2**bit_position
    upper_count = 2 ** (number_of_qubits - bit_position - 1)
    lower_count = 2**bit_position
    upper_stride = 2 ** (bit_position + 1)

    u00 = u[0, 0]
    u01 = u[0, 1]
    u10 = u[1, 0]
    u11 = u[1, 1]

    for upper in range(upper_count):
        idx_upper = upper * upper_stride
        for lower in range(lower_count):
            idx0 = idx_upper + lower
            idx1 = idx0 + two_pow_q

            amp0 = input_state[idx0]
            amp1 = input_state[idx1]

            output_state[idx0] = u00 * amp0 + u01 * amp1
            output_state[idx1] = u10 * amp0 + u11 * amp1


@numba.njit(cache=True)
def _cx_gate_flat_inplace(
    number_of_qubits: int,
    control_bit_position: int,
    target_bit_position: int,
    state: np.ndarray,
) -> None:
    if control_bit_position > target_bit_position:
        higher_bit_position = control_bit_position
        lower_bit_position = target_bit_position
        control_is_higher_bit = True
    else:
        higher_bit_position = target_bit_position
        lower_bit_position = control_bit_position
        control_is_higher_bit = False

    higher_bit_weight = 2**higher_bit_position
    lower_bit_weight = 2**lower_bit_position

    upper_count = 2 ** (number_of_qubits - higher_bit_position - 1)
    middle_count = 2 ** (higher_bit_position - lower_bit_position - 1)
    lower_count = 2**lower_bit_position

    upper_stride = 2 ** (higher_bit_position + 1)
    middle_stride = 2 ** (lower_bit_position + 1)

    for upper in range(upper_count):
        upper_base = upper * upper_stride
        for middle in range(middle_count):
            middle_base = upper_base + middle * middle_stride
            for lower in range(lower_count):
                i00 = middle_base + lower
                i01 = i00 + lower_bit_weight
                i10 = i00 + higher_bit_weight
                i11 = i10 + lower_bit_weight

                if control_is_higher_bit:
                    source_index = i10
                    target_index = i11
                else:
                    source_index = i01
                    target_index = i11

                tmp = state[source_index]
                state[source_index] = state[target_index]
                state[target_index] = tmp


@numba.njit(cache=True)
def u_gate_numba(
    number_of_qubits: int, acting_on: int, u: np.ndarray, vec: np.ndarray
) -> np.ndarray:
    """
    Apply a single-qubit gate without using numpy.einsum.

    This variant uses exactly two loops and explicit index composition to avoid
    per-element bit extraction in the inner update.
    """
    input_state = np.ascontiguousarray(vec.reshape(-1))
    output_state = np.empty(input_state.size, dtype=np.complex128)

    q = _axis_to_bit_position(number_of_qubits, acting_on)
    _u_gate_flat_inplace(number_of_qubits, q, u, input_state, output_state)

    # Step 5: reshape to tensor form, matching existing API behavior.
    # Using vec.shape keeps the dimensionality explicit and Numba-friendly.
    return output_state.reshape(vec.shape)


@numba.njit(cache=True)
def cx_gate_numba(
    number_of_qubits: int, control: int, target: int, vec: np.ndarray
) -> np.ndarray:
    """
    Apply a CX gate without using numpy.einsum.

    This variant avoids per-index bit extraction by traversing the state in
    structured blocks with exactly three nested loops.
    """
    input_state = np.ascontiguousarray(vec.reshape(-1))
    control_bit_position = _axis_to_bit_position(number_of_qubits, control)
    target_bit_position = _axis_to_bit_position(number_of_qubits, target)
    _cx_gate_flat_inplace(
        number_of_qubits, control_bit_position, target_bit_position, input_state
    )

    return input_state.reshape(vec.shape)


@numba.njit(cache=True)
def simulate_circuit_numba_compiled(
    number_of_qubits: int,
    gate_kinds: np.ndarray,
    u_axes: np.ndarray,
    u_mats: np.ndarray,
    cx_controls: np.ndarray,
    cx_targets: np.ndarray,
) -> np.ndarray:
    state_size = 2**number_of_qubits
    state_a = np.zeros(state_size, dtype=np.complex128)
    state_a[0] = 1.0 + 0.0j
    state_b = np.empty(state_size, dtype=np.complex128)

    u_index = 0
    cx_index = 0

    for gate_kind in gate_kinds:
        if gate_kind == 0:
            axis = u_axes[u_index]
            bit_position = _axis_to_bit_position(number_of_qubits, axis)
            _u_gate_flat_inplace(
                number_of_qubits,
                bit_position,
                u_mats[u_index],
                state_a,
                state_b,
            )

            tmp = state_a
            state_a = state_b
            state_b = tmp
            u_index += 1
        else:
            control_axis = cx_controls[cx_index]
            target_axis = cx_targets[cx_index]
            control_bit_position = _axis_to_bit_position(number_of_qubits, control_axis)
            target_bit_position = _axis_to_bit_position(number_of_qubits, target_axis)
            _cx_gate_flat_inplace(
                number_of_qubits,
                control_bit_position,
                target_bit_position,
                state_a,
            )
            cx_index += 1

    return state_a
