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
def u_gate_numba(
    number_of_qubits: int, acting_on: int, u: np.ndarray, vec: np.ndarray
) -> np.ndarray:
    """
    Apply a single-qubit gate without using numpy.einsum.

    This variant uses exactly two loops and explicit index composition to avoid
    per-element bit extraction in the inner update.
    """
    # Step 1: flatten input tensor state and allocate output.
    input_state = np.ascontiguousarray(vec.reshape(-1))
    state_size = input_state.size
    output_state = np.zeros(state_size, dtype=np.complex128)

    # Step 2: map the target tensor axis to the corresponding bit position q.
    q = _axis_to_bit_position(number_of_qubits, acting_on)
    two_pow_q = 2**q

    # Step 3: precompute loop ranges for the two-loop traversal.
    # upper: bits above q
    # lower: bits below q
    upper_count = 2 ** (number_of_qubits - q - 1)
    lower_count = 2**q
    upper_stride = 2 ** (q + 1)

    # Step 4: iterate over all (upper, lower) pairs and construct:
    #   idx0 = idx_lower + idx_upper + 0 * 2**q
    #   idx1 = idx_lower + idx_upper + 1 * 2**q
    for upper in range(upper_count):
        idx_upper = upper * upper_stride
        for lower in range(lower_count):
            idx_lower = lower

            idx0 = idx_lower + idx_upper + 0 * two_pow_q
            idx1 = idx_lower + idx_upper + 1 * two_pow_q

            amp0 = input_state[idx0]
            amp1 = input_state[idx1]

            output_state[idx0] = u[0, 0] * amp0 + u[0, 1] * amp1
            output_state[idx1] = u[1, 0] * amp0 + u[1, 1] * amp1

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
    # Step 1: flatten input tensor state.
    input_state = np.ascontiguousarray(vec.reshape(-1))

    # Step 2: map tensor axes to bit positions in flattened indexing.
    control_bit_position = _axis_to_bit_position(number_of_qubits, control)
    target_bit_position = _axis_to_bit_position(number_of_qubits, target)

    # Step 3: identify the higher and lower bit positions.
    # We iterate over blocks where these two bits are the only varying bits.
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

    # Step 4: precompute loop ranges for the three-loop traversal.
    # upper: bits above the higher of control/target
    # middle: bits between higher and lower
    # lower: bits below the lower of control/target
    upper_count = 2 ** (number_of_qubits - higher_bit_position - 1)
    middle_count = 2 ** (higher_bit_position - lower_bit_position - 1)
    lower_count = 2**lower_bit_position

    upper_stride = 2 ** (higher_bit_position + 1)
    middle_stride = 2 ** (lower_bit_position + 1)

    # Step 5: walk through the state in three nested loops.
    # For each (upper, middle, lower), we get one 4-state block:
    # i00, i01, i10, i11 for (higher_bit, lower_bit) in {0,1} x {0,1}.
    for upper in range(upper_count):
        upper_base = upper * upper_stride
        for middle in range(middle_count):
            middle_base = upper_base + middle * middle_stride
            for lower in range(lower_count):
                i00 = middle_base + lower
                i01 = i00 + lower_bit_weight
                i10 = i00 + higher_bit_weight
                i11 = i10 + lower_bit_weight

                # Step 6: apply CX routing inside each 4-state block.
                # Perform only the affected swap using explicit source/target
                # indices to avoid an additional output buffer.
                if control_is_higher_bit:
                    source_index = i10
                    target_index = i11
                else:
                    source_index = i01
                    target_index = i11

                tmp = input_state[source_index]
                input_state[source_index] = input_state[target_index]
                input_state[target_index] = tmp

    # Step 7: reshape to tensor form, matching existing API behavior.
    # Using vec.shape keeps the dimensionality explicit and Numba-friendly.
    return input_state.reshape(vec.shape)
