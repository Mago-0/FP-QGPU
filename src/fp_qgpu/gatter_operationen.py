import numpy as np
from qiskit import QuantumCircuit
import numba


def u_gate(
    number_of_qubits: int, acting_on: int, u: np.ndarray, vec: np.ndarray
) -> np.ndarray:
    num = number_of_qubits
    act_on = acting_on
    u_gate = u

    old_indices = [i for i in range(num)]
    new_indices = old_indices.copy()
    new_indices[act_on] = 51

    phi = np.einsum(u_gate, [51, act_on], vec, old_indices, new_indices)
    return phi


@numba.njit(cache=True)
def u_gate_numba(
    number_of_qubits: int, acting_on: int, u: np.ndarray, vec: np.ndarray
) -> np.ndarray:
    num = number_of_qubits
    q = num - acting_on - 1

    vec_flat = vec.reshape(vec.size)
    psi_flat = np.zeros(vec_flat.shape[0], dtype=np.complex128)

    lower_size = 2**q
    upper_size = 2 ** (num - q - 1)  # qiskit counts qubits from right to left

    bit_q = 2**q
    upper_shift = q + 1

    for lower in range(lower_size):
        idx_lower = lower
        for upper in range(upper_size):
            idx_upper = upper * (2**upper_shift)

            idx_0 = idx_lower + idx_upper + 0 * bit_q
            idx_1 = idx_lower + idx_upper + 1 * bit_q

            v0 = vec_flat[idx_0]
            v1 = vec_flat[idx_1]

            psi_flat[idx_0] = u[0, 0] * v0 + u[0, 1] * v1
            psi_flat[idx_1] = u[1, 0] * v0 + u[1, 1] * v1

    return psi_flat.reshape(vec.shape)


def cx(number_of_qubits: int, control: int, target: int, vec: np.ndarray) -> np.ndarray:
    num = number_of_qubits

    control = control
    target = target
    cx = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    ).reshape(2, 2, 2, 2)  # cx gate

    old_indices = [i for i in range(num)]
    new_indices = old_indices.copy()

    control_idx = num
    target_idk = control_idx + 1

    new_indices[control] = control_idx
    new_indices[target] = target_idk

    phi = np.einsum(
        cx, [control_idx, target_idk, control, target], vec, old_indices, new_indices
    )
    return phi


def extract_gates(transpiled_qc: QuantumCircuit) -> list[str]:
    gate_list: list[str] = []

    for gate in transpiled_qc.data:
        gate_list.append(gate.name)
        print(gate_list)

    return gate_list


def get_circuit(qc: QuantumCircuit) -> list[list[object]]:
    circuit: list[list[object]] = []
    for gate in qc.data:
        acting_on = [qc.find_bit(q).index for q in gate.qubits]
        circuit.append([gate.name, acting_on, gate.matrix])
        # print('other paramters (such as angles):', gate[0].params)
    return circuit  # containes information about each gate
