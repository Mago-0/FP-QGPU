import numpy as np
from qiskit import QuantumCircuit
from fp_qgpu.circuits import simple00


def u_gate(number_of_qubits, acting_on, u, vec):
    num = number_of_qubits
    act_on = acting_on
    u_gate = u

    old_indices = [i for i in range(num)]
    new_indices = old_indices.copy()
    new_indices[act_on] = 51

    phi = np.einsum(u_gate, [51, act_on], vec, old_indices, new_indices)
    return phi


def cx(number_of_qubits, control, target, vec):
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


def extract_gates(transpiled_qc):
    gate_list = []

    for gate in transpiled_qc.data:
        gate_list.append(gate.name)
        print(gate_list)

    return gate_list


def get_circuit(qc: QuantumCircuit) -> None:
    circuit = []
    for gate in qc.data:
        acting_on = [qc.find_bit(q).index for q in gate.qubits]
        circuit.append([gate.name, acting_on, gate.matrix])
        # print('other paramters (such as angles):', gate[0].params)
    return circuit  # containes information about each gate


print(get_circuit(simple00()))
