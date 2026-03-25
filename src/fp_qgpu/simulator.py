from fp_qgpu.gatter_operationen import get_circuit
from qiskit import QuantumCircuit
from fp_qgpu.gatter_operationen import u_gate, cx
import numpy as np


def simulator_own(transpiled_circuite: QuantumCircuit) -> None:
    num = transpiled_circuite.num_qubits
    circuit = get_circuit(transpiled_circuite)
    psi_vec = np.zeros(2**num, dtype=complex)
    psi_vec[0] = 1
    psi = np.reshape(psi_vec, [2] * num)

    for gate in circuit:
        name = gate[0]
        acting_on = gate[1]
        # acting_on = num - 1 - acting_on[0] # da die qubits in umgekehrter reihenfolge angeordnet sind, muss das acting_on angepasst werden
        matrix = gate[2]

        if name == "u":
            acting_on = num - 1 - acting_on[0]
            psi = u_gate(num, acting_on, matrix, psi)

        if name == "cx":
            control = num - 1 - acting_on[0]
            target = num - 1 - acting_on[1]
            psi = cx(num, control, target, psi)

        if name == "barrier":
            continue

        if name == "measure":
            continue

    psi = np.reshape(psi, 2**num)
    return psi
