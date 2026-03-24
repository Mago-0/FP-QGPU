import numpy as np


def u_gate(number_of_qubits, acting_on, alpha, beta, gamma):
    num = number_of_qubits
    psi_vec = np.zeros(2**num, dtype=complex)
    psi_vec[0] = 1
    psi = np.reshape(psi_vec, [2] * num)

    act_on = acting_on
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)  # ersetzten mit u
    # u_gate = np.array(
    #     [np.cos(alpha), -np.exp(1j * beta) * np.sin(alpha / 2)],
    #     [
    #         np.exp(1j * gamma) * np.sin(alpha / 2),
    #         np.exp(1j * (gamma + beta)) * np.cos(alpha / 2),
    #     ],
    # )

    old_indices = [i for i in range(num)]
    new_indices = old_indices.copy()
    new_indices[act_on] = 51

    phi = np.einsum(sigma_x, [51, act_on], psi, old_indices, new_indices)
    phi_vec = np.reshape(phi, 2**num)
    print(phi_vec)


def cx(number_of_qubits, control, target):
    num = number_of_qubits
    psi_vec = np.zeros(2**num, dtype=complex)
    psi_vec[2] = 1
    psi = np.reshape(psi_vec, [2] * num)

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
        cx, [control_idx, target_idk, control, target], psi, old_indices, new_indices
    )
    phi_vec = np.reshape(phi, 2**num)
    print(phi_vec)


cx(2, 0, 1)
