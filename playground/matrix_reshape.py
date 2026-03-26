import numpy as np

n = 2

psi = np.zeros(2**n, dtype=complex)
psi[0] = 1.0
psi = psi.reshape(n, n)

print(f"\nTensorform psi[i,j]:\n{psi}")

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)

psi_x1 = np.einsum("ai,ij->aj", sigma_x, psi)
print(f"\nTensorform sigma_x @ psi:\n{psi_x1}")

psi_x1 = np.reshape(psi_x1, 2**n)
print(f"\nVectorform sigma_x @ psi:\n{psi_x1}")

print(list(range(1,22,2)))