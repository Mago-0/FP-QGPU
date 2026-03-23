from fp_qgpu import hello
from fp_qgpu.pauli_matricies import pauli_x, pauli_y

print(hello())

b = 1 + 2

a = pauli_x() @ pauli_y()
print(f"this is sig_x* sig_Y: {a}")


def matrixaddition(a, b):
    return a() + b()


print(matrixaddition(pauli_x, pauli_y))
