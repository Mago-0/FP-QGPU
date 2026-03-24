from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile


# einfacher Test-Circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

thisdict = {"qc": qc, "shots": 1024, "seed": 42}


def simulator_mock(qc: QuantumCircuit, shots: int = 1024, seed: int | None = None):
    simulator = AerSimulator(seed_simulator=seed)

    # Transpile
    qc.measure_all()
    circ = transpile(qc, simulator)

    # Run
    result = simulator.run(circ, shots=shots).result()
    counts = result.get_counts(circ)

    return counts


result = simulator_mock(**thisdict)
print(result)
