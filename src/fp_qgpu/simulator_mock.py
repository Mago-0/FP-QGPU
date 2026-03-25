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

    # Get statevector
    qc_st = qc.remove_final_measurements(inplace=False)
    qc_st.save_statevector()  # tell qiskit what we want to get from the simulator
    circ_st = transpile(qc_st, simulator)
    result_st = simulator.run(circ_st).result()
    state_vector = result_st.get_statevector(circ_st)  # extract statevector from result

    # Get counts
    circ = transpile(qc, simulator)
    result = simulator.run(circ, shots=shots).result()

    if qc.num_clbits > 0:
        counts = result.get_counts(circ)
    else:
        counts = None
    return counts, state_vector


result = simulator_mock(**thisdict)
print(result)
