"""Reference simulation wrapper around Qiskit Aer."""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np


def simulator_mock(
    qc: QuantumCircuit, shots: int = 1024, seed: int | None = None
) -> tuple[dict[str, int] | None, np.ndarray]:
    """Simulate a circuit and return measurement counts and statevector.

    If the circuit has no classical bits, counts are returned as ``None``.
    """
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
