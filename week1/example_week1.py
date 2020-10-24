import random
import quimb.tensor as qtn
import quimb as qu
%config InlineBackend.figure_formats = ['svg']

# 10 qubits and tag the initial wavefunction tensors
circ = qtn.Circuit(N=10, tags='PSI0')

# initial layer of hadamards
for i in range(10):
    circ.apply_gate('H', i, gate_round=0)

# # 8 rounds of entangling gates
for r in range(1, 9):
    #
    #     # even pairs
    #     for i in range(0, 10, 2):
    #         circ.apply_gate('CNOT', i, i + 1, gate_round=r)
    #
    # Y-rotations
    for i in range(10):
        circ.apply_gate('RY', 1.234, i, gate_round=r)
#
#     # odd pairs
#     for i in range(1, 9, 2):
#         circ.apply_gate('CZ', i, i + 1, gate_round=r)

# final layer of hadamards
for i in range(10):
    circ.apply_gate('H', i, gate_round=r + 1)

circ
circ.psi.graph(color=['PSI0', 'H', 'RY'])

#
# random.seed(10)
#
# bitstring = "".join(random.choice('01') for _ in range(10))
# print(bitstring)

# the squeeze removes all size 1 bonds
# psi_sample = qtn.MPS_computational_state(bitstring, tags='PSI_f').squeeze()
# c_tn = circ.psi & psi_sample
# c_tn
circ.psi.graph(color=['PSI0'] + [f'ROUND_{i}' for i in range(10)])

circ.psi.select(['CNOT', 'I3', 'ROUND_3'], which='all')
"  ".join(qtn.circuit.GATE_FUNCTIONS)
list(qtn.circuit.GATE_FUNCTIONS)

circ.amplitude('1101010101')
circ.local_expectation(qu.pauli('Z'), (4)) + \
    circ.local_expectation(qu.pauli('Z'), (5))
circ.local_expectation(qu.pauli('Z') & qu.pauli('Z'), (4, 5))
circ.local_expectation(
    [qu.pauli('X') & qu.pauli('X'),
     qu.pauli('Y') & qu.pauli('Y'),
     qu.pauli('Z') & qu.pauli('Z')],
    where=(4, 5),
)
circ.local_expectation(qu.pauli('Z'), 0) + \
    circ.local_expectation(qu.pauli('Z'), 1)
