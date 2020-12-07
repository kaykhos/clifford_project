from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from quimb.tensor.tensor_gen import SpinHam
import quimb.tensor.tensor_gen as qt_gen
import quimb as qu
import quimb.tensor as qtn
import random
import numpy as np
import pandas as pd
import scipy.linalg as la
# %%full single qubit clifford gates
cliff_gates = []
for i in range(24):
    cliff_gates.append(i + 1)
gate1 = ['IDEN']
gate2 = ['H']
gate3 = ['S']
gate4 = ['X']
gate5 = ['Y']
gate6 = ['Z']
gate7 = ['H', 'S']
gate8 = ['S', 'H']
gate9 = ['S', 'S', 'S']
gate10 = ['S', 'H', 'S']
gate11 = ['S', 'S', 'H']
gate12 = ['H', 'S', 'H']
gate13 = ['H', 'S', 'S']
gate14 = ['H', 'S', 'S', 'S']
gate15 = ['H', 'S', 'H', 'S']
gate16 = ['S', 'H', 'S', 'S']
gate17 = ['S', 'S', 'H', 'S']
gate18 = ['S', 'H', 'S', 'S', 'H']
gate19 = ['S', 'S', 'H', 'S', 'S']
gate20 = ['H', 'S', 'S', 'H', 'S']
gate21 = ['S', 'S', 'H', 'S', 'H']
gate22 = ['S', 'H', 'S', 'S', 'S']
gate23 = ['H', 'S', 'H', 'S', 'S']
gate24 = ['S', 'H', 'S', 'S', 'H', 'S']

# %% randomly generate circuit, and update the circuit


def qcnewcircuit(nb_qubits, depth):
    """
    Creates a new random 'cliffod' circuit"""
    Warning("Currently only makes a reduced Clifford circuit")
    global op_list
    op_list = []
    # Construct circuit
    circuit = qtn.Circuit(N=nb_qubits)
    # Need to increase the gate set here... maybe this isn't the best way
    # Might need to use u3 params instead, but this will do for now
    # gate_list = ['H', 'X', 'Y',
    #              'Z', 'IDEN', 'S']

    def entangle_layer(circ):
        """
        Creates a linear entangeling layer"""
        for ii in range(0, nb_qubits - 1, 2):
            circ.apply_gate('CNOT', ii, ii + 1)
        for ii in range(1, nb_qubits - 1, 2):
            circ.apply_gate('CNOT', ii, ii + 1)

    def rotaiton_layer(circ):
        """
        Creates a layer of single qubit rotations based on the list 'single_rotatoins'"""

        # random_points = np.random.randint(
        #     0, len(gate_list), circ.num_qubits)
        for ii in range(nb_qubits):
            # for random_number in cliff_gates:
            random_number = random.choice(cliff_gates)
            exec(f'random_gate= gate{random_number}', globals(), globals())
            op_list.append(random_number)
            x = len(random_gate)
            for j in range(x):
                gate = random_gate[j]
                circ.apply_gate(gate, ii)

    # Apply first rotation layer (else CX layer does nothing)
    rotaiton_layer(circuit)

    # Loop though and alternate rotation and entangelment layers
    for ii in range(depth):
        entangle_layer(circuit)
        rotaiton_layer(circuit)
    return circuit


applied_gates = []
gate_choice = []
# gate_list = ['H', 'X', 'Y',
#              'Z', 'IDEN', 'S']


def random_apply_gate(n, d):  # randomly select gate here, gate is selected from gate_list
    def random_select_gate_from_list():
        gate_to_select = cliff_gates.copy()
        if len(applied_gates) > 0:
            gate_to_select.remove(gate_choice[0])
        # for random_number in cliff_gates:
        random_number = random.choice(cliff_gates)
        exec(f'random_select_gate= gate{random_number}')
        applied_gates.insert(0, random_number)
        gate_choice.insert(0, random_number)
    random_select_gate_from_list()
    max = n - 1 + n * d
    num = random.randint(0, max)
    operations.pop(num)
    operations.insert(num, applied_gates[0])


def update_qc(nb_qubits, depth=1):  # compute the circuit, with CNOT and single qubit gates
    global gate_number, gate_apply
    circuit = qtn.Circuit(N=nb_qubits)

    def entangle_layer(circ):
        """
        Creates a linear entangeling layer"""
        for ii in range(0, nb_qubits - 1, 2):
            circ.apply_gate('CNOT', ii, ii + 1)
        for ii in range(1, nb_qubits - 1, 2):
            circ.apply_gate('CNOT', ii, ii + 1)

    def apply_rotation_gates(n, d, qubit):
        gate_number = operations[n * d + qubit]
        e = '{}'.format(gate_number)
        q = 'gate_apply=gate' + e + '.copy()'
        exec(q, globals(), globals())
        b = len(gate_apply)
        for l in range(0, b, 1):
            x = gate_apply[l]
            circuit.apply_gate(x, qubit)

    def last_rotation_layer(n, qubit):

        gate_number = operations[n]

        e = '{}'.format(gate_number)
        q = 'gate_last=gate' + e + '.copy()'
        exec(q, globals(), globals())
        c = len(gate_last)
        for l in range(0, c, 1):
            x = gate_last[l]
            circuit.apply_gate(x, qubit)

    for d in range(depth):
        for ii in range(nb_qubits):
            # gate_number= operations[nb_qubits * d + i]
            # exec(f'gate_to_apply= gate{gate_number}')
            # b=len(gate_to_apply)
            # for l in range(0,b,1):
            #     x=gate_to_apply[l]
            #     circuit.apply_gate(x, i)
            apply_rotation_gates(nb_qubits, d, ii)
            entangle_layer(circuit)
    for j in range(nb_qubits * depth, nb_qubits + nb_qubits * depth):
        for i in range(nb_qubits):
            last_rotation_layer(j, i)
            # gate_number= operations[j]
            # exec(f'gate_to_apply_last= gate{gate_number}')
            # c=len(gate_to_apply_last)
            # for l in range(0,c,1):
            #     x=gate_to_apply_last[j]
            #     circuit.apply_gate(x, iii)
            # x = operations[j]
            # circuit.apply_gate(x, iii)
    return circuit

# %% create Hamiltonian and randomly generate one with pauli strings


def random_string():
    global string
    pauli = ['x', 'y', 'z', 'i']
    string = []
    for i in range(16):
        random_string = random.choice(pauli)
        string.append(random_string)


def random_ham():

    def pauli(xyz, dim=2):
        return qu.gen.operators.pauli(xyz, dim)
    random_string()
    a = np.kron(pauli(string[0]), pauli(string[1]))
    b = np.kron(pauli(string[2]), pauli(string[3]))
    c = np.kron(pauli(string[4]), pauli(string[5]))
    d = np.kron(pauli(string[6]), pauli(string[7]))
    e = np.kron(pauli(string[8]), pauli(string[9]))
    f = np.kron(pauli(string[10]), pauli(string[11]))
    g = np.kron(pauli(string[12]), pauli(string[13]))
    h = np.kron(pauli(string[14]), pauli(string[15]))
    ham = np.kron(a, b) + np.kron(c, d) + np.kron(e, f) + np.kron(g, h)

    return ham


# %%find eigenvalues of Hamiltonian
test_ham = random_ham()
# print(test_ham)
# S = Ham @ Ham.T

evals, evecs = la.eig(test_ham)
# print(4 * evals.real)
min_eval = (evals.real).min()
print(min_eval)
# print(evecs)

# %%
min_eigenvalue = []
min_circuit = []
for t in range(3):
    data = {}
    Ham = random_ham()
    evals, evecs = la.eig(Ham)
    min_eval = (evals.real).min()

    for a in range(100):
        qc = qcnewcircuit(4, 4)
        operations = op_list.copy()
        op_copy = op_list.copy()
        E1 = 5
        t0 = 0
        exec(f'E_{a} = np.array([])')
        exec(f't_{a} = np.array([])')
        for i in range(1000):
            qc_new = update_qc(4, depth=4)
            psi = qc_new.to_dense()
            # Ham = random_ham()
            E2 = np.real(qu.core.expectation(psi, Ham))
            if E1 > E2:
                P = 1
            else:
                beta = 10
                P = np.exp(beta * (E1 - E2))
            random_prob = random.random()
            if random_prob < P:  # accept the new value
                exec(f'E_{a}=np.append(E_{a}, E2)')
                exec(f't_{a}=np.append(t_{a}, i)')
                E1 = E2
                op_copy = operations.copy()
            else:  # not accept the new value
                exec(f'E_{a}=np.append(E_{a}, E1)')
                exec(f't_{a}=np.append(t_{a}, i)')
                operations = op_copy.copy()

            random_apply_gate(4, 4)

        exec(f'data["E_{a}"] = E_{a}')

    # print(min_eval)

    data["t"] = t_0
    df = pd.DataFrame(data)
    cols_to_sum = df.columns[: df.shape[1] - 1]
    df['average'] = df[cols_to_sum].sum(axis=1) / len(df.columns)
    min_value = df.min()
    min_data = min_value.min()
    min_circuit.append(min_data)
    min_eigenvalue.append(min_eval)
    exec(f'df.to_csv("random_pauli_n4d4_3{t}.csv", index=False)')
print(min_eigenvalue)
print(min_circuit)
# %%plotting
min_values = df.min()
print(min_values.min())
t_star = []
for i in range(100):
    exec(f'colm= df["E_{i}"]')
    exec(f'E{i}=df[colm < -0.99].index.tolist()')
    exec(f'a = len(E{i})')
    if a == 0:
        a = 0
    else:
        exec(f't_star.append(E{i}[0])')

s_400 = len([i for i in t_star if i < 400])
P = s_400 / len(t_star)
plt.grid(axis='y', alpha=0.75)
plt.xlabel(r'$t^*$')
plt.ylabel('Probability')
plt.title(r'Probability distribution for h=0')
plt.text(400, 0.2, f'P($t^* < 400$)={P}')
plt.text(400, 0.25, f'Total number in 100 runs={len(t_star)}')
plt.hist(t_star, weights=np.ones(len(t_star)) / len(t_star), bins=10,
         color='#0504aa', alpha=0.9, rwidth=0.5, align='right')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()
