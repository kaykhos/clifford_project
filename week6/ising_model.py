from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from quimb.tensor.tensor_gen import SpinHam
import quimb.tensor.tensor_gen as qt_gen
import quimb as qu
import quimb.tensor as qtn
import random
import numpy as np
import pandas as pd
import sys
import scipy.linalg as la
# %% randomly generate circuit


def qcnewcircuit(nb_qubits, depth):
    """
    Creates a new random 'cliffod' circuit"""
    Warning("Currently only makes a reduced Clifford circuit")
    global op_list, applied_qubits, entangle_layer
    op_list = []
    applied_qubits = []
    # Construct circuit
    circuit = qtn.Circuit(N=nb_qubits)
    # Need to increase the gate set here... maybe this isn't the best way
    # Might need to use u3 params instead, but this will do for now
    gate_list = ['H', 'X', 'Y',
                 'Z', 'IDEN', 'S']

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
            random_gate = random.choice(gate_list)
            circ.apply_gate(random_gate, ii)
            op_list.append(random_gate)
    # Apply first rotation layer (else CX layer does nothing)
    rotaiton_layer(circuit)

    # Loop though and alternate rotation and entangelment layers
    for ii in range(depth):
        entangle_layer(circuit)
        rotaiton_layer(circuit)
    return circuit


def ising(n, jz=1.0, h=0.0, **ham_opts):  # generate Ising Hamiltonian with X and Z fields
    return qu.ham_heis(n, j=(0, 0, jz), b=(h, 0, h), cyclic=False, **ham_opts)


applied_gates = []
gate_choice = []
gate_list = ['H', 'X', 'Y',
             'Z', 'IDEN', 'S']


def random_apply_gate(n, d):  # randomly select gate here, gate is selected from gate_list
    def random_select_gate_from_list():
        gate_to_select = gate_list.copy()
        if len(applied_gates) > 0:
            gate_to_select.remove(gate_choice[0])
        random_gate = random.choice(gate_to_select)
        applied_gates.insert(0, random_gate)
        gate_choice.insert(0, random_gate)
    random_select_gate_from_list()
    max = n - 1 + n * d
    num = random.randint(0, max)
    operations.pop(num)
    operations.insert(num, applied_gates[0])


def update_qc(nb_qubits, depth=1):  # compute the circuit, with CNOT and single qubit gates

    circuit = qtn.Circuit(N=nb_qubits)

    def entangle_layer(circ):
        """
        Creates a linear entangeling layer"""
        for ii in range(0, nb_qubits - 1, 2):
            circ.apply_gate('CNOT', ii, ii + 1)
        for ii in range(1, nb_qubits - 1, 2):
            circ.apply_gate('CNOT', ii, ii + 1)
    for d in range(depth):
        for i in range(nb_qubits):
            x = operations[nb_qubits * d + i]
            circuit.apply_gate(x, i)
            entangle_layer(circuit)
    for j in range(nb_qubits * depth, nb_qubits + nb_qubits * depth):
        for iii in range(nb_qubits):
            x = operations[j]
            circuit.apply_gate(x, iii)
    return circuit


# %% Ising model Hamiltonian
data = {}

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
        Ham = ising(4, jz=1.0, h=0.5 * 1)
        E2 = np.real(qu.core.expectation(psi, Ham))
        if E1 > E2:
            P = 1
        else:
            beta = 5
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
        random_apply_gate(4, 4)
        random_apply_gate(4, 4)
        random_apply_gate(4, 4)
        random_apply_gate(4, 4)
    exec(f'data["E_{a}"] = E_{a}')

# print(Energy)
# len(t_0)

data["t"] = t_0
df = pd.DataFrame(data)
cols_to_sum = df.columns[: df.shape[1] - 1]
df['average'] = df[cols_to_sum].sum(axis=1) / len(df.columns)
df.to_csv("ising_n4d4_h1_beta5.csv", index=False)

# %%
average = df['average']
t = df['t']
plt.figure(figsize=(18, 16), dpi=100)
plt.xlabel('time', fontsize=20)
plt.ylabel('E', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(t, average, color='skyblue', label='depth=1')

# %%
print(Ham)
# S = Ham @ Ham.T

evals, evecs = la.eig(Ham)
print(4 * evals.real)
min_eval = (4 * evals.real).min()
print(min_eval)
print(evecs)
# %%
min_values = df.min()
print(min_values.min())
t_star = []
for i in range(100):
    exec(f'colm= df["E_{i}"]')
    exec(f'E{i}=df[colm < -0.999].index.tolist()')
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
# plt.show()
# plt.savefig(fname="100run_h=0.png", figsize=(18, 16), dpi=400)
