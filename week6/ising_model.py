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
import circuit_functions as cf
sys.path.append("/Users/arthur/Desktop/physics/MSci/clifford_project/week4")
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


def ising(n, jz=1.0, h=0.0, **ham_opts):
    return qu.ham_heis(n, j=(0, 0, jz), b=(h, 0, h), **ham_opts)


applied_gates = []
gate_choice = []
gate_list = ['H', 'X', 'Y',
             'Z', 'IDEN', 'S']


def random_apply_gate(n, d):
    def random_select_gate_from_list():
        gate_to_select = gate_list.copy()
        if len(applied_gates) > 0:
            gate_to_select.remove(gate_choice[0])
        random_gate = random.choice(gate_to_select)
        applied_gates.insert(0, random_gate)
        gate_choice.insert(0, random_gate)
    random_select_gate_from_list()
    max = n - 1 + 4 * d
    num = random.randint(0, max)
    operations.pop(num)
    operations.insert(num, applied_gates[0])


def update_qc(nb_qubits, depth=1):

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
# def _ham_ising(j=1.0, h=0.0, *, S=1 / 2, cyclic=False):
#     H = qt_gen.SpinHam(S=1 / 2, cyclic=cyclic)
#     H += j, 'Z', 'Z'
#     H -= h, 'X'
#     H -= h, 'Z'
#     return H
#
#
#
# def ham_1d_ising(n, j=1.0, h=0.0, *, S=1 / 2,cyclic=False, **mpo_opts):
#     r"""Ising Hamiltonian in nni form.
#
#     .. math::
#
#         H_\mathrm{Ising} =
#         J \sum_{i} \sigma^Z_i \sigma^Z_{i + 1} -
#         h \sum_{i} \sigma^X_i
#
#     Note the default convention of antiferromagnetic interactions and spin
#     operators not Pauli matrices.
#
#     Parameters
#     ----------
#     L : int
#         The number of sites.
#     j : float, optional
#         The ZZ interaction strength. Positive is antiferromagnetic.
#     bx : float, optional
#         The X-magnetic field strength.
#     S : {1/2, 1, 3/2, ...}, optional
#         The underlying spin of the system, defaults to 1/2.
#     cyclic : bool, optional
#         Generate a hamiltonian with periodic boundary conditions or not,
#         default is open boundary conditions.
#     mpo_opts or local_ham_1d_opts
#         Supplied to :class:`~quimb.tensor.tensor_1d.LocalHam1D`.
#
#     Returns
#     -------
#     NNI
#     """
#     H = _ham_ising(j=j, h=h, S=S, cyclic=cyclic)
#     return H.build_mpo(n, **mpo_opts)
for a in range(100):
    qc = qcnewcircuit(8, 3)
    operations = op_list.copy()
    E1 = 2
    t0 = 0
    exec(f'E_{a} = np.array([])')
    exec(f't_{a} = np.array([])')
    for i in range(100):
        qc_new = update_qc(8, depth=3)
        psi = qc_new.to_dense()
        Ham = ising(8, jz=1.0, h=-0.0)
        E2 = np.real(qu.core.expectation(psi, Ham))
        if abs(E1) > abs(E2):
            P = 1
        else:
            beta = 10
            P = np.exp(beta * (E1 - E2))
        random_prob = random.random()
        if random_prob < P:  # accept the new value
            exec(f'E_{a}=np.append(E_{a}, E2)')
            exec(f't_{a}=np.append(t_{a}, i)')
            E1 = E2

        else:  # not accept the new value
            exec(f'E_{a}=np.append(E_{a}, E1)')
            exec(f't_{a}=np.append(t_{a}, i)')

        random_apply_gate(8, 3)

    exec(f'data["E_{a}"] = E_{a}')

# print(Energy)
# len(t_0)

data["t"] = t_0
df = pd.DataFrame(data)
cols_to_sum = df.columns[: df.shape[1] - 1]
df['average'] = df[cols_to_sum].sum(axis=1) / len(df.columns)
df.to_csv("ising_h0.csv", index=False)


# %%
t_star = []
for i in range(100):
    exec(f'colm= df["E_{i}"]')
    exec(f'E{i}=df[colm == 0].index.tolist()')
    exec(f'a = len(E{i})')
    if a == 0:
        a = 0
    else:
        exec(f't_star.append(E{i}[0])')

s_30 = len([i for i in t_star if i < 30])
P = s_30 / len(t_star)
plt.grid(axis='y', alpha=0.75)
plt.xlabel(r'$t^*$')
plt.ylabel('Probability')
plt.title(r'Probability distribution for h=0')
plt.text(30, 0.3, f'P($t^* < 30$)={P}')
plt.hist(t_star, weights=np.ones(len(t_star)) / len(t_star), bins=10,
         color='#0504aa', alpha=0.9, rwidth=0.5, align='right')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()
plt.savefig(fname="100run_h=0.png", figsize=(18, 16), dpi=400)
