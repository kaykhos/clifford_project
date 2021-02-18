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
%config InlineBackend.figure_formats = ['svg']
# %%full single qubit clifford gates
cliff_gates = []
for i in range(24):
    cliff_gates.append(i + 1)
gate1 = ['Z', 'Z']
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
    for i in range(20):
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
    i = np.kron(pauli(string[16]), pauli(string[17]))
    j = np.kron(pauli(string[18]), pauli(string[19]))
    ham = np.kron(a, b) + np.kron(c, d) + np.kron(e, f) + \
        np.kron(g, h) + np.kron(i, j)

    return ham


def overlap_modules_square(psi_1, psi_2):

    # return (psi_1.adjoint().compose(psi_2).eval().real) * (psi_2.adjoint().compose(psi_1).eval().real)
    return np.real((np.conjugate(psi_1) @ psi_2) * (np.conjugate(psi_2) @ psi_1))


# %%find eigenvalues of Hamiltonian
test_ham = random_ham()
# print(test_ham)
# S = Ham @ Ham.T

evals, evecs = la.eig(test_ham)
min_eval = (evals).min()
index = np.where(evals == min_eval)[0]
evec_min = evecs[:, index[0]]
qc1 = qcnewcircuit(4, 4)
psi1 = qc1.to_dense()
infide = 1 - overlap_modules_square(psi1, evec_min)
infide
print(evec_min)
1 - qu.calc.fidelity(psi1, evec_min)

# %%
# min_eigenvalue = []
# min_circuit = []
for t in range(1):
    data = {}
    # data_E = {}
    Ham = random_ham()
    evals, evecs = la.eig(Ham)
    min_eval = (evals).min()
    index = np.where(evals == min_eval)[0]
    evec_min = evecs[:, index[0]]
    for a in range(1):
        qc = qcnewcircuit(4, 4)
        operations = op_list.copy()
        op_copy = op_list.copy()
        E1 = 5
        t0 = 0
        E3 = 1
        exec(f'E_{a} = np.array([])')
        exec(f't_{a} = np.array([])')
        exec(f'energy_{a} = np.array([])')
        for i in range(1000):
            qc_new = update_qc(4, depth=4)
            psi = qc_new.to_dense()
            # Ham = random_ham()
            E2 = 1 - qu.calc.fidelity(psi, evec_min)
            Energy = np.real(qu.core.expectation(psi, Ham))
            if E3 > E2:
                P = 1
            else:
                beta = 25
                P = np.exp(beta * (E3 - E2))
            random_prob = random.random()
            if random_prob < P:  # accept the new value
                E3 = E2
                exec(f'E_{a}=np.append(E_{a}, E2)')
                exec(f't_{a}=np.append(t_{a}, i)')
                exec(f'energy_{a}=np.append(energy_{a}, Energy)')
                E1 = Energy
                op_copy = operations.copy()
            else:  # not accept the new value
                exec(f'E_{a}=np.append(E_{a}, E3)')
                exec(f't_{a}=np.append(t_{a}, i)')
                exec(f'energy_{a}=np.append(energy_{a}, E1)')
                operations = op_copy.copy()

            random_apply_gate(4, 4)

        exec(f'data["E_{a}"] = E_{a}')
        exec(f'data["energy_{a}"] = energy_{a}')

    # print(min_eval)

    data["t"] = t_0
    df = pd.DataFrame(data)
    # df_e = pd.DataFrame(data_E)
    cols_to_sum = df.columns[: df.shape[1] - 1]
    df['average'] = df[cols_to_sum].sum(axis=1) / len(df.columns)
    x = t
    exec(f'df.to_csv("random_pauli_n4d4_b5_infid{x}.csv", index=False)')
    # exec(f'df_e.to_csv("random_pauli_n4d4_b22_E_{x}.csv", index=False)')
print(np.real(min_eval))
# %%
print(Energy)
print(E3)
# average = df['energy_0']
infid = df['E_0']
plt.figure(figsize=(18, 16), dpi=100)
plt.xlabel('time', fontsize=20)
plt.ylabel('E', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(t_0, infid, color='skyblue', label='depth=1')
# plt.plot(t2, average2, color='blue', label='depth=2')
# plt.plot(t3, average3, color='red', label='depth=3')
# plt.plot(t4, average4, color='green', label='depth4')
# plt.plot(t5, average5, color='orange', label='depth=5')
plt.legend()
plt.show()
# %% minimization algorithm
circ_random = qcnewcircuit(4, 4)  # apply randome input state
operations_random = op_list.copy()  # apply the ramdom circuit

operations_mycircuit = op_copy.copy()  # apply the optimized circuit


# compute the circuit, with CNOT and single qubit gates, adding U 3 gates for optimization
def my_circuit(nb_qubits, depth=1):
    global gate_numbers, gates_apply
    circuit = qtn.Circuit(N=nb_qubits)

    def entangle_layer(circ, gate_round=None):
        """
        Creates a linear entangeling layer"""
        for ii in range(0, nb_qubits - 1, 2):
            circ.apply_gate('CNOT', ii, ii + 1, gate_round=gate_round)
        for ii in range(1, nb_qubits - 1, 2):
            circ.apply_gate('CNOT', ii, ii + 1, gate_round=gate_round)

    def apply_rotation_gates(n, d, qubit, gate_round=None):
        gate_numbers = operations_mycircuit[n * d + qubit]
        e = '{}'.format(gate_numbers)
        q = 'gates_apply=gate' + e + '.copy()'
        exec(q, globals(), globals())
        b = len(gates_apply)
        for l in range(0, b, 1):
            x = gates_apply[l]
            circuit.apply_gate(x, qubit, gate_round=gate_round)

    def last_rotation_layer(n, qubit, gate_round=None):

        gate_numbers = operations_mycircuit[n]

        e = '{}'.format(gate_numbers)
        q = 'gate_last=gate' + e + '.copy()'
        exec(q, globals(), globals())
        c = len(gate_last)
        for l in range(0, c, 1):
            x = gate_last[l]
            circuit.apply_gate(x, qubit, gate_round=gate_round)

    def U3_gates(qubit, gate_round=None):
        circuit.apply_gate('U3', 0, 0, 0, qubit, gate_round=gate_round)

    for d in range(depth):
        for ii in range(nb_qubits):
            apply_rotation_gates(nb_qubits, d, ii, gate_round=d)
            U3_gates(ii, gate_round=d)
        entangle_layer(circuit, gate_round=d)
    for j in range(nb_qubits * depth, nb_qubits + nb_qubits * depth):
        for i in range(nb_qubits):
            last_rotation_layer(j, i, gate_round=depth)
            U3_gates(i, gate_round=depth)

    return circuit


def random_circuit(nb_qubits, depth=1):
    # global gate_numbers, gates_apply
    circuit = qtn.Circuit(N=nb_qubits)

    def entangle_layer(circ, gate_round=None):
        """
        Creates a linear entangeling layer"""
        for ii in range(0, nb_qubits - 1, 2):
            circ.apply_gate('CNOT', ii, ii + 1, gate_round=gate_round)
        for ii in range(1, nb_qubits - 1, 2):
            circ.apply_gate('CNOT', ii, ii + 1, gate_round=gate_round)

    def apply_rotation_gates(n, d, qubit, gate_round=None):
        gate_numbers = operations_random[n * d + qubit]
        e = '{}'.format(gate_numbers)
        q = 'gates_apply=gate' + e + '.copy()'
        exec(q, globals(), globals())
        b = len(gates_apply)
        for l in range(0, b, 1):
            x = gates_apply[l]
            circuit.apply_gate(x, qubit, gate_round=gate_round)

    def last_rotation_layer(n, qubit, gate_round=None):

        gate_numbers = operations_random[n]

        e = '{}'.format(gate_numbers)
        q = 'gate_last=gate' + e + '.copy()'
        exec(q, globals(), globals())
        c = len(gate_last)
        for l in range(0, c, 1):
            x = gate_last[l]
            circuit.apply_gate(x, qubit, gate_round=gate_round)

    def U3_gates(qubit, gate_round=None):
        circuit.apply_gate('U3', 0, 0, 0, qubit, gate_round=gate_round)

    for d in range(depth):
        for ii in range(nb_qubits):
            apply_rotation_gates(nb_qubits, d, ii, gate_round=d)
            U3_gates(ii, gate_round=d)
        entangle_layer(circuit, gate_round=d)
    for j in range(nb_qubits * depth, nb_qubits + nb_qubits * depth):
        for i in range(nb_qubits):
            last_rotation_layer(j, i, gate_round=depth)
            U3_gates(i, gate_round=depth)

    return circuit


circ_with_U3 = my_circuit(4, 4)

ran_circ = qtn.circuit_gen.circ_ansatz_1D_rand(4, 4)
# circ_with_U3.psi.graph(color=['PSI0'] + [f'ROUND_{i}' for i in range(5)])
V = circ_with_U3
H_0 = Ham

V1 = ran_circ


gs = qu.groundstate(H_0)
target = qtn.Dense1D(gs)


def negative_overlap(psi, target):
    a = - abs((psi.H & target).contract(all, optimize='auto-hq'))

    return a


V.psi.graph(color=['H', 'CNOT', 'S', 'U3'])
(V.psi.H & target).graph()
target.graph()

print(gs)
print(evec_min)
np.real(qu.core.expectation(V.to_dense(), Ham))
-qu.calc.fidelity(V.to_dense(), gs)
negative_overlap(V.psi, target)

# def loss(V, U):
#     return 1 - abs((V.H & U).contract(all, optimize='auto-hq')) / 2**n
#     # return 1-qu.calc.fidelity(V,U)


def normalize_state(psi):
    return psi / (psi.H @ psi) ** 0.5

# def negative_overlap(V, target):
#     return - np.real(qu.core.expectation(V.to_dense(), target.to_dense()))


tnopt = qtn.optimize.TNOptimizer(
    V1.psi,                        # the tensor network we want to optimize
    loss_fn=negative_overlap,
    norm_fn=normalize_state,                    # the function we want to minimize
    # supply U to the loss function as a constant TN
    loss_constants={'target': target},
    tags=['U3'],              # only optimize U3 tensors
    autodiff_backend='auto',   # use 'autograd' for non-compiled optimization
    optimizer='L-BFGS-B',     # the optimization algorithm
)

V_opt = tnopt.optimize(n=500)
print(V_opt)
circ_with_U3.update_params_from(V_opt)

np.real(qu.core.expectation(psi, Ham))  # energy before
np.real(qu.core.expectation(V_opt.to_dense(), Ham))  # energy after
1 - qu.calc.fidelity(psi, evec_min)  # infid before
1 - qu.calc.fidelity(V_opt.to_dense(), gs)  # infid after
negative_overlap(psi, target)
1 + negative_overlap(V_opt, target)

# the basic gate specification
circ_with_U3.psi.graph(color=['H', 'CNOT', 'S', 'U3'])
circ_with_U3.gates

V_opt['U3', 'ROUND_3', 'I2']
# %%


# %%plotting
df = df.drop("t", 1)

min_values = df.min()
print(min_values.min())
t_star = []
for i in range(100):
    exec(f'colm= df["E_{i}"]')
    exec(f'E{i}=df[colm < 0.24].index.tolist()')
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
plt.hist(t_star, weights=np.ones(len(t_star)) / len(t_star), bins=8,
         color='#0504aa', alpha=0.9, rwidth=0.5, align='right')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()
