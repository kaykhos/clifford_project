from pylab import cm
import matplotlib as mpl
from tqdm import tqdm
from quimb.tensor.optimize import TNOptimizer
from numpy.random import randint
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


# The following lists record the gate selected, and the new gate applied in ramdom_apply_gate
applied_gates = []
gate_choice = []


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


def random_string():  # used to randomly select a Pauli string
    global string
    pauli = ['x', 'y', 'z', 'i']
    string = []
    for i in range(30):
        random_string = random.choice(pauli)
        string.append(random_string)


def random_ham():  # randomly generate a Paili string Hamiltonian for n=4, 16x16 matrix

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


def infidility(psi, target):
    a = 1 - abs((psi.H & target).contract(all, optimize='auto-hq'))

    return a


def ising(n, jz=1.0, h=0.0, **ham_opts):  # generate Ising Hamiltonian with X and Z fields
    Z = np.zeros((2**(n), 2**(n)), dtype=complex)
    return qu.ham_heis(n, j=(0, 0, jz), b=(h, 0, h), cyclic=False, **ham_opts)


# %%
n = 8  # the number of qubits
d = 1  # the circuit depth
beta = 31  # beta in Metropolis algorithm
Ham = ising(n, jz=1.0, h=0.5 * 1)  # the Hamiltonian used
# Ham = random_ham()
for t in range(1):
    data = {}
    evals, evecs = la.eig(Ham)
    min_eval = (evals).min()  # calculate groundstate eigenvalue
    evec_min = qu.groundstate(Ham)  # calculate groundstate of Hamiltonian
    for a in range(1):
        qc = qcnewcircuit(n, d)
        operations = op_list.copy()
        op_copy = op_list.copy()
        E1 = 5
        t0 = 0
        E3 = 1
        exec(f'E_{a} = np.array([])')
        exec(f't_{a} = np.array([])')
        exec(f'energy_{a} = np.array([])')
        for i in range(1000):
            qc_new = update_qc(n, depth=d)
            psi = qc_new.to_dense()
            E2 = 1 - qu.calc.fidelity(psi, evec_min)
            Energy = np.real(qu.core.expectation(psi, Ham))
            if E3 > E2:
                P = 1
            else:

                P = np.exp(beta * (E3 - E2))
            random_prob = random.random()
            if random_prob < P:  # accept the new value
                if E3 < 0.36:
                    exec(f'E_{a}=np.append(E_{a}, E3)')
                    exec(f't_{a}=np.append(t_{a}, i)')
                    exec(f'energy_{a}=np.append(energy_{a}, E1)')
                    operations = op_copy.copy()
                else:
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

            random_apply_gate(n, d)

        exec(f'data["E_{a}"] = E_{a}')
        exec(f'data["energy_{a}"] = energy_{a}')

data["t"] = t_0
df = pd.DataFrame(data)
# df_e = pd.DataFrame(data_E)
# cols_to_sum = df.columns[: df.shape[1] - 1]
# print(df)
# df['average'] = df[cols_to_sum].sum(axis=1) / len(df.columns)
# exec(f'df.to_csv("random_pauli_n4d4_good.csv", index=False)')
# exec(f'df_e.to_csv("random_pauli_n4d4_b22_E_{x}.csv", index=False)')
# %% used to plot the minimization process
print(np.real(min_eval))
infi = df['E_0']
ener = df['energy_0']
plt.figure(figsize=(9, 8), dpi=100)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Infidelity', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(t_0, infi, color='blue', label='d=3')
plt.legend()
plt.show()
# %% SPSA algorithm


class SPSAWraper(TNOptimizer):
    global infid_cliff, infid_random
    """
    A wrapper for TNOPtimizer that uses all of it's nice stuff to handle params,
    but runs spsa (not included in scipy) instead, see
    quimb.tensor.optimize.TNOptimizer for full definitions"""

    def __init__(self, **args):
        super().__init__(**args)
        if self._method == 'SPSA':
            self.optimize = self._make_spsa_optimizer()

    def _make_spsa_optimizer(self):
        """
        Creates the method to overwrite the TNOptimizer.optimize method
        Returns
        -------
        A callable (spsa) optimize function based on qiskits implementation
        """
        def optimize(maxiter: int = 1000,
                     save_steps: int = 1,
                     tol=None,
                     c0: float = 0.1,
                     c1: float = 0.1,
                     c2: float = 0.9,
                     c3: float = 0.9,
                     c4: float = 10.0):
            """
            This method is heavily based on qiskits optimizers.spsa method,
            adapted here to worth with on quibs tn's without exact gradients
            Parameters
            ----------
            maxiter: Maximum number of iterations to perform.
            save_steps: Save intermediate info every save_steps step. It has a min. value of 1.
            last_avg: Averaged parameters over the last_avg iterations.
                If last_avg = 1, only the last iteration is considered. It has a min. value of 1.
            c0: The initial a. Step size to update parameters.
            c1: The initial c. The step size used to approximate gradient.
            c2: The alpha in the paper, and it is used to adjust a (c0) at each iteration.
            c3: The gamma in the paper, and it is used to adjust c (c1) at each iteration.
            c4: The parameter used to control a as well.

            Returns
            -------
            TYPE : updated object? (same return as TNOptimize)
            """
            _spsa_vars = [c0, c1, c2, c3, c4]
            theta = self.vectorizer.vector
            nb_params = len(theta)
            use_exact_grads = 'grads' in self._method

            if save_steps:
                theta_vec = [theta]
                cost_vec = [self.vectorized_value_and_grad(theta)[0]]

            for ii in tqdm(range(maxiter)):

                a_spsa = float(_spsa_vars[0]) / \
                    ((ii + 1 + _spsa_vars[4])**_spsa_vars[2])
                c_spsa = float(_spsa_vars[1]) / ((ii + 1)**_spsa_vars[3])
                delta = 2 * randint(0, 2, size=nb_params) - 1
                # plus and minus directions

                if use_exact_grads:
                    raise NotImplementedError(
                        'Will use grad calc to project on to SP-direction')
                else:
                    theta_plus = theta + c_spsa * delta
                    theta_minus = theta - c_spsa * delta

                    cost_plus = self.vectorized_value_and_grad(theta_plus)[0]
                    cost_minus = self.vectorized_value_and_grad(theta_minus)[0]
                    # derivative estimate
                    g_spsa = (cost_plus - cost_minus) * delta / (2.0 * c_spsa)
                    # updated theta
                    theta = theta - a_spsa * g_spsa
                # if tol is not None:
                #     if (cost_plus + cost_minus)/2 < tol:
                #         break
                if save_steps:
                    theta_vec.append(theta)
                    cost_vec.append(cost_plus / 2 + cost_minus / 2)
                infid = 1 + self.vectorized_value_and_grad(theta)[0]

                # infid_cliff=[]
                # infid_random=[]
                # infid_cliff= np.append(infid_cliff, infid)
                print(infid, end='\r')
                # if infid < tol:
                #     break
            result_dict = {'hyper_parameters': _spsa_vars,
                           'maxiter': maxiter,
                           'theta_opt': theta,
                           'cost_opt': self.vectorized_value_and_grad(theta)[0],
                           'grad_opt': self.vectorized_value_and_grad(theta)[1]}
            if save_steps:
                result_dict['theta_history'] = theta_vec
                result_dict['cost_history'] = cost_vec
            self.result_dict = result_dict
            return self.inject_res_vector_and_return_tn()
        return optimize


# %% minimization algorithm, test on SPSA


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


# generate Ising Hamiltonian with X and Z fields
def ising_complex(n, jz=1.0, h=0.0, **ham_opts):
    Z = np.zeros((2**(n), 2**(n)), dtype=complex)
    return qu.ham_heis(n, j=(0, 0, jz), b=(h, 0, h), cyclic=False, **ham_opts) + Z


circ_with_U3 = my_circuit(n, d)

ran_circ = qtn.circuit_gen.circ_ansatz_1D_rand(n, d)
V = circ_with_U3
H_0 = ising(n, jz=1.0, h=0.5 * 1)
# H_0 = Ham
V1 = ran_circ


gs = qu.groundstate(H_0)
gs_complex = gs + np.zeros((2**(n), 1), dtype=complex)
target = qtn.Dense1D(gs_complex)


def negative_overlap(psi, target):
    a = - abs((psi.H & target).contract(all, optimize='auto-hq'))

    return a


def normalize_state(psi):
    return psi / (psi.H @ psi) ** 0.5


optimizer = SPSAWraper(tn=V.psi,
                       loss_fn=negative_overlap,
                       norm_fn=normalize_state,
                       loss_constants={'target': target},
                       tags=['U3'],
                       autodiff_backend='auto',
                       optimizer='SPSA')

V_opt = optimizer.optimize(200)

results = optimizer.result_dict
infid_cliff = [x + 1 for x in results['cost_history']]
print(infid_cliff)

circ_with_U3.update_params_from(V_opt)

np.real(qu.core.expectation(V.to_dense(), Ham))  # energy before
np.real(qu.core.expectation(V_opt.to_dense(), H_0))  # energy after
1 - qu.calc.fidelity(V.to_dense(), gs_complex)  # infid before
1 - qu.calc.fidelity(V_opt.to_dense(), gs)  # infid after
1 + negative_overlap(V.psi, target)

# %%test for 10 pre-optimize states and 10 random states
data1 = {}

for i in range(10):
    optimizer = SPSAWraper(tn=V.psi,
                           loss_fn=negative_overlap,
                           norm_fn=normalize_state,
                           loss_constants={'target': target},
                           tags=['U3'],
                           autodiff_backend='auto',
                           optimizer='SPSA')
    V_opt = optimizer.optimize(200)
    results = optimizer.result_dict
    infid_cliff = [x + 1 for x in results['cost_history']]
    exec(f'data1["infid_cliff_{i}"] = infid_cliff')
# results = optimizer.result_dict
# infid_cliff = [x + 1 for x in results['cost_history']]
# data1["infid_cliff_9"] = infid_cliff

for i in range(10):
    ran_circ = qtn.circuit_gen.circ_ansatz_1D_rand(n, d)
    V1 = ran_circ
    optimizer = SPSAWraper(tn=V1.psi,
                           loss_fn=negative_overlap,
                           norm_fn=normalize_state,
                           loss_constants={'target': target},
                           tags=['U3'],
                           autodiff_backend='auto',
                           optimizer='SPSA')

    V_opt = optimizer.optimize(200)
    results = optimizer.result_dict
    infid_random = [x + 1 for x in results['cost_history']]
    exec(f'data1["infid_random_{i}"] = infid_random')

df1 = pd.DataFrame(data1)
df1.to_csv("spsa_result_pauli_n6d2.csv", index=False)
# %% plot the # optimized circuit structure
# the basic gate specification
circ_with_U3.psi.graph(color=['H', 'CNOT', 'S', 'U3'])
circ_with_U3.gates
V_opt['U3', 'ROUND_3', 'I2']

# %%plot the SPSA results
# average = df['energy_0']
infi = df['E_0']
t = np.arange(0, 201, 1)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])

# plt.figure(figsize=(9, 8), dpi=100)
ax.set_xlabel('Number of Steps', fontsize=20)
ax.set_ylabel('Infidelity', fontsize=20)
colors = cm.get_cmap('tab10', 2)
ax.plot(t, infid_cliff, linewidth=2, color=colors(0), label='Clifford Circuit')
# ax.plot(t, infid_random, linewidth=2, color=colors(1), label='Random Input')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.xaxis.set_tick_params(which='major', size=10,
                         width=2, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=7,
                         width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10,
                         width=2, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=7, width=2,
                         direction='in', right='on')
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(20))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
plt.legend(fontsize=20)
plt.show()
# plt.savefig('cliff_vs_random_t.png', figsize=(12, 10),
#             dpi=300, transparent=False, bbox_inches='tight')
# %%plotting the Probability distribution for the number of runs reached minimum
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
