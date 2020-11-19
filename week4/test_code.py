# qiskit                0.23.0
# qiskit-aer            0.7.0
# qiskit-aqua           0.8.0
# qiskit-ibmq-provider  0.11.0
# qiskit-ignis          0.5.0
# qiskit-terra          0.16.0
import matplotlib.pyplot as plt
from qiskit.aqua.operators import StateFn
from qiskit import QuantumCircuit
from docplex.mp.model import Model
import random
import qiskit
import numpy as np
from qiskit import Aer
from qiskit.quantum_info import Clifford
import pandas as pd
import sys
import circuit_functions as cf
sys.path.append("/Users/arthur/Desktop/physics/MSci/clifford_project/week4")
qiskit.__qiskit_version__
# %%


def qcnewcircuit(nb_qubits, depth):
    """
    Creates a new random 'cliffod' circuit"""
    Warning("Currently only makes a reduced Clifford circuit")
    global op_list, applied_qubits
    op_list = []
    applied_qubits = []
    # Construct circuit
    circuit = QuantumCircuit(nb_qubits)
    # Need to increase the gate set here... maybe this isn't the best way
    # Might need to use u3 params instead, but this will do for now
    gate_list = ['qc_new.id', 'qc_new.x', 'qc_new.y',
                 'qc_new.z', 'qc_new.h', 'qc_new.s']

    def entangle_layer(circ):
        """
        Creates a linear entangeling layer"""
        for ii in range(0, circ.num_qubits - 1, 2):
            circ.cx(ii, ii + 1)
        for ii in range(1, circ.num_qubits - 1, 2):
            circ.cx(ii, ii + 1)

    def rotaiton_layer(qc_new):
        """
        Creates a layer of single qubit rotations based on the list 'single_rotatoins'"""

        # random_points = np.random.randint(
        #     0, len(gate_list), circ.num_qubits)
        for ii in range(nb_qubits):
            random_gate = random.choice(gate_list)
            a = '({})'.format(ii)
            k = random_gate + a
            op_list.append(k)
            # applied_qubits.append(ii)
            exec(k)

    # Apply first rotation layer (else CX layer does nothing)
    rotaiton_layer(circuit)

    # Loop though and alternate rotation and entangelment layers
    for ii in range(depth):
        entangle_layer(circuit)
        circuit.barrier()
        rotaiton_layer(circuit)
    return circuit


qc = qcnewcircuit(4, 5)
cliff_sf_target = StateFn(qc)
qc_initial = QuantumCircuit(4)
# operations_i = ['qc_new.x(0)', 'qc_new.y(1)', 'qc_new.s(2)','qc_new.x(3)','qc_new.z(0)', 'qc_new.z(1)', 'qc_new.id(2)','qc_new.id(3)']
print(qc)
print(op_list)

# %%


def overlap_modules_square(psi_2):
    psi_1 = cliff_sf_target
    # return (psi_1.adjoint().compose(psi_2).eval().real) * (psi_2.adjoint().compose(psi_1).eval().real)
    return np.real((~psi_1 @ psi_2).eval() * (~psi_2 @ psi_1).eval())


# radom apply GATE_FUNCTIONS
applied_gates = []
gate_choice = []
gate_list = ['qc_new.id', 'qc_new.x', 'qc_new.y',
             'qc_new.z', 'qc_new.h', 'qc_new.s']


def random_apply_gate(d):
    max = 3 + 4 * d
    num = random.randint(0, max)  # randomly select gate here
    list_qubit0 = [0, 4, 8, 12, 16, 20]
    list_qubit1 = [1, 5, 9, 13, 17, 21]
    list_qubit2 = [2, 6, 10, 14, 18, 22]
    list_qubit3 = [3, 7, 11, 15, 19, 23]
    if num in list_qubit0:
        random_select_gate(0, num)
    elif num in list_qubit1:
        random_select_gate(1, num)
    elif num in list_qubit2:
        random_select_gate(2, num)
    elif num in list_qubit3:
        random_select_gate(3, num)


def random_select_gate(n, gate):
    gate_to_select = gate_list.copy()
    if len(applied_gates) > 0:
        gate_to_select.remove(gate_choice[0])
    random_gate = random.choice(gate_to_select)
    a = '({})'.format(n)
    k = random_gate + a
    applied_gates.insert(0, k)
    gate_choice.insert(0, random_gate)
    operations.pop(gate)
    operations.insert(gate, applied_gates[0])


def update_qc(depth):  # n is the nth gate

    # operations.pop(n)
    # operations.insert(n, applied_gates[0])
    for op in range(0, 4, 1):
        exec(operations[op])
    if depth >= 1:
        qc_new.cx(0, 1)
        qc_new.cx(2, 3)
        qc_new.cx(1, 2)
        qc_new.barrier()
        for op in range(4, 8, 1):
            exec(operations[op])
    if depth >= 2:
        qc_new.cx(0, 1)
        qc_new.cx(2, 3)
        qc_new.cx(1, 2)
        qc_new.barrier()
        for op in range(8, 12, 1):
            exec(operations[op])
    if depth >= 3:
        qc_new.cx(0, 1)
        qc_new.cx(2, 3)
        qc_new.cx(1, 2)
        qc_new.barrier()
        for op in range(12, 16, 1):
            exec(operations[op])
    if depth >= 4:
        qc_new.cx(0, 1)
        qc_new.cx(2, 3)
        qc_new.cx(1, 2)
        qc_new.barrier()
        for op in range(16, 20, 1):
            exec(operations[op])
    if depth >= 5:
        qc_new.cx(0, 1)
        qc_new.cx(2, 3)
        qc_new.cx(1, 2)
        qc_new.barrier()
        for op in range(20, 24, 1):
            exec(operations[op])


# def run_operation(nb_q, d):
data = {}
for b in range(1000):  # run the next for loop for n times, and reset parameters
    exec(f'E_{b} = np.array([1])')
    exec(f't_{b} = np.array([0])')
    qc_new = qc_initial.copy('qc_new')
    operations = op_list.copy()
    E1 = 1
    for j in range(2000):
        if E1 > 0.2:
            random_apply_gate(5)
            random_apply_gate(5)
            random_apply_gate(5)
            random_apply_gate(5)
            random_apply_gate(5)
            update_qc(5)
        else:
            random_apply_gate(5)
            update_qc(5)
        cliff_sf_output = StateFn(qc_new)
        # E1 = 1
        E2 = 1 - overlap_modules_square(cliff_sf_output)
        # compare between old(E1) and new(E2), and choose whether to accept the change
        if E1 > E2:
            P = 1
        else:  # E1<E2,accept with a probability
            beta = 20
            P = np.exp(beta * (E1 - E2))
        random_prob = random.random()
        if random_prob < P:  # accept the new value
            exec(f'E_{b}=np.append(E_{b}, E2)')
            exec(f't_{b}=np.append(t_{b}, j + 1)')
            E1 = E2
            qc_result = qc_new.copy('qc_result')

        else:  # not accept the new value
            exec(f'E_{b}=np.append(E_{b}, E1)')
            exec(f't_{b}=np.append(t_{b}, j + 1)')

        exec(f'data["E_{b}"] = E_{b}')
        qc_new = qc_initial.copy('qc_new_copy')  # initialize the circuit
        applied_gates = []
data["t"] = t_0
df = pd.DataFrame(data)
cols_to_sum = df.columns[: df.shape[1] - 1]
df['average'] = df[cols_to_sum].sum(axis=1) / 100
df.to_csv("my_circuit_5_beta20_1krun.csv", index=False)


# %%
operations = op_list.copy()
qc_new = qc_initial.copy('qc_new_copy')
random_apply_gate(0)
update_qc(0, 5)
print(qc_new)

run_operation(4, 5)
print(E2)
# print(t)
# print(E)
print(applied_gates)
print(num)
print(operations)
len(operations)
print(qc_result)
print(qc)

average = df['average']
t = df['t']
plt.figure(figsize=(18, 16), dpi=100)
plt.xlabel('time', fontsize=20)
plt.ylabel('E', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(t, average, marker='o', markerfacecolor='blue',
         markersize=5, color='skyblue')
# %%
