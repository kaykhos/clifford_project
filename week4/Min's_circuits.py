#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:33:18 2020
author: linmin
"""
import random
import numpy as np
import math
import quimb.tensor as qtn
import time

start = time.time()

# Number of qubits
n = 4

# Depth of the circuit
d = 1

# Number of single qubit gates
g = n * (d + 1)

# Maximum value of t
times = 1000 


Beta = 20

Energy_matrix = [1]

qubit_structure = []

energy = 1

t_star = 0

t = [0]
counts = 1
#Generate the sturucture array of the circuit
for i in range (d + 1):
    for j in range(n):
        qubit_structure.append(j)
        
def NN(n):
    """
    Generate an integer from 0 to 5 excluding n, 
    for example NN(2) = randomly select a number from 0, 1, 3, 4, 5
    """
    r = list(range(0, n)) + list(range(n + 1, 5))
    return random.choice(r)


def generate_gates():
    """
    Randomly generate an array represents the gates acting on circuit, 
    for if generate_gates = [0, 3, 1], it means we want to act gate 0 on
    first qubit, gate 3 on second qubit, and so on.
    """
    randomlist = []
    for i in range (g):
        m = random.randint(0, 5)
        randomlist.append(m)
    return randomlist

target_gates = generate_gates()

output_gates = generate_gates()


def gates(i, j, qc): # i is the qubit,j is the gate
    #qc = qtn.Circuit(data=qca)
    if j == 0:
       # qc.h(i)
        qc.apply_gate('H', i)
    elif j == 1:
       # qc.s(i)
        qc.apply_gate('S', i)
    elif j == 2:
       # qc.x(i)
        qc.apply_gate('X', i)
    elif j == 3:
       # qc.y(i)
        qc.apply_gate('Y', i)
    elif j == 4:
       # qc.z(i)
        qc.apply_gate('Z', i)
    elif j == 5:
       # qc.id(i)
        qc.apply_gate('IDEN', i)
 
def Target_circuit():
    """
    Generate the circuit of the target state
    """
    qc = qtn.Circuit(N=n, tags='PSI0')
    #qc = qtn.Tensor(data=qcc)
    for i in range (d):
        for j in range (n):
            gates(qubit_structure[n * i + j], target_gates[n * i + j], qc)
        for j in range (0, n - 1, 2):   
            qc.apply_gate('CNOT', j, j + 1, gate_round=1)
        for j in range (1, n - 1 ,2): 
            qc.apply_gate('CNOT', j, j + 1, gate_round=1)
        #qc.barrier()
    for i in range (n * d , n * d + n):
        gates(qubit_structure[i], target_gates[i], qc)
    
    return qc

#Transfer the circuit to state vector
target = Target_circuit().to_dense()

def Output_circuit():
    """
    Generate the circuit of the output state
    """
    global energy, t_star
    P = 0
    qc = qtn.Circuit(N=n, tags='PSI0')
    new_gates = []
    Ord_Ga = random.randint(0, g - 1)#Nth gates 
    Ind_Ga = NN(output_gates[Ord_Ga])
    for i in range (g):
        new_gates.append(output_gates[i])
    new_gates[Ord_Ga] = Ind_Ga
    for i in range (d):
        for j in range (n):
            gates(qubit_structure[n * i + j], target_gates[n * i + j], qc)
        for j in range (0, n - 1, 2):   
            qc.apply_gate('CNOT', j, j + 1, gate_round=1)
        for j in range (1, n - 1 ,2): 
            qc.apply_gate('CNOT', j, j + 1, gate_round=1)
    for i in range (n * d , n * d + n):
        gates(qubit_structure[i], target_gates[i], qc)
    
    psi = qc.to_dense()
    E = np.real(1 - ( psi.H @ target )*( target.H @ psi ))
    if E < 0.01 and t_star == 0:
        t_star = counts
    if E < np.real(energy):
        P = 1
    else:
        P = math.exp(-Beta*(np.real(E) - np.real(energy)))
    if random.random() < P:
        energy = E
        Energy_matrix.append(np.real(energy))
        output_gates[Ord_Ga] = Ind_Ga
    else:
        Energy_matrix.append(energy)
    return qc





