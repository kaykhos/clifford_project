#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:46:51 2020

@author: linmin
"""
from docplex.mp.model import Model
from qiskit.optimization.problems import QuadraticProgram
from qiskit.optimization.converters import QuadraticProgramToIsing
from qiskit import QuantumCircuit
from qiskit.aqua.operators import StateFn
import random
import numpy as np


# Number of qubits
n = 5
# Number of gates
g = 10

qc = QuantumCircuit(n)
qc.cx(0, 3)

def gates():
    a = random.randint(0, 4)#generate a random gate
    b = random.randint(0, n - 1)#generate a nunber that represents the index of qubit
    #c = random.sample(range(n), 2)
    if a == 0:
        qc.h(b)
    elif a == 1:
        qc.s(b)
    elif a == 2:
        qc.x(b)
    elif a == 3:
        qc.y(b)
    elif a == 4:
        qc.z(b)
#    elif a == 5:
#        qc.cx(c[0], c[1])

#Acting the gates on circuit
for i in range (g):
    gates()

# Prepare the Ising Hamiltonian
#n = 3 #number of qubits
a = 1.0 
k = 2
t = range(1, n+1)

mdl = Model()# build model with docplex
x = [mdl.binary_var() for i in range(n)]
objective = a*(k - mdl.sum(t[i]*x[i] for i in range(n)))**2
mdl.minimize(objective)

qp = QuadraticProgram()# convert to Qiskit's quadratic program
qp.from_docplex(mdl)

qp2ising = QuadraticProgramToIsing()# convert to Ising Hamiltonian
H, offset = qp2ising.encode(qp)
H_matrix = np.real(H.to_matrix())    

psi = StateFn(qc)

expectation_value = (~psi @ H @ psi).eval()

print(psi)
print(expectation_value)
