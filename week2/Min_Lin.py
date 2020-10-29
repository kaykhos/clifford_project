#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:40:45 2020

@author: linmin
"""
import numpy as np
from docplex.mp.model import Model
from qiskit.optimization.problems import QuadraticProgram
from qiskit.optimization.converters import QuadraticProgramToIsing
from qiskit import QuantumCircuit
from qiskit.aqua.operators import StateFn

from qiskit.aqua.operators import Z, X, Y, I, H, S
import random

count = 0
w = 0
P_0 = I.to_matrix()
P_1 = X.to_matrix()
P_2 = Y.to_matrix()
P_3 = Z.to_matrix()
HH = H.to_matrix()
S = S.to_matrix()
OPE = [P_0, HH, S]
pauli = [P_0, P_1, P_2, P_3]
Pauli = [I, X, Y, Z]

R = 2 #number of iteration == R

n = 5 #number of qubits
x_matrix = np.zeros((R, n))
z_matrix = np.zeros((R, n))
matrix = np.zeros((R, 2*n))
x_vector = []
z_vector = []
vector = []

ssample = np.zeros((R, n))

def generate_numbers(k):
    n = 5 #number of qubits
    randomlist = []
    for i in range (k - 1):
        randomlist.append(0)
    for i in range(k - 1, n):
        n = random.randint(0,3)
        randomlist.append(n)
    return randomlist

#sampling the first two rows 
for i in range (R):
    count = i//2 + 1
    sample = generate_numbers(count)
  #  x_vector = np.zeros(n)
   # z_vector = np.zeros(n)
  #  vector = []
    for j in range (n):
        if sample[j] == 0:
            x_matrix[i][j] = 0
            z_matrix[i][j] = 0
            matrix[i][2*j] = 0
            matrix[i][2*j + 1] = 0
        if sample[j] == 1:
            x_matrix[i][j] = 1
            z_matrix[i][j] = 0
            matrix[i][2*j] = 1
            matrix[i][2*j + 1] = 0
        if sample[j] == 2:
            x_matrix[i][j] = 1
            z_matrix[i][j] = 1
            matrix[i][2*j] = 1
            matrix[i][2*j + 1] = 1
        if sample[j] == 3:
            x_matrix[i][j] = 0
            z_matrix[i][j] = 1
            matrix[i][2*j] = 0
            matrix[i][2*j + 1] = 1
        ssample[i][j] = sample[j]

def find_index():
    for i in range (R):
        u = 0
        v = 0
        for k in range (n):
            if x_matrix[i - 1][k] != 0 and u == 0 and i % 2 == 1:
                u = k + 1
               
            if z_matrix[i - 1][k] != 0 and v == 0 and i % 2 == 1:
                v = k + 1
            if u > v and i % 2 == 1:
                w = v - 1
                z_matrix[i][w] = 0
                ssample[i][w] = x_matrix[i][w]
                matrix[i][2*w + 1] = 0
            if u <= v and i % 2 == 1:
                w = u - 1
                x_matrix[i][w] = 0
                matrix[i][2*w] = 0
                if z_matrix[i][w] == 1:
                    ssample[i][w] = 3
                else:
                    ssample[i][w] = 0
            if u == 0 and i % 2 == 1:
                w = v - 1
                z_matrix[i][w] = 0
                matrix[i][2*w + 1] = 0
                ssample[i][w] = x_matrix[i][w]
            if v == 0 and i % 2 == 1:
                w = u - 1
                x_matrix[i][w] = 0
                matrix[i][2*w] = 0
                if z_matrix[i][w] == 1:
                    ssample[i][w] = 3
                else:
                    ssample[i][w] = 0

    return (x_matrix, z_matrix, matrix, ssample, w)
find_index()

q =  find_index()[4]
def transform():
    k = 0
    if matrix[1][2*q] == 0:
        qq = 2*q
    else:
        qq = 2*q + 1
    for i in range (n):
        if ssample[0][i] != 0 and ssample[1][i] != 0 and ssample[0][i] != ssample[1][i]:
            k = k + 1
    if (-1)**k == -1:
        if (qq % 2) == 0 and matrix[0][qq + 1] == 0 and  matrix[1][qq + 1] == 0 and matrix[1][qq] == 0:
            matrix[1][qq + 1] = 1 
            matrix[1][qq] = 1
            x_matrix[1][q] = 1
            z_matrix[1][q] = 1
        if (qq % 2) == 0 and matrix[0][qq + 1] == 0 and  matrix[1][qq + 1] == 1 and matrix[1][qq] == 0:
            matrix[1][qq + 1] = 0
            matrix[1][qq] = 1
            x_matrix[1][q] = 1
            z_matrix[1][q] = 0
        if (qq % 2) == 0 and matrix[0][qq + 1] == 1 and  matrix[1][qq + 1] == 0 and matrix[1][qq] == 0:
            matrix[1][qq] = 1
            x_matrix[1][q] = 1
        if (qq % 2) == 0 and matrix[0][qq + 1] == 1 and  matrix[1][qq + 1] == 1 and matrix[1][qq] == 0:
            matrix[1][qq] = 1
            x_matrix[1][q] = 1
        if (qq % 2) == 1 and matrix[0][qq - 1] == 0 and  matrix[1][qq - 1] == 0 and matrix[1][qq] == 0:
            matrix[1][qq] = 1
            matrix[1][qq - 1] = 1
            x_matrix[1][q] = 1
            z_matrix[1][q] = 1
        if (qq % 2) == 1 and matrix[0][qq - 1] == 0 and  matrix[1][qq - 1] == 1 and matrix[1][qq] == 0:
            matrix[1][qq] = 1
            matrix[1][qq - 1] = 0
            x_matrix[1][q] = 0
            z_matrix[1][q] = 1
    return (x_matrix, z_matrix, matrix)

transform()
#0 means H-gates, 1 means S-gates.
def step_one(r):
    List = np.zeros(n)
    for i in range (n):
        if z_matrix[r][i] == 1:
            if x_matrix[r][i] == 0:
                List[i] = 1
            else:
                List[i] = 2
            x_matrix[r][i] = 1
            z_matrix[r][i] = 0
    #print('aaaab', List)
    return (x_matrix, z_matrix, List)
ak = step_one(0)[2]
step_one(0)
#ak = step_one(0)[2]
#x_matrix = step_one(0)[0]
z_matrix = step_one(0)[1]
#List = step_one(0)[2]
#print('aaaab', z_matrix)

def step_two(r):
    j_matrix = []
    group = []
    for i in range (n):
        if x_matrix[r][i] == 1:
            j_matrix.append(i)
    for i in range (len(j_matrix)):      
        x_matrix[r][j_matrix[i]] = 0
        x_matrix[r][j_matrix[0]] = 1
    for i in range (len(j_matrix) - 1):
        group.append([j_matrix[0], j_matrix[i + 1]])
#    print('aaaabb', x_matrix, group)
    return(x_matrix, group)

bk = step_two(0)[1]
step_two(0)

count_1 = 0
for i in range (n - 1):
    if z_matrix[1][i + 1] == 0:
        count_1 = count_1 + 1
if count_1 != n - 1 or z_matrix[1][1] != 1:
    xx = x_matrix[0][0] 
    xxx = x_matrix[1][0]
    zz = z_matrix[0][0] 
    zzz = z_matrix[1][0]
    x_matrix[0][0] = zz
    x_matrix[1][0] = zzz
    z_matrix[0][0] = xx
    z_matrix[1][0] = xxx

aak = step_one(1)[2]
step_one(1)
bbk = step_two(1)[1]
step_two(1)

#ak = step_one(1)[2]
  
print(ak)
print(bk)
print(aak)
print(bbk)
#print(z_matrix)

cx_1 = []
cx_2 = []
cxx_1 = []
cxx_2 = []

qc = QuantumCircuit(n)
for i in range (n):
    if ak[i] == 1:
        qc.h(i)
    if ak[i] == 2:
        qc.s(i)
for i in range (len(bk)):
    cx_1.append(bk[i][0])
    cx_2.append(bk[i][1])
    qc.cx(cx_1[i], cx_2[i])
    
for i in range (n):
    if aak[i] == 1:
        qc.h(i)
    if aak[i] == 2:
        qc.s(i)

for i in range (len(bbk)):
    cxx_1.append(bbk[i][0])
    cxx_2.append(bbk[i][1])
    qc.cx(cxx_1[i], cxx_2[i])
print(qc)


psi = StateFn(qc)
    
    
    
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
        
expectation_value = (~psi @ H @ psi).eval()


print('expectation_value:', expectation_value)
print(qc)
print('Hamiltonian of Ising model:')
print(H_matrix)
print('dimension of Hamiltonian:', H_matrix.shape)
        
        
        

    

        



    
    
    
    
    
    
    
    
    
    
    
    