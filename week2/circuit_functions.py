#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:54:26 2020

@author: kiran

Some code copied from qcoptim to generate circuits
"""
import qiskit as qk
import numpy as np

def newCircuit(nb_qubits =4 , 
               depth = 1,
               verbose = False):
    """
    Creates a new random 'cliffod' circuit"""
    if verbose:
        Warning("Currently only makes a reduced Clifford circuit")
    
    # Construct circuit
    circuit = qk.QuantumCircuit(nb_qubits)
    # Need to increase the gate set here... maybe this isn't the best way
    # Might need to use u3 params instead, but this will do for now
    single_rotatoins = [circuit.h,
                        circuit.s,
                        circuit.x,
                        circuit.y,
                        circuit.z]
    
    def entangle_layer(circ):
        """
        Creates a linear entangeling layer"""
        for ii in range(0,circ.num_qubits-1, 2):
            circ.cx(ii,ii+1)
        for ii in range(1,circ.num_qubits-1, 2):
            circ.cx(ii,ii+1)
    
    def rotaiton_layer(circ):
        """
        Creates a layer of single qubit rotations based on the list 'single_rotatoins'"""
        random_points = np.random.randint(0, len(single_rotatoins), circ.num_qubits)
        for ii in range(circ.num_qubits):
            single_rotatoins[random_points[ii]](ii)
    

    # Apply first rotation layer (else CX layer does nothing)
    rotaiton_layer(circuit)
    
    # Loop though and alternate rotation and entangelment layers
    for ii in range(depth):
        entangle_layer(circuit)
        circuit.barrier() # this just makes circ.draw() look better
        rotaiton_layer(circuit)
    if verbose:
        print(circuit)
    return circuit


def updateCircuit(circuit,
                  verbose = False):
    """
    Takes an input circuit and switches exactly 1 single qubit gate to something 
    else. (Only tested with circuis made with newCircuit())
    """
    if verbose:
        Warning("Currently only replaces to h,s,x,y,z gates")
    possible_gates = list('hsxyz')
    
    # Convert circuit to qasm string so we can use string processing to switch
    qasm = circuit.qasm().split(';')
    
    
    # Make sure the gate you choose is not a cx gate
    gate_to_switch = np.random.randint(3,len(qasm)-1)
    while qasm[gate_to_switch][1:3] == 'cx' or qasm[gate_to_switch][1:3] == 'ba':
        gate_to_switch = np.random.randint(3,len(qasm)-1)
    
    # Get a new gate and make sure it's different form the current gate
    this_gate = qasm[gate_to_switch][1]
    new_gate = np.random.choice(possible_gates)
    while new_gate == this_gate:
        new_gate = np.random.choice(possible_gates)
    
    qasm[gate_to_switch] = '\n' + new_gate + ' ' + qasm[gate_to_switch].split(' ')[1]
    
    qasm = ';'.join(qasm)    
    circuit = qk.QuantumCircuit.from_qasm_str(qasm)
    
    if verbose:
        print(circuit)
        
    return circuit


# Quick test to make sure everyhing goes smoothly
if __name__ == '__main__':
    
    # Create circuit
    circuit = newCircuit(4, 4, True)
    
    # Update it 50 times and check for errors
    for ii in range(50):
        circuit = updateCircuit(circuit)
    
    print('passed basic test')