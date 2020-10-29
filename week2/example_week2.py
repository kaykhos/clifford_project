from qiskit.quantum_info import Clifford
from qiskit.aqua.operators import PauliExpectation, CircuitSampler, StateFn
from qiskit.aqua import QuantumInstance
from qiskit import Aer
import numpy as np
import qiskit
from docplex.mp.model import Model
from qiskit.optimization.problems import QuadraticProgram
from qiskit.optimization.converters import QuadraticProgramToIsing
from qiskit import QuantumCircuit
from qiskit.aqua.operators import StateFn


# Prepare the Ising Hamiltonian
n = 3  # number of qubits
a = 1.0
k = 2
t = range(1, n + 1)

mdl = Model()  # build model with docplex
x = [mdl.binary_var() for i in range(n)]
objective = a * (k - mdl.sum(t[i] * x[i] for i in range(n)))**2
mdl.minimize(objective)

qp = QuadraticProgram()  # convert to Qiskit's quadratic program
qp.from_docplex(mdl)

qp2ising = QuadraticProgramToIsing()  # convert to Ising Hamiltonian
H, offset = qp2ising.encode(qp)
op = H.to_opflow()
# help(H)

# Prepare the state
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cy(1, 2)
psi = StateFn(qc)  # wrap it into a statefunction
print(psi)

# Calculate the expectation value for Ising Hamiltonian
print('expectation_value:', psi.adjoint().compose(op).compose(psi).eval().real)

# %% another way of calculating expectation


# define your backend or quantum instance (where you can add settings)
backend = Aer.get_backend('qasm_simulator')
q_instance = QuantumInstance(backend, shots=1024)

# define the state to sample
measurable_expression = StateFn(op, is_measurement=True).compose(psi)

# convert to expectation value
expectation = PauliExpectation().convert(measurable_expression)

# get state sampler (you can also pass the backend directly)
sampler = CircuitSampler(q_instance).convert(expectation)

# evaluate
print('Sampled:', sampler.eval().real)

# %%

print(qc)
print('Hamiltonian of Ising model:')
print(H.print_details())

# %% Randomly sample Clifford circuits and calculate <H> for each one.
# (E.g. only change single qubit gates, not their locations in the circuit)


# Bell state generation circuit
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
qc.cx(0, 3)
cliff = Clifford(qc)
cliff_circ = cliff.to_circuit()
cliff_sf = StateFn(cliff_circ)

# Print the Clifford
print(cliff)
print(cliff_circ)
# Print the Clifford destabilizer rows
print(cliff.destabilizer)

# Print the Clifford stabilizer rows
print(cliff.stabilizer)

#
# Prepare the Ising Hamiltonian
n = 4  # number of qubits
a = 1.0
k = 2
t = range(1, n + 1)

mdl = Model()  # build model with docplex
x = [mdl.binary_var() for i in range(n)]
objective = a * (k - mdl.sum(t[i] * x[i] for i in range(n)))**2
mdl.minimize(objective)

qp = QuadraticProgram()  # convert to Qiskit's quadratic program
qp.from_docplex(mdl)

qp2ising = QuadraticProgramToIsing()  # convert to Ising Hamiltonian
H, offset = qp2ising.encode(qp)
op = H.to_opflow()

print('expectation_value:', cliff_sf.adjoint().compose(
    op).compose(cliff_sf).eval().real)
