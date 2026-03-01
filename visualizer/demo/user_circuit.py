# Edit this file to define your own squin circuit, then run the smoke scene.
# The scene will show this code in a Jupyter-style cell, then visualize
# Logical -> Native+Placement -> Qubit Routing for your circuit.
#
# Required: define a callable named "circuit" with @squin.kernel(typeinfer=True, fold=True)
# that uses qubit.qalloc (e.g. q = qubit.qalloc(3)) and squin gates.

from bloqade import qubit, squin


@squin.kernel(typeinfer=True, fold=True)
def circuit():
    q = qubit.qalloc(3)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    squin.cx(q[0], q[2])
