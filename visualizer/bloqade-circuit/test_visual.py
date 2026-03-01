"""Quick test to see if matplotlib window opens."""

from bloqade import squin
from bloqade.visual.circuit import circuit_drawer
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend for better macOS compatibility
import matplotlib.pyplot as plt

@squin.kernel
def test():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])

print("Creating circuit visualization...")
fig = circuit_drawer(test, output="mpl", interactive=True)

if fig:
    print("Figure created. Showing window...")
    plt.show(block=True)  # block=True keeps window open
    print("Window closed.")
else:
    print("No figure returned.")
