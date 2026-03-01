"""Test that matplotlib window stays open."""

from bloqade import squin
from bloqade.visual.circuit import circuit_drawer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

@squin.kernel
def test():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])

print("Opening window - it will stay open until you manually close it...")
print("(The window should NOT close automatically)")
fig = circuit_drawer(test, output="mpl", interactive=True)

# Explicitly keep it open
if fig:
    plt.show(block=True)  # This blocks until window is closed

print("Window closed!")
