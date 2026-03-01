"""Example showing how to open interactive matplotlib windows."""

from bloqade import squin
from bloqade.visual.circuit import circuit_drawer
import matplotlib
# Set backend before importing pyplot for better macOS compatibility
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5
import matplotlib.pyplot as plt

@squin.kernel
def bell_state():
    """Create a Bell state."""
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])

@squin.kernel
def example_circuit():
    """Example circuit with multiple gates."""
    q = squin.qalloc(3)
    squin.h(q[0])
    squin.x(q[1])
    squin.cx(q[0], q[1])
    squin.cx(q[1], q[2])
    squin.t(q[0])

if __name__ == "__main__":
    print("=" * 60)
    print("Interactive Circuit Visualization")
    print("=" * 60)
    print()
    print("This will open a matplotlib window.")
    print("Close the window to continue to the next circuit.")
    print()
    
    # Show first circuit - window will stay open until you close it
    print("1. Showing Bell state circuit...")
    fig1 = circuit_drawer(bell_state, output="mpl", interactive=True)
    print("   ✓ Bell state window closed.")
    print()
    
    # Show second circuit
    print("2. Showing example circuit...")
    fig2 = circuit_drawer(example_circuit, output="mpl", interactive=True)
    print("   ✓ Example circuit window closed.")
    print()
    
    print("All visualizations complete!")
    print()
    print("Tip: If windows don't open, try:")
    print("  - matplotlib.use('TkAgg')  # for Tk backend")
    print("  - matplotlib.use('Qt5Agg')  # for Qt backend (requires PyQt5)")
