"""Example usage of circuit visualization."""

from bloqade import squin
from bloqade.visual.circuit import circuit_drawer

# Create a simple circuit
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
    print("Bell State Circuit (Text Output)")
    print("=" * 60)
    result = circuit_drawer(bell_state, output="text")
    print(result)
    print()
    
    print("=" * 60)
    print("Example Circuit (Text Output)")
    print("=" * 60)
    result = circuit_drawer(example_circuit, output="text")
    print(result)
    print()
    
    print("=" * 60)
    print("Bell State Circuit (Matplotlib Output)")
    print("=" * 60)
    print("Opening matplotlib window...")
    fig = circuit_drawer(bell_state, output="mpl", interactive=True)
    if fig:
        print("✓ Circuit visualization displayed!")
        print("Close the matplotlib window to continue...")
    
    print()
    print("=" * 60)
    print("Saving circuit to file")
    print("=" * 60)
    circuit_drawer(example_circuit, output="mpl", filename="circuit_example.png")
    print("✓ Saved to circuit_example.png")
