"""Example showing visual (matplotlib) circuit visualization."""

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
    print("Creating visual circuit diagrams...")
    print()
    
    # Option 1: Show interactive plot (opens a window)
    print("1. Opening interactive matplotlib window for Bell state circuit...")
    fig = circuit_drawer(
        bell_state, 
        output="mpl", 
        interactive=True,
        scale=1.2
    )
    print("   ✓ Window opened! Close it to continue.")
    print()
    
    # Option 2: Save to file
    print("2. Saving example circuit to 'circuit_visual.png'...")
    circuit_drawer(
        example_circuit,
        output="mpl",
        filename="circuit_visual.png",
        scale=1.5
    )
    print("   ✓ Saved!")
    print()
    
    # Option 3: Customize style
    print("3. Creating styled circuit...")
    custom_style = {
        "gate_color": "#2E86AB",
        "gate_text_color": "white",
        "wire_color": "#333333",
        "background_color": "#F5F5F5",
        "fontsize": 14,
    }
    circuit_drawer(
        bell_state,
        output="mpl",
        filename="circuit_styled.png",
        style=custom_style,
        scale=1.3
    )
    print("   ✓ Saved styled version to 'circuit_styled.png'!")
    print()
    
    print("All visualizations complete!")
    print("Check the generated PNG files in the current directory.")
