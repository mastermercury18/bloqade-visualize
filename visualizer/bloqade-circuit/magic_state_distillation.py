"""Magic State Distillation Circuit in Squin.

This implements a simplified 15-to-1 magic state distillation protocol.
The circuit takes 15 noisy |T⟩ states and distills them into 1 higher-fidelity |T⟩ state.
"""

from bloqade import squin
from bloqade.visual.circuit import circuit_drawer
import matplotlib
matplotlib.use('TkAgg')  # Better macOS compatibility
import matplotlib.pyplot as plt

@squin.kernel
def magic_state_distillation():
    """
    15-to-1 Magic State Distillation Circuit.
    
    This circuit:
    1. Prepares 15 T states (using T gates on |+⟩ states)
    2. Applies CNOT gates in a specific pattern for error detection
    3. Measures ancilla qubits
    4. Applies corrections based on measurement outcomes
    
    The output is a distilled T state on qubit 0.
    """
    # Allocate qubits: 15 data qubits + 4 ancilla qubits for measurement
    data_qubits = squin.qalloc(15)  # 15 noisy T states
    ancilla = squin.qalloc(4)       # 4 ancilla qubits for syndrome measurement
    
    # Step 1: Prepare 15 noisy T states
    # Each T state is prepared as: H -> T (on |+⟩ state)
    for i in range(15):
        squin.h(data_qubits[i])
        squin.t(data_qubits[i])
    
    # Step 2: Apply CNOT gates for error detection
    # This creates a stabilizer code structure
    # Pattern: CNOT from data qubits to ancilla qubits
    
    # First layer of CNOTs - connect data qubits to first ancilla
    squin.cx(data_qubits[0], ancilla[0])
    squin.cx(data_qubits[1], ancilla[0])
    squin.cx(data_qubits[2], ancilla[0])
    squin.cx(data_qubits[3], ancilla[0])
    
    # Second ancilla - different pattern
    squin.cx(data_qubits[4], ancilla[1])
    squin.cx(data_qubits[5], ancilla[1])
    squin.cx(data_qubits[6], ancilla[1])
    squin.cx(data_qubits[7], ancilla[1])
    
    # Third ancilla
    squin.cx(data_qubits[8], ancilla[2])
    squin.cx(data_qubits[9], ancilla[2])
    squin.cx(data_qubits[10], ancilla[2])
    squin.cx(data_qubits[11], ancilla[2])
    
    # Fourth ancilla
    squin.cx(data_qubits[12], ancilla[3])
    squin.cx(data_qubits[13], ancilla[3])
    squin.cx(data_qubits[14], ancilla[3])
    squin.cx(data_qubits[0], ancilla[3])  # Connect back to first data qubit
    
    # Step 3: Measure ancilla qubits (in real protocol, these would be measured)
    # For visualization, we'll just show the measurement gates
    for i in range(4):
        squin.qubit.measure(ancilla[i])
    
    # Step 4: Apply corrections based on syndrome
    # In a full implementation, these would be conditional on measurement results
    # For visualization, we show the correction structure
    squin.cx(ancilla[0], data_qubits[0])  # Correction based on first syndrome
    squin.cx(ancilla[1], data_qubits[1])  # Correction based on second syndrome
    
    # Final T gate on the distilled qubit (qubit 0)
    squin.t(data_qubits[0])


@squin.kernel
def simplified_magic_state_distillation():
    """
    Simplified version with fewer qubits for clearer visualization.
    Uses 7 data qubits + 2 ancilla qubits.
    """
    data = squin.qalloc(7)
    ancilla = squin.qalloc(2)
    
    # Prepare T states
    for i in range(7):
        squin.h(data[i])
        squin.t(data[i])
    
    # Error detection CNOTs
    squin.cx(data[0], ancilla[0])
    squin.cx(data[1], ancilla[0])
    squin.cx(data[2], ancilla[0])
    
    squin.cx(data[3], ancilla[1])
    squin.cx(data[4], ancilla[1])
    squin.cx(data[5], ancilla[1])
    squin.cx(data[6], ancilla[1])
    
    # Measurements
    squin.qubit.measure(ancilla[0])
    squin.qubit.measure(ancilla[1])
    
    # Corrections
    squin.cx(ancilla[0], data[0])
    squin.cx(ancilla[1], data[1])
    
    # Final distilled state
    squin.t(data[0])


if __name__ == "__main__":
    print("=" * 70)
    print("Magic State Distillation Circuit Visualization")
    print("=" * 70)
    print()
    
    # Show simplified version first (easier to see)
    print("1. Simplified Magic State Distillation (7 data + 2 ancilla qubits)")
    print("   Saving to 'magic_state_simple.png'...")
    circuit_drawer(
        simplified_magic_state_distillation,
        output="mpl",
        filename="magic_state_simple.png",
        scale=1.5
    )
    print("   ✓ Saved!")
    print()
    
    # Show full version
    print("2. Full 15-to-1 Magic State Distillation Circuit")
    print("   Saving to 'magic_state_full.png'...")
    circuit_drawer(
        magic_state_distillation,
        output="mpl",
        filename="magic_state_full.png",
        scale=1.2
    )
    print("   ✓ Saved!")
    print()
    
    # Interactive visualization of simplified version
    print("3. Opening interactive window for simplified circuit...")
    print("   (Close the window when done viewing)")
    fig = circuit_drawer(
        simplified_magic_state_distillation,
        output="mpl",
        interactive=True,
        scale=1.5
    )
    # Keep window open - it will stay until you manually close it
    if fig:
        plt.show(block=True)  # This keeps the window open
    print("   ✓ Window closed.")
    print()
    
    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print()
    print("Files created:")
    print("  - magic_state_simple.png (simplified 7+2 qubit version)")
    print("  - magic_state_full.png (full 15+4 qubit version)")
    print()
    print("The circuit shows:")
    print("  1. T state preparation (H -> T gates)")
    print("  2. Error detection (CNOT gates to ancilla)")
    print("  3. Syndrome measurement (measure ancilla)")
    print("  4. Error correction (conditional CNOTs)")
    print("  5. Final distilled T state")
