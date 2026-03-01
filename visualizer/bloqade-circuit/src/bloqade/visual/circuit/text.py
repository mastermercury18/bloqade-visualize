"""Text-based circuit drawer for squin circuits."""

from __future__ import annotations

import shutil
from typing import Any

from .utils import InstructionNode


class TextDrawing:
    """ASCII art drawing of a quantum circuit."""

    def __init__(
        self,
        qubits: list[int],
        nodes: list[list[InstructionNode]],
        circuit: Any,
        reverse_bits: bool = False,
        initial_state: bool = False,
    ):
        """Initialize a text drawing.

        Args:
            qubits: List of qubit indices.
            nodes: List of layers, where each layer contains InstructionNodes.
            circuit: The original circuit (for reference).
            reverse_bits: Whether bits are reversed.
            initial_state: Whether to show initial state.
        """
        self.qubits = qubits
        self.nodes = nodes
        self.circuit = circuit
        self.reverse_bits = reverse_bits
        self.initial_state = initial_state
        self.plotbarriers = True
        self.line_length: int | None = None
        self.vertical_compression = "medium"

    def __str__(self) -> str:
        """Return the string representation of the circuit."""
        return self.single_string()

    def single_string(self) -> str:
        """Return the circuit as a single string."""
        lines = self._build_lines()
        return "\n".join(lines)

    def _build_lines(self) -> list[str]:
        """Build the lines for the circuit diagram."""
        if not self.qubits:
            return ["(empty circuit)"]

        # Determine line length
        if self.line_length is None:
            try:
                terminal_size = shutil.get_terminal_size()
                line_length = terminal_size.columns
            except (OSError, AttributeError):
                line_length = 80
        else:
            line_length = self.line_length if self.line_length > 0 else float("inf")

        # Build the circuit
        n_qubits = len(self.qubits)
        layers = self.nodes

        # Initialize wire lines
        wire_lines = [f"q{self.qubits[i]}: " for i in range(n_qubits)]

        # Process each layer
        for layer_idx, layer in enumerate(layers):
            # Determine layer width
            layer_width = self._calculate_layer_width(layer)
            
            # Initialize layer columns for each qubit
            layer_cols = [" " * layer_width for _ in range(n_qubits)]
            
            # Place gates in the layer
            for gate in layer:
                gate_str = self._format_gate(gate)
                # Pad to layer width
                gate_str = gate_str.center(layer_width)
                
                # Find qubit positions in our qubit list
                gate_qubit_positions = []
                for q in gate.qubits:
                    if q in self.qubits:
                        gate_qubit_positions.append(self.qubits.index(q))
                
                if gate_qubit_positions:
                    # Place gate on first qubit
                    layer_cols[gate_qubit_positions[0]] = gate_str
                    
                    # For multi-qubit gates, mark other qubits
                    if len(gate_qubit_positions) > 1:
                        for pos in gate_qubit_positions[1:]:
                            # Use a simple marker for connections
                            layer_cols[pos] = "●".center(layer_width)
            
            # Add layer to wire lines
            for i in range(n_qubits):
                wire_lines[i] += layer_cols[i]

            # Add layer separator
            if layer_idx < len(layers) - 1:
                separator = "─" * 3
                for i in range(n_qubits):
                    wire_lines[i] += separator

        # Handle line wrapping
        if line_length < float("inf"):
            wrapped_lines = []
            for line in wire_lines:
                if len(line) <= line_length:
                    wrapped_lines.append(line)
                else:
                    # Wrap the line
                    parts = []
                    current = line
                    while len(current) > line_length:
                        parts.append(current[:line_length])
                        current = current[line_length:]
                    if current:
                        parts.append(current)
                    wrapped_lines.extend(parts)
            return wrapped_lines

        return wire_lines

    def _calculate_layer_width(self, layer: list[InstructionNode]) -> int:
        """Calculate the width needed for a layer."""
        if not layer:
            return 3
        max_width = 0
        for gate in layer:
            gate_str = self._format_gate(gate)
            max_width = max(max_width, len(gate_str))
        return max(max_width, 3)

    def _format_gate(self, gate: InstructionNode) -> str:
        """Format a gate as a string."""
        name = gate.name.upper()
        if gate.adjoint:
            name += "†"
        
        # Add parameters if present
        if gate.params:
            param_strs = []
            for key, value in gate.params.items():
                if key == "rads":
                    # Format radians nicely
                    value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                    param_strs.append(f"{value_str}")
                elif key == "exponent":
                    param_strs.append(f"^{value}")
                else:
                    param_strs.append(f"{key}={value}")
            if param_strs:
                name += "(" + ",".join(param_strs) + ")"
        
        # Pad to minimum width
        min_width = 3
        if len(name) < min_width:
            name = name.center(min_width)
        
        return name

    def dump(self, filename: str) -> None:
        """Dump the circuit to a file."""
        if not filename.endswith(".txt"):
            raise ValueError("Filename must end with .txt")
        
        with open(filename, "w") as f:
            f.write(self.single_string())
