"""Matplotlib-based circuit drawer for squin circuits."""

from __future__ import annotations

from typing import Any

try:
    import matplotlib
    # Set interactive backend for macOS
    import sys
    if sys.platform == "darwin":
        matplotlib.use("TkAgg")  # Use TkAgg backend on macOS for better compatibility
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .utils import InstructionNode


class MatplotlibDrawer:
    """Matplotlib-based circuit drawer."""

    def __init__(
        self,
        qubits: list[int],
        nodes: list[list[InstructionNode]],
        circuit: Any,
        scale: float | None = None,
        style: dict | str | None = None,
        reverse_bits: bool = False,
        plot_barriers: bool = True,
        fold: int = 25,
        ax: Any | None = None,
        initial_state: bool = False,
    ):
        """Initialize the matplotlib drawer.

        Args:
            qubits: List of qubit indices.
            nodes: List of layers, where each layer contains InstructionNodes.
            circuit: The original circuit (for reference).
            scale: Scaling factor for the figure.
            style: Style dictionary or name.
            reverse_bits: Whether bits are reversed.
            plot_barriers: Whether to plot barriers.
            fold: Number of layers before folding.
            ax: Optional matplotlib axes to use.
            initial_state: Whether to show initial state.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for matplotlib circuit drawing. "
                "Install it with: pip install matplotlib"
            )

        self.qubits = qubits
        self.nodes = nodes
        self.circuit = circuit
        self.scale = scale if scale is not None else 1.0
        self.style = style if isinstance(style, dict) else {}
        self.reverse_bits = reverse_bits
        self.plot_barriers = plot_barriers
        self.fold = fold
        self.ax = ax
        self.initial_state = initial_state

        # Default style
        self.default_style = {
            "gate_color": "#1f77b4",
            "gate_text_color": "white",
            "wire_color": "black",
            "background_color": "white",
            "fontsize": 12,
        }
        self.style_dict = {**self.default_style, **self.style}

    def draw(self, filename: str | None = None) -> Any:
        """Draw the circuit.

        Args:
            filename: Optional filename to save the figure.

        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        if self.ax is None:
            fig, ax = plt.subplots(figsize=(12 * self.scale, 6 * self.scale))
        else:
            ax = self.ax
            fig = ax.figure

        n_qubits = len(self.qubits)
        if n_qubits == 0:
            ax.text(0.5, 0.5, "Empty Circuit", ha="center", va="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return fig if self.ax is None else None

        # Calculate layout
        layers = self.nodes
        n_layers = len(layers)

        # Set up coordinate system
        # Y-axis: qubits (top to bottom)
        # X-axis: layers (left to right)
        qubit_spacing = 1.0
        layer_spacing = 1.0

        # Draw wires
        for i, qubit_idx in enumerate(self.qubits):
            y = n_qubits - i - 1  # Reverse so q0 is at top
            x_start = 0
            x_end = n_layers * layer_spacing
            ax.plot(
                [x_start, x_end],
                [y, y],
                color=self.style_dict["wire_color"],
                linewidth=1,
                zorder=0,
            )

            # Add qubit label
            ax.text(
                x_start - 0.2,
                y,
                f"q{qubit_idx}",
                ha="right",
                va="center",
                fontsize=self.style_dict["fontsize"],
            )

            # Add initial state
            if self.initial_state:
                ax.text(
                    x_start - 0.1,
                    y,
                    "|0⟩",
                    ha="left",
                    va="center",
                    fontsize=self.style_dict["fontsize"] - 2,
                )

        # Draw gates
        for layer_idx, layer in enumerate(layers):
            x = layer_idx * layer_spacing + layer_spacing / 2

            for gate in layer:
                self._draw_gate(ax, gate, x, layer_spacing, qubit_spacing, n_qubits)

        # Set limits and style
        ax.set_xlim(-0.5, n_layers * layer_spacing + 0.5)
        ax.set_ylim(-0.5, n_qubits - 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(self.style_dict["background_color"])

        if filename:
            fig.savefig(filename, bbox_inches="tight", dpi=150)

        return fig if self.ax is None else None

    def _draw_gate(
        self,
        ax: Any,
        gate: InstructionNode,
        x: float,
        layer_spacing: float,
        qubit_spacing: float,
        n_qubits: int,
    ):
        """Draw a single gate.

        Args:
            ax: Matplotlib axes.
            gate: The gate to draw.
            x: X position of the gate.
            layer_spacing: Spacing between layers.
            qubit_spacing: Spacing between qubits.
            n_qubits: Total number of qubits.
        """
        gate_qubits = gate.qubits
        if not gate_qubits:
            return

        # Calculate gate position
        min_qubit = min(gate_qubits)
        max_qubit = max(gate_qubits)
        
        # Map qubit indices to y positions
        qubit_to_y = {self.qubits[i]: n_qubits - i - 1 for i in range(n_qubits)}
        y_positions = [qubit_to_y.get(q, q) for q in gate_qubits]
        y_min = min(y_positions)
        y_max = max(y_positions)

        gate_width = layer_spacing * 0.6
        gate_height = (y_max - y_min) * qubit_spacing + 0.4

        # Draw gate box
        gate_y = (y_min + y_max) / 2
        gate_box = FancyBboxPatch(
            (x - gate_width / 2, gate_y - gate_height / 2),
            gate_width,
            gate_height,
            boxstyle="round,pad=0.1",
            facecolor=self.style_dict["gate_color"],
            edgecolor="black",
            linewidth=1.5,
            zorder=1,
        )
        ax.add_patch(gate_box)

        # Draw gate label
        gate_name = gate.name.upper()
        if gate.adjoint:
            gate_name += "†"
        
        # Add parameters if present
        if gate.params:
            param_strs = []
            for key, value in gate.params.items():
                if key == "rads":
                    value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                    param_strs.append(value_str)
                elif key == "exponent":
                    param_strs.append(f"^{value}")
            if param_strs:
                gate_name += "\n" + ",".join(param_strs)

        ax.text(
            x,
            gate_y,
            gate_name,
            ha="center",
            va="center",
            color=self.style_dict["gate_text_color"],
            fontsize=self.style_dict["fontsize"] - 2,
            weight="bold",
            zorder=2,
        )

        # Draw control lines for multi-qubit gates
        if len(gate_qubits) > 1:
            for y in y_positions[1:]:
                # Draw vertical line
                ax.plot(
                    [x, x],
                    [y, gate_y],
                    color=self.style_dict["wire_color"],
                    linewidth=2,
                    zorder=0,
                )
                # Draw control dot
                control_dot = Circle(
                    (x, y),
                    0.08,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=2,
                )
                ax.add_patch(control_dot)
