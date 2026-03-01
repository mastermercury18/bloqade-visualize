"""Utility functions for circuit visualization."""

from __future__ import annotations

from typing import Any
from collections import defaultdict
from dataclasses import dataclass

import cirq
from kirin import ir

from bloqade.cirq_utils import emit_circuit


@dataclass
class InstructionNode:
    """Represents a gate instruction in the circuit."""

    gate_type: str
    qubits: list[int]
    name: str
    params: dict[str, Any] | None = None
    adjoint: bool = False


def get_layered_instructions(
    circuit: ir.Method,
    reverse_bits: bool = False,
    justify: str | None = None,
    idle_wires: bool = True,
) -> tuple[list[int], list[list[InstructionNode]]]:
    """Extract layered instructions from a squin kernel method.

    This function converts the squin circuit to a cirq circuit first,
    then extracts the layered instructions from the cirq circuit.

    Args:
        circuit: The squin kernel method to analyze.
        reverse_bits: Whether to reverse the bit order.
        justify: How to justify gates ("left", "right", or "none").
        idle_wires: Whether to include idle wires.

    Returns:
        A tuple of (qubit_list, layered_nodes) where:
        - qubit_list: List of qubit indices
        - layered_nodes: List of layers, where each layer is a list of InstructionNode
    """
    if justify is None:
        justify = "left"

    # Convert squin circuit to cirq circuit
    try:
        cirq_circuit = emit_circuit(circuit, ignore_returns=True)
    except Exception as e:
        raise ValueError(f"Failed to convert squin circuit to cirq circuit: {e}")

    # Extract qubits
    all_qubits = sorted(cirq_circuit.all_qubits())
    if reverse_bits:
        all_qubits = list(reversed(all_qubits))

    # Create mapping from cirq qubits to indices
    # For LineQubit, use the x coordinate; for others, use position in sorted list
    qubit_to_index = {}
    for i, q in enumerate(all_qubits):
        if hasattr(q, 'x'):  # LineQubit
            qubit_to_index[q] = q.x
        else:
            qubit_to_index[q] = i
    
    qubits = [qubit_to_index[q] for q in all_qubits]

    # Extract gates from moments
    gates = []
    for moment in cirq_circuit:
        for op in moment:
            gate_info = _extract_gate_from_cirq_op(op, qubit_to_index)
            if gate_info:
                gates.append(gate_info)

    # Layer the gates
    layers = _layer_instructions(gates, qubits, justify)

    return (qubits, layers)


def _extract_gate_from_cirq_op(
    op: cirq.Operation, qubit_to_index: dict[cirq.Qid, int]
) -> InstructionNode | None:
    """Extract gate information from a cirq operation.

    Args:
        op: The cirq operation.
        qubit_to_index: Mapping from cirq qubits to indices.

    Returns:
        InstructionNode if the operation is a gate, None otherwise.
    """
    qubits = [qubit_to_index[q] for q in op.qubits]
    gate = op.gate

    # Get gate name
    gate_name = gate.__class__.__name__.lower()

    # Handle special cases
    params = {}
    adjoint = False

    # Check for adjoint gates
    if hasattr(gate, "_exponent") and gate._exponent == -1:
        adjoint = True
        gate_name = gate_name.replace("pow", "")

    # Extract parameters
    if hasattr(gate, "_exponent"):
        exp = gate._exponent
        if exp != 1 and exp != -1:
            params["exponent"] = exp

    # Use string-based gate name detection for better compatibility
    gate_class_name = gate.__class__.__name__.lower()
    
    # Map common gate types
    if "xpowgate" in gate_class_name or gate_class_name == "x":
        gate_name = "x"
    elif "ypowgate" in gate_class_name or gate_class_name == "y":
        gate_name = "y"
    elif "zpowgate" in gate_class_name or gate_class_name == "z":
        gate_name = "z"
    elif "hpowgate" in gate_class_name or gate_class_name == "h":
        gate_name = "h"
    elif "spowgate" in gate_class_name or gate_class_name == "s":
        gate_name = "s"
    elif "tpowgate" in gate_class_name or gate_class_name == "t":
        gate_name = "t"
    elif "cxpowgate" in gate_class_name or "cnot" in gate_class_name or gate_class_name == "cx":
        gate_name = "cx"
    elif "czpowgate" in gate_class_name or gate_class_name == "cz":
        gate_name = "cz"
    elif "cypowgate" in gate_class_name or gate_class_name == "cy":
        gate_name = "cy"
    elif "rx" in gate_class_name:
        gate_name = "rx"
        if hasattr(gate, "_rads"):
            params["rads"] = gate._rads
    elif "ry" in gate_class_name:
        gate_name = "ry"
        if hasattr(gate, "_rads"):
            params["rads"] = gate._rads
    elif "rz" in gate_class_name:
        gate_name = "rz"
        if hasattr(gate, "_rads"):
            params["rads"] = gate._rads

    return InstructionNode(
        gate_type=type(gate).__name__,
        qubits=qubits,
        name=gate_name,
        params=params if params else None,
        adjoint=adjoint,
    )


def _layer_instructions(
    gates: list[InstructionNode],
    qubits: list[int],
    justify: str,
) -> list[list[InstructionNode]]:
    """Layer instructions based on qubit dependencies.

    Args:
        gates: List of instruction nodes.
        qubits: List of qubit indices.
        justify: How to justify gates.

    Returns:
        List of layers, where each layer contains gates that can execute in parallel.
    """
    if justify == "none":
        # Each gate gets its own layer
        return [[gate] for gate in gates]

    # Simple greedy layering: place gates in the earliest layer where
    # all their qubits are available
    layers: list[list[InstructionNode]] = []
    qubit_last_layer: dict[int, int] = defaultdict(lambda: -1)

    for gate in gates:
        # Find the earliest layer where this gate can be placed
        earliest_layer = max(
            (qubit_last_layer[q] + 1 for q in gate.qubits),
            default=0,
        )

        # Ensure we have enough layers
        while len(layers) <= earliest_layer:
            layers.append([])

        layers[earliest_layer].append(gate)

        # Update last layer for each qubit
        for q in gate.qubits:
            qubit_last_layer[q] = earliest_layer

    return layers
