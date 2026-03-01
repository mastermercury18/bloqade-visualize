from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Sequence

from kirin import ir
from kirin.dialects import py


@dataclass(frozen=True)
class GateOp:
    name: str
    qubits: tuple[ir.SSAValue, ...]
    controls: tuple[ir.SSAValue, ...]
    targets: tuple[ir.SSAValue, ...]
    stmt: ir.Statement


def _is_ilist_new(stmt: ir.Statement) -> bool:
    module = stmt.__class__.__module__
    return stmt.__class__.__name__ == "New" and "kirin.dialects.ilist" in module


def _collect_ilist_defs(mt: ir.Method) -> dict[ir.SSAValue, tuple[ir.SSAValue, ...]]:
    defs: dict[ir.SSAValue, tuple[ir.SSAValue, ...]] = {}
    for stmt in mt.callable_region.walk():
        if _is_ilist_new(stmt) and stmt.results:
            defs[stmt.results[0]] = tuple(stmt.args)
    return defs


def _flatten_qubits(
    value: object,
    list_defs: dict[ir.SSAValue, tuple[ir.SSAValue, ...]],
) -> tuple[ir.SSAValue, ...]:
    if isinstance(value, tuple):
        items = value
    elif isinstance(value, list):
        items = tuple(value)
    else:
        items = (value,)

    out: list[ir.SSAValue] = []
    for item in items:
        if isinstance(item, ir.SSAValue) and item in list_defs:
            out.extend(list_defs[item])
        elif isinstance(item, ir.SSAValue):
            out.append(item)
        elif isinstance(item, (tuple, list)):
            out.extend(_flatten_qubits(item, list_defs))
    return tuple(out)


def _qubit_args_from_stmt(
    stmt: ir.Statement,
    list_defs: dict[ir.SSAValue, tuple[ir.SSAValue, ...]],
) -> tuple[ir.SSAValue, ...]:
    qubits: list[ir.SSAValue] = []

    controls = getattr(stmt, "controls", ())
    targets = getattr(stmt, "targets", ())
    if controls or targets:
        qubits.extend(_flatten_qubits(controls, list_defs))
        qubits.extend(_flatten_qubits(targets, list_defs))
        return tuple(qubits)

    qubits_attr = getattr(stmt, "qubits", None)
    if qubits_attr is not None:
        return _flatten_qubits(qubits_attr, list_defs)

    qubit_attr = getattr(stmt, "qubit", None)
    if qubit_attr is not None:
        return _flatten_qubits(qubit_attr, list_defs)

    reg_attr = getattr(stmt, "reg", None)
    if reg_attr is not None:
        return _flatten_qubits(reg_attr, list_defs)

    if hasattr(stmt, "args"):
        for arg in stmt.args:
            if _is_qubit_arg(arg):
                qubits.extend(_flatten_qubits(arg, list_defs))

    return tuple(qubits)


def _is_qubit_arg(arg: object) -> bool:
    t = getattr(arg, "type", None)
    if t is None:
        t = getattr(arg, "type_", None)
    if t is None:
        return False
    name = str(t)
    return "Qubit" in name or "qubit" in name


def _const_value(value: object) -> object | None:
    if isinstance(value, ir.SSAValue) and isinstance(value.owner, py.Constant):
        return value.owner.value
    return None


def _qubit_key(value: object) -> object:
    if not isinstance(value, ir.SSAValue):
        return value
    owner = value.owner
    if isinstance(owner, py.GetItem):
        idx = _const_value(owner.index)
        if isinstance(idx, int):
            return ("getitem", owner.obj, idx)
    return value


def _gate_label(stmt: ir.Statement) -> str:
    return getattr(stmt, "name", stmt.__class__.__name__)


def _is_squin_gate(stmt: ir.Statement) -> bool:
    module = stmt.__class__.__module__
    return "squin.gate" in module or "native.dialects.gate" in module


def _is_measure(stmt: ir.Statement) -> bool:
    if stmt.__class__.__name__ != "Invoke":
        return False
    callee = getattr(stmt, "callee", None)
    if callee is None:
        return False
    return "measure" in getattr(callee, "__name__", "")


def _is_func_invoke(stmt: ir.Statement) -> bool:
    module = stmt.__class__.__module__
    return stmt.__class__.__name__ == "Invoke" and "kirin.dialects.func" in module


def _callee_name(callee: object) -> str:
    for attr in ("name", "sym_name"):
        name = getattr(callee, attr, None)
        if name:
            return str(name)
    text = str(callee)
    match = re.search(r'Method\\(\"([^\"]+)\"\\)', text)
    if match:
        return match.group(1)
    return getattr(callee, "__name__", "Gate")


def _extract_gate_ops(mt: ir.Method) -> list[GateOp]:
    list_defs = _collect_ilist_defs(mt)
    ops: list[GateOp] = []
    for stmt in mt.callable_region.walk():
        if _is_squin_gate(stmt):
            controls = _flatten_qubits(getattr(stmt, "controls", ()), list_defs)
            targets = _flatten_qubits(getattr(stmt, "targets", ()), list_defs)
            qubits = _qubit_args_from_stmt(stmt, list_defs)
            ops.append(
                GateOp(
                    name=_gate_label(stmt),
                    qubits=qubits,
                    controls=controls,
                    targets=targets,
                    stmt=stmt,
                )
            )
        elif _is_measure(stmt):
            qubits = _qubit_args_from_stmt(stmt, list_defs)
            ops.append(
                GateOp(
                    name="M",
                    qubits=qubits,
                    controls=(),
                    targets=(),
                    stmt=stmt,
                )
            )
        elif _is_func_invoke(stmt):
            callee = getattr(stmt, "callee", None)
            qubits = _qubit_args_from_stmt(stmt, list_defs)
            if callee is not None and qubits:
                ops.append(
                    GateOp(
                        name=_callee_name(callee),
                        qubits=qubits,
                        controls=(),
                        targets=(),
                        stmt=stmt,
                    )
                )
    return ops


def build_squin_circuit_mobject(
    mt: ir.Method,
    *,
    wire_spacing: float = 0.8,
    step_spacing: float = 1.4,
    gate_width: float = 0.8,
    gate_height: float = 0.5,
    left_padding: float = 1.5,
    show_qubit_labels: bool = True,
    qubit_labeler: Callable[[int, ir.SSAValue], str] | None = None,
    gate_labeler: Callable[[GateOp], str] | None = None,
):
    try:
        from manim import VGroup, Line, Rectangle, Text, Dot
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Manim is required for squin circuit visualization."
        ) from exc

    ops = _extract_gate_ops(mt)
    qubits: list[object] = []
    seen: set[object] = set()
    key_to_value: dict[object, ir.SSAValue] = {}
    for op in ops:
        for q in op.controls + op.targets + op.qubits:
            key = _qubit_key(q)
            if key not in seen:
                seen.add(key)
                qubits.append(key)
                if isinstance(q, ir.SSAValue):
                    key_to_value[key] = q

    if not qubits:
        # Fall back to method entry block args if no gates were parsed yet.
        try:
            entry_block = mt.callable_region.blocks[0]
        except Exception:
            entry_block = None
        if entry_block is not None:
            for arg in entry_block.args:
                if _is_qubit_arg(arg) and arg not in seen:
                    key = _qubit_key(arg)
                    if key not in seen:
                        seen.add(key)
                        qubits.append(key)
                        if isinstance(arg, ir.SSAValue):
                            key_to_value[key] = arg

    if not qubits:
        return VGroup()

    if qubit_labeler is None:
        qubit_labeler = lambda i, _q: f"q{i}"

    if gate_labeler is None:
        gate_labeler = lambda op: op.name

    height = (len(qubits) - 1) * wire_spacing
    total_width = max(1, len(ops)) * step_spacing

    wires = VGroup()
    labels = VGroup()
    y_for_qubit: dict[object, float] = {}
    for i, key in enumerate(qubits):
        y = height / 2 - i * wire_spacing
        y_for_qubit[key] = y
        wires.add(Line((0, y, 0), (left_padding + total_width, y, 0)))
        if show_qubit_labels:
            qval = key_to_value.get(key, key)
            labels.add(
                Text(qubit_labeler(i, qval), font_size=24).move_to((-0.6, y, 0))
            )

    gates = VGroup()
    for idx, op in enumerate(ops):
        x = left_padding + idx * step_spacing
        label = gate_labeler(op)

        if op.controls or op.targets:
            if op.controls and op.targets and len(op.controls) == len(op.targets):
                for ctrl, tgt in zip(op.controls, op.targets):
                    y_ctrl = y_for_qubit[_qubit_key(ctrl)]
                    y_tgt = y_for_qubit[_qubit_key(tgt)]
                    gates.add(Line((x, y_ctrl, 0), (x, y_tgt, 0)))
                    gates.add(Dot(point=(x, y_ctrl, 0), radius=0.06))
                    box = Rectangle(width=gate_width, height=gate_height)
                    box.move_to((x, y_tgt, 0))
                    gates.add(box)
                    gates.add(Text(label, font_size=20).move_to(box.get_center()))
            else:
                ys = [y_for_qubit[_qubit_key(q)] for q in op.controls + op.targets]
                if ys:
                    gates.add(Line((x, min(ys), 0), (x, max(ys), 0)))
                for ctrl in op.controls:
                    gates.add(
                        Dot(point=(x, y_for_qubit[_qubit_key(ctrl)], 0), radius=0.06)
                    )
                for tgt in op.targets:
                    box = Rectangle(width=gate_width, height=gate_height)
                    box.move_to((x, y_for_qubit[_qubit_key(tgt)], 0))
                    gates.add(box)
                    gates.add(Text(label, font_size=20).move_to(box.get_center()))
        else:
            for q in op.qubits:
                box = Rectangle(width=gate_width, height=gate_height)
                box.move_to((x, y_for_qubit[_qubit_key(q)], 0))
                gates.add(box)
                gates.add(Text(label, font_size=20).move_to(box.get_center()))

    return VGroup(wires, labels, gates)


def make_squin_circuit_scene(
    mt: ir.Method,
    **kwargs,
):
    try:
        from manim import Scene
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Manim is required for squin circuit visualization."
        ) from exc

    class SquinCircuitScene(Scene):
        def construct(self):
            circuit = build_squin_circuit_mobject(mt, **kwargs)
            self.add(circuit)

    return SquinCircuitScene


def enable_squin_draw() -> None:
    from kirin.ir.method import Method

    def _draw(self: ir.Method, **kwargs):
        return build_squin_circuit_mobject(self, **kwargs)

    if not hasattr(Method, "draw"):
        setattr(Method, "draw", _draw)


def _looks_like_tex(label: str) -> bool:
    return "\\" in label or "{" in label or "}" in label


def _normalize_gate_kind(label: str) -> tuple[str, dict[str, object]]:
    raw = (label or "").strip()
    if not raw:
        return "CUSTOM", {}
    compact = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    params: dict[str, object] = {}
    for suffix in ("ADJ", "DAG", "DAGGER"):
        if compact.endswith(suffix):
            params["dag"] = True
            compact = compact[: -len(suffix)]
            break
    sqrt_map = {
        "SQRTX": "SQX",
        "SQRTY": "SQY",
        "SQRTZ": "SQZ",
    }
    compact = sqrt_map.get(compact, compact)
    if compact == "CX":
        compact = "CNOT"
    return compact, params


def build_squin_circuit_qat(
    mt: ir.Method,
    *,
    wire_spacing: float = 1.0,
    gate_size: float = 0.5,
    gate_spacing: float = 0.1,
    wire_length: float | None = None,
    gate_shape: str | None = "square",
    show_qubit_labels: bool = True,
    right_labels: bool | Sequence[str] = False,
    use_qat_defaults: bool = False,
    qat_style: str | None = None,
    qat_format: str | None = None,
    include_background: bool | None = None,
    slot_offset: int = 0,
    qubit_labeler: Callable[[int, ir.SSAValue], str] | None = None,
    gate_labeler: Callable[[GateOp], str] | None = None,
    strict: bool = False,
    **circuit_kwargs,
):
    try:
        from manimlib import VGroup
        from quantum_animation_toolbox.quera_circuit_lib import QuantumCircuit
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "quantum-animation-toolbox and ManimGL are required for QAT visualization."
        ) from exc

    ops = _extract_gate_ops(mt)
    qubits: list[object] = []
    seen: set[object] = set()
    key_to_value: dict[object, ir.SSAValue] = {}
    for op in ops:
        for q in op.controls + op.targets + op.qubits:
            key = _qubit_key(q)
            if key not in seen:
                seen.add(key)
                qubits.append(key)
                if isinstance(q, ir.SSAValue):
                    key_to_value[key] = q

    if not qubits:
        try:
            entry_block = mt.callable_region.blocks[0]
        except Exception:
            entry_block = None
        if entry_block is not None:
            for arg in entry_block.args:
                if _is_qubit_arg(arg) and arg not in seen:
                    key = _qubit_key(arg)
                    if key not in seen:
                        seen.add(key)
                        qubits.append(key)
                        if isinstance(arg, ir.SSAValue):
                            key_to_value[key] = arg

    if not qubits:
        return VGroup()

    if qubit_labeler is None:
        qubit_labeler = lambda i, _q: f"q_{{{i}}}"

    fmt = str(qat_format).strip().lower() if qat_format is not None else None
    if fmt in {"qat", "quera", "demo"}:
        wire_spacing = 1.0
        gate_size = 0.6
        gate_spacing = 0.3
        gate_shape = "square"
        wire_length = 5
        right_labels = True
        show_qubit_labels = True
        slot_offset = max(slot_offset, 1)
        if qubit_labeler is None:
            qubit_labeler = lambda _i, _q: r"\ket{0}"
        if include_background is None:
            include_background = True

    left_labels = (
        [
            qubit_labeler(i, key_to_value.get(q, q))
            for i, q in enumerate(qubits)
        ]
        if show_qubit_labels
        else False
    )

    circuit_args: dict[str, object] = {
        "num_wires": len(qubits),
        "left_labels": left_labels,
        "right_labels": right_labels,
        **circuit_kwargs,
    }

    if use_qat_defaults:
        if wire_length is not None:
            circuit_args["wire_length"] = wire_length
    else:
        pitch = gate_size + gate_spacing
        if wire_length is None:
            wire_length = max(1.0, len(ops) * pitch)
        circuit_args.update(
            {
                "wire_length": wire_length,
                "wire_spacing": wire_spacing,
                "gate_size": gate_size,
                "gate_spacing": gate_spacing,
                "gate_shape": gate_shape,
            }
        )

    circuit = QuantumCircuit(**circuit_args)

    wire_for = {q: i for i, q in enumerate(qubits)}

    def _add_custom_gate(label: str, wires: list[int], slot: int) -> None:
        if not wires:
            return
        top = min(wires)
        bottom = max(wires)
        wire_idx = (top, bottom) if top != bottom else top
        circuit.add_gate(
            "CUSTOM",
            wire_idx,
            slot,
            gate_shape=gate_shape,
            name=label or "U",
        )

    def _add_single_gate(kind: str, wire_idx: int, slot: int, extra: dict[str, object]):
        circuit.add_gate(
            kind,
            wire_idx,
            slot,
            gate_shape=gate_shape,
            **extra,
        )

    for idx, op in enumerate(ops):
        slot = idx + slot_offset
        raw_label = gate_labeler(op) if gate_labeler is not None else op.name
        raw_label = str(raw_label)
        kind_up, extra_params = _normalize_gate_kind(raw_label)

        if _looks_like_tex(raw_label) and kind_up != "CUSTOM":
            tex_wires = [
                wire_for[_qubit_key(q)] for q in (op.controls + op.targets + op.qubits)
            ]
            _add_custom_gate(raw_label, tex_wires, slot)
            continue

        if kind_up in {"CNOT", "CX"} and not (op.controls or op.targets):
            if len(op.qubits) == 2:
                ctrl, tgt = op.qubits
                _add_single_gate(
                    "CNOT",
                    wire_for[_qubit_key(tgt)],
                    slot,
                    {"control_idx": wire_for[_qubit_key(ctrl)], **extra_params},
                )
                continue
            if strict:
                raise ValueError(
                    f"{raw_label} requires 2 qubits when no controls/targets are provided."
                )

        if op.controls or op.targets:
            control_idxs = [wire_for[_qubit_key(q)] for q in op.controls]
            target_idxs = [wire_for[_qubit_key(q)] for q in op.targets]

            if kind_up in {"CNOT", "CZ"}:
                if len(control_idxs) != 1:
                    if strict:
                        raise ValueError(
                            f"{raw_label} requires a single control; got {len(control_idxs)}."
                        )
                    _add_custom_gate(raw_label, control_idxs + target_idxs, slot)
                    continue
                for tgt in target_idxs:
                    _add_single_gate(
                        kind_up,
                        tgt,
                        slot,
                        {"control_idx": control_idxs[0], **extra_params},
                    )
                continue

            if kind_up.startswith("C") and len(control_idxs) == 1:
                for tgt in target_idxs:
                    _add_single_gate(
                        kind_up,
                        tgt,
                        slot,
                        {"control_idx": control_idxs[0], **extra_params},
                    )
                continue

            if strict:
                raise ValueError(
                    f"Unsupported controlled gate {raw_label} with controls={len(control_idxs)} "
                    f"targets={len(target_idxs)}."
                )
            _add_custom_gate(raw_label, control_idxs + target_idxs, slot)
            continue

        if kind_up in {"M", "MEASURE", "MEAS"}:
            for q in op.qubits:
                _add_single_gate("M", wire_for[_qubit_key(q)], slot, extra_params)
            continue

        if kind_up in {"P", "PHASE"}:
            for q in op.qubits:
                _add_single_gate("PHASE", wire_for[_qubit_key(q)], slot, extra_params)
            continue

        if kind_up == "SWAP" and len(op.qubits) == 2:
            q0, q1 = op.qubits
            _add_single_gate(
                "SWAP",
                wire_for[_qubit_key(q1)],
                slot,
                {"control_idx": wire_for[_qubit_key(q0)], **extra_params},
            )
            continue

        for q in op.qubits:
            _add_single_gate(kind_up, wire_for[_qubit_key(q)], slot, extra_params)

    if qat_style is not None:
        style = str(qat_style).strip().lower()
        if style in {"qat", "quera", "default"}:
            from manimlib import BLACK, ORANGE
            from quantum_animation_toolbox.quera_colors import QUERA_PURPLE

            circuit.bg_color = BLACK
            circuit.recolor_wires("all", QUERA_PURPLE)
            circuit.recolor_labels("all", QUERA_PURPLE)
            circuit.recolor_gates("all", ORANGE)

    if include_background:
        from quantum_animation_toolbox.quera_colors import BackgroundColor, BLACK
        from manimlib import VGroup

        return VGroup(BackgroundColor(BLACK), circuit)

    return circuit


def make_squin_circuit_scene_qat(
    mt: ir.Method,
    **kwargs,
):
    try:
        from manimlib import Scene
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ManimGL is required for QAT circuit visualization."
        ) from exc

    class SquinCircuitScene(Scene):
        def construct(self):
            circuit = build_squin_circuit_qat(mt, **kwargs)
            self.add(circuit)

    return SquinCircuitScene


def enable_squin_draw_qat() -> None:
    from kirin.ir.method import Method

    def _draw_qat(self: ir.Method, **kwargs):
        return build_squin_circuit_qat(self, **kwargs)

    if not hasattr(Method, "draw_qat"):
        setattr(Method, "draw_qat", _draw_qat)
