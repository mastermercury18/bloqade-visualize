from manimlib import (
    DOWN,
    LEFT,
    UP,
    FadeIn,
    FadeOut,
    Scene,
    Tex,
    VGroup,
    Group,
    Dot,
    Circle,
    Line,
    Rectangle,
    Polygon,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    ORANGE,
    TEAL,
    YELLOW,
    GREEN,
    GREY_B,
)

from dataclasses import dataclass
import numpy as np
import sys
from pathlib import Path

from bloqade import qubit, squin
from bloqade.lanes.heuristics.fixed import LogicalLayoutHeuristic, LogicalPlacementStrategy
from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress, MoveType

# IMPORTANT:
# - emit_native: for a circuit visualization QAT can render
# - compute_layout_and_gate_sites_xy: for placed/native ops + physical sites (+ xy)
from bloqade.lanes.native_pipeline import GateSiteXY
from bloqade.lanes.upstream import NativeToPlace
from bloqade.analysis import address
from bloqade.lanes.analysis import layout as layout_analysis
from bloqade.lanes.analysis import placement as placement_analysis
from bloqade.lanes.dialects import place

# Prefer the local bloqade-circuit implementation for squin -> native decomposition.
_ROOT = Path(__file__).resolve().parents[1]
_CIRCUIT_SRC = _ROOT / "bloqade-circuit" / "src"
if _CIRCUIT_SRC.exists():
    sys.path.insert(0, str(_CIRCUIT_SRC))

from bloqade.native.upstream.squin2native import SquinToNative
from bloqade.rewrite.passes.aggressive_unroll import AggressiveUnroll

from bloqade.lanes.visualize_squin import build_squin_circuit_qat, _extract_gate_ops
from quantum_animation_toolbox.quera_colors import BackgroundColor, BLACK

@squin.kernel
def demo_logical(q0: qubit.Qubit, q1: qubit.Qubit, q2: qubit.Qubit):
    squin.h(q0)
    squin.cx(q0, q1)
    squin.cx(q0, q2)


@squin.kernel(typeinfer=True, fold=True)
def demo_native():
    q = qubit.qalloc(3)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    squin.cx(q[0], q[2])


def make_laser_cone(
    tip_point,
    height=0.55,
    base_width=0.35,
    color=GREEN,
    opacity=0.35,
    glow_radius=0.06,
):
    tip = np.array(tip_point)
    apex = tip + np.array([0.0, height, 0.0])
    left = apex + np.array([-base_width / 2, 0.0, 0.0])
    right = apex + np.array([base_width / 2, 0.0, 0.0])

    cone = Polygon(left, right, tip)
    cone.set_fill(color, opacity=opacity)
    cone.set_stroke(color, width=0)

    impact = Dot(tip, radius=glow_radius, color=color).set_opacity(0.9)

    laser = VGroup(cone, impact)
    laser.set_z_index(50)
    return laser


def compute_layout_and_gate_sites_xy_from_native(
    native_mt,
    *,
    layout_heuristic=None,
    placement_strategy=None,
    include_ops=(place.R, place.Rz, place.CZ, place.EndMeasure),
):
    """
    Same as compute_layout_and_gate_sites_xy, but starts from an already-native Method.
    This keeps placement aligned with the exact IR you visualize (e.g., after AggressiveUnroll).
    """
    place_mt = NativeToPlace().emit(native_mt)

    address_analysis = address.AddressAnalysis(place_mt.dialects)
    address_frame, _ = address_analysis.run_no_raise(place_mt)
    all_qubits = tuple(range(address_analysis.next_address))

    if layout_heuristic is None:
        layout_heuristic = LogicalLayoutHeuristic()
    if placement_strategy is None:
        placement_strategy = LogicalPlacementStrategy()

    initial_layout = layout_analysis.LayoutAnalysis(
        place_mt.dialects,
        layout_heuristic,
        address_frame.entries,
        all_qubits,
    ).get_layout_no_raise(place_mt)

    placement = placement_analysis.PlacementAnalysis(
        place_mt.dialects,
        initial_layout,
        address_frame.entries,
        placement_strategy,
    )
    placement_frame, _ = placement.run_no_raise(place_mt)

    include_ops = tuple(include_ops)
    arch_spec = layout_heuristic.arch_spec

    def loc_to_xy(loc: LocationAddress) -> tuple[float, float]:
        pts = arch_spec.get_positions(loc)
        if not pts:
            raise RuntimeError(f"No arch_spec position for site {loc}")
        x, y = pts[0][0], pts[0][1]
        return float(x), float(y)

    gates_xy = []
    for stmt in place_mt.callable_region.walk():
        if not isinstance(stmt, place.StaticPlacement):
            continue

        for blk in stmt.body.blocks:
            for inner in blk.stmts:
                if not isinstance(inner, include_ops):
                    continue

                state_after = placement_frame.entries.get(inner.state_after)
                if not isinstance(state_after, placement_analysis.ConcreteState):
                    continue

                logical_qubits = tuple(int(q) for q in inner.qubits)
                sites = tuple(state_after.layout[i] for i in logical_qubits)
                xy = tuple(loc_to_xy(loc) for loc in sites)

                gates_xy.append(
                    GateSiteXY(
                        op=type(inner).__name__,
                        logical_qubits=logical_qubits,
                        sites=sites,
                        xy=xy,
                    )
                )

    return initial_layout, gates_xy


@dataclass(frozen=True)
class GateRoute:
    op: str
    logical_qubits: tuple[int, ...]
    sites: tuple[LocationAddress, ...]
    xy: tuple[tuple[float, float], ...]
    move_layers: tuple[tuple[LaneAddress, ...], ...]
    layout_after: tuple[LocationAddress, ...]


def compute_layout_and_gate_routes_from_native(
    native_mt,
    *,
    layout_heuristic=None,
    placement_strategy=None,
    include_ops=(place.R, place.Rz, place.CZ, place.EndMeasure),
):
    """
    Returns initial layout plus a per-gate routing plan (move layers) and final gate sites.
    """
    place_mt = NativeToPlace().emit(native_mt)

    address_analysis = address.AddressAnalysis(place_mt.dialects)
    address_frame, _ = address_analysis.run_no_raise(place_mt)
    all_qubits = tuple(range(address_analysis.next_address))

    if layout_heuristic is None:
        layout_heuristic = LogicalLayoutHeuristic()
    if placement_strategy is None:
        placement_strategy = LogicalPlacementStrategy()

    initial_layout = layout_analysis.LayoutAnalysis(
        place_mt.dialects,
        layout_heuristic,
        address_frame.entries,
        all_qubits,
    ).get_layout_no_raise(place_mt)

    placement = placement_analysis.PlacementAnalysis(
        place_mt.dialects,
        initial_layout,
        address_frame.entries,
        placement_strategy,
    )
    placement_frame, _ = placement.run_no_raise(place_mt)

    include_ops = tuple(include_ops)
    arch_spec = layout_heuristic.arch_spec

    def loc_to_xy(loc: LocationAddress) -> tuple[float, float]:
        pts = arch_spec.get_positions(loc)
        if not pts:
            raise RuntimeError(f"No arch_spec position for site {loc}")
        x, y = pts[0][0], pts[0][1]
        return float(x), float(y)

    routes: list[GateRoute] = []
    for stmt in place_mt.callable_region.walk():
        if not isinstance(stmt, place.StaticPlacement):
            continue

        for blk in stmt.body.blocks:
            for inner in blk.stmts:
                if not isinstance(inner, include_ops):
                    continue

                state_after = placement_frame.entries.get(inner.state_after)
                if not isinstance(state_after, placement_analysis.ConcreteState):
                    continue

                logical_qubits = tuple(int(q) for q in inner.qubits)
                sites = tuple(state_after.layout[i] for i in logical_qubits)
                xy = tuple(loc_to_xy(loc) for loc in sites)
                move_layers = state_after.get_move_layers()

                routes.append(
                    GateRoute(
                        op=type(inner).__name__,
                        logical_qubits=logical_qubits,
                        sites=sites,
                        xy=xy,
                        move_layers=move_layers,
                        layout_after=state_after.layout,
                    )
                )

    return initial_layout, routes


class SquinQATSmokeScene(Scene):
    def construct(self):
        def _split_group(maybe_group):
            if hasattr(maybe_group, "gates"):
                return None, maybe_group
            try:
                if len(maybe_group) >= 2:
                    return maybe_group[0], maybe_group[1]
            except Exception:
                pass
            return None, maybe_group

        # =========================
        # Layer 1: Logical (UNCHANGED)
        # =========================
        title = Tex(r"\textbf{Logical: Defines the logical circuit we begin with.}").scale(0.7)
        title.to_edge(UP)

        group = build_squin_circuit_qat(
            demo_logical, use_qat_defaults=True, qat_style="quera", qat_format="demo"
        )

        background, circuit = _split_group(group)

        if background is not None:
            self.add(background)

        self.play(FadeIn(title), run_time=0.6)

        if getattr(circuit, "wires", None) is not None:
            self.play(FadeIn(circuit.wires), run_time=1.0)
        if getattr(circuit, "labels", None) is not None:
            self.play(FadeIn(circuit.labels), run_time=1.2)

        for gate in getattr(circuit, "gates", []):
            self.play(FadeIn(gate), run_time=0.8)

        all_objs = Group(*self.mobjects)
        self.play(FadeOut(all_objs), run_time=0.6)
        self.clear()

        # =========================
        # Layer 2: Visualize "native" + map to array
        # =========================
        layer2_bg = BackgroundColor(BLACK)
        layer2_title = Tex(
            r"\textbf{Native + Placement: Gates compiled and mapped to physical sites.}"
        ).scale(0.7)
        layer2_title.to_edge(UP)
        self.add(layer2_bg)
        self.play(FadeIn(layer2_title), run_time=0.6)

        # (A) VISUAL CIRCUIT: use the local squin2native decomposition directly
        native_mt = SquinToNative().emit(demo_native)
        # Inline broadcast kernels so decompositions (e.g., CX -> sqrt_y, CZ, sqrt_y) appear explicitly.
        AggressiveUnroll(native_mt.dialects, no_raise=True).fixpoint(native_mt)
        print("=== Native IR (after inline/decompose) ===")
        native_mt.print()
        print("=== Visualizer gate ops (after inline/decompose) ===")
        native_ops = list(_extract_gate_ops(native_mt))
        for op in native_ops:
            qubits = [str(q) for q in op.qubits]
            controls = [str(q) for q in op.controls]
            targets = [str(q) for q in op.targets]
            print(f"{op.name} qubits={qubits} controls={controls} targets={targets}")
        # Ensure the wire length can accommodate the decomposed gate count.
        # QAT uses discrete slots: num_slots = int(wire_length // pitch) + 1.
        slot_offset = 1
        gate_size = 0.6
        gate_spacing = 0.3
        pitch = gate_size + gate_spacing
        max_slot = slot_offset + len(native_ops) - 1
        wire_length = max(5.0, (max_slot + 1) * pitch)
        group2 = build_squin_circuit_qat(
            native_mt,
            use_qat_defaults=False,
            qat_style="quera",
            qat_format=None,
            include_background=True,
            wire_spacing=1.0,
            gate_size=gate_size,
            gate_spacing=gate_spacing,
            wire_length=wire_length,
            gate_shape="square",
            right_labels=True,
            show_qubit_labels=True,
            slot_offset=slot_offset,
        )
        _bg2, native_circuit = _split_group(group2)

        native_circuit.scale(0.85)
        native_circuit.to_edge(LEFT)

        # Fade in circuit subparts explicitly (more reliable than FadeIn(circuit) in some QAT versions)
        if getattr(native_circuit, "wires", None) is not None:
            self.play(FadeIn(native_circuit.wires), run_time=0.6)
        if getattr(native_circuit, "labels", None) is not None:
            self.play(FadeIn(native_circuit.labels), run_time=0.6)
        for g in getattr(native_circuit, "gates", []):
            self.play(FadeIn(g), run_time=1)

        # (B) EXTRACTION: placed/native ops + physical sites (+xy)
        # Use the same unrolled native IR as the visual circuit to keep indices aligned.
        layout, gates_xy = compute_layout_and_gate_sites_xy_from_native(native_mt)

        print("DEBUG layer2: initial layout size =", len(layout))
        print("DEBUG layer2: extracted placed/native ops =", len(gates_xy))
        if not gates_xy:
            print("ERROR: gates_xy is empty. No placed ops found to animate.")
            print("       This usually means no ConcreteState was produced or include_ops mismatch.")
            # Still continue to show the grid so you can see something.

        # (C) BUILD PHYSICAL GRID
        arch_spec = LogicalLayoutHeuristic().arch_spec
        raw_positions: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for word_id, word in enumerate(arch_spec.words):
            for site_id in range(len(word.sites)):
                loc = LocationAddress(word_id, site_id)
                pts = arch_spec.get_positions(loc)
                if pts:
                    raw_positions[(word_id, site_id)] = pts

        xs = [p[0] for pts in raw_positions.values() for p in pts]
        ys = [p[1] for pts in raw_positions.values() for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2

        target_width = 5.2
        target_height = 3.0
        # Place grid to the RIGHT of center (manim coords are centered at (0,0))
        target_center = (3.8, -0.2, 0)

        scale = min(
            target_width / (x_max - x_min + 1e-6),
            target_height / (y_max - y_min + 1e-6),
        )

        row_spacing = 1.12  # slightly increase spacing between rows

        def map_pos(pt):
            return (
                (pt[0] - x_mid) * scale + target_center[0],
                (pt[1] - y_mid) * scale * row_spacing + target_center[1],
                0,
            )

        grid = VGroup()
        grid.set_z_index(5)
        grid_positions = {}
        mapped_points = []
        for (word_id, site_id), pts in raw_positions.items():
            mapped = [map_pos(pt) for pt in pts]
            mapped_points.extend(mapped)
            for pos in mapped:
                # Empty traps: hollow circles
                ring = Circle(radius=0.055, color=GREY_B, stroke_width=1.2)
                ring.move_to(pos)
                ring.set_z_index(5)
                grid.add(ring)
            grid_positions[(word_id, site_id)] = mapped

        # Faint grid lines
        if mapped_points:
            xs_m = sorted({round(p[0], 3) for p in mapped_points})
            ys_m = sorted({round(p[1], 3) for p in mapped_points})
            x_lo = min(xs_m) - 0.2
            x_hi = max(xs_m) + 0.2
            y_lo = min(ys_m) - 0.2
            y_hi = max(ys_m) + 0.2
            grid_lines = VGroup()
            for x in xs_m:
                grid_lines.add(
                    Line((x, y_lo, 0), (x, y_hi, 0), color=GREY_B, stroke_width=1.0)
                    .set_opacity(0.15)
                    .set_z_index(2)
                )
            for y in ys_m:
                grid_lines.add(
                    Line((x_lo, y, 0), (x_hi, y, 0), color=GREY_B, stroke_width=1.0)
                    .set_opacity(0.15)
                    .set_z_index(2)
                )
            grid.add(grid_lines)

        # Site numbering (small font)
        site_labels = VGroup()
        for (word_id, site_id), mapped in grid_positions.items():
            label = Tex(f"{word_id},{site_id}").scale(0.25)
            label.set_color(GREY_B).set_opacity(0.6).set_z_index(6)
            label.move_to(mapped[0] + np.array([0.12, 0.10, 0]))
            site_labels.add(label)
        grid.add(site_labels)

        # Subtle chip boundary
        if mapped_points:
            x_lo = min(p[0] for p in mapped_points) - 0.35
            x_hi = max(p[0] for p in mapped_points) + 0.35
            y_lo = min(p[1] for p in mapped_points) - 0.35
            y_hi = max(p[1] for p in mapped_points) + 0.35
            chip = Rectangle(
                width=x_hi - x_lo,
                height=y_hi - y_lo,
                stroke_width=1.2,
                color=GREY_B,
            ).set_opacity(0.25)
            chip.move_to(((x_lo + x_hi) / 2, (y_lo + y_hi) / 2, 0))
            chip.set_z_index(1)
            grid.add(chip)

        # Faint coordinate axes in corner
        if mapped_points:
            x_lo = min(p[0] for p in mapped_points) - 0.25
            y_lo = min(p[1] for p in mapped_points) - 0.25
            axis_len = 0.5
            axes = VGroup(
                Line((x_lo, y_lo, 0), (x_lo + axis_len, y_lo, 0), color=GREY_B)
                .set_opacity(0.3)
                .set_z_index(3),
                Line((x_lo, y_lo, 0), (x_lo, y_lo + axis_len, 0), color=GREY_B)
                .set_opacity(0.3)
                .set_z_index(3),
            )
            grid.add(axes)


        grid_label = Tex(r"\textbf{Physical layout}").scale(0.55)
        grid_label.next_to(grid, UP, buff=0.3)
        grid_label.set_z_index(6)

        grid_group = VGroup(grid_label, grid)
        self.play(FadeIn(grid_group), run_time=0.6)

        mapping_title = Tex(r"\textbf{Gate} \rightarrow \textbf{Site}").scale(0.5)
        mapping_title.next_to(grid, DOWN, buff=0.5)
        mapping_title.set_z_index(6)
        self.play(FadeIn(mapping_title), run_time=0.3)

        # Draw initial logical->physical placement dots (occupied sites)
        colors = [ORANGE, TEAL, YELLOW]
        for i, addr in enumerate(layout):
            color = colors[i % len(colors)]
            key = (addr.word_id, addr.site_id)
            if key not in grid_positions:
                print("WARN: layout site not in grid_positions:", key)
                continue
            targets = grid_positions[key]
            for target in targets:
                placed = Circle(radius=0.07, color=color, stroke_width=0)
                placed.set_fill(color, opacity=0.95).set_z_index(10)
                placed.move_to(target)
                self.add(placed)

        # (D) ANIMATE LASERS USING gates_xy (this guarantees something happens if extraction worked)
        # If you want to slow down / speed up, tune these:
        on_time = 0.8
        off_time = 0.6

        native_gates = list(getattr(native_circuit, "gates", []))
        for gate_idx, gate_info in enumerate(gates_xy):
            # gate_info.sites is tuple[LocationAddress,...]
            targets = []
            for loc in gate_info.sites:
                key = (loc.word_id, loc.site_id)
                if key in grid_positions:
                    targets.extend(grid_positions[key])
                else:
                    print("WARN: gate site not in grid_positions:", key)

            if not targets:
                continue

            laser_pulses = [
                make_laser_cone(
                    target,
                    height=0.55,
                    base_width=0.35,
                    color=GREEN,
                    opacity=0.35,
                    glow_radius=0.06,
                )
                for target in targets
            ]
            atom_group = VGroup(*laser_pulses).set_z_index(50)

            gate_glow = None
            if gate_idx < len(native_gates):
                gate = native_gates[gate_idx]
                glow_scale = 1.6
                gate_glow = Rectangle(
                    width=gate.get_width() * glow_scale,
                    height=gate.get_height() * glow_scale,
                    color=GREEN,
                )
                gate_glow.set_fill(GREEN, opacity=0.22)
                gate_glow.set_stroke(GREEN, width=2).set_opacity(0.35)
                gate_glow.move_to(gate.get_center())
                gate_z = getattr(gate, "z_index", 0)
                gate_glow.set_z_index(gate_z + 1)

            if gate_glow is not None:
                self.play(FadeIn(atom_group), FadeIn(gate_glow), run_time=on_time)
            else:
                self.play(FadeIn(atom_group), run_time=on_time)
            self.wait(0.5)
            if gate_glow is not None:
                self.play(FadeOut(atom_group), FadeOut(gate_glow), run_time=off_time)
            else:
                self.play(FadeOut(atom_group), run_time=off_time)
            self.wait(0.5)

        # =========================
        # Layer 3: Qubit Routing
        # =========================
        all_objs = Group(*self.mobjects)
        self.play(FadeOut(all_objs), run_time=0.6)
        self.clear()

        routing_bg = BackgroundColor(BLACK)
        routing_title = Tex(r"\textbf{Qubit Routing: Moves happen in layers, then the gate fires.}").scale(0.8)
        routing_title.to_edge(UP)
        self.add(routing_bg)
        self.play(FadeIn(routing_title), run_time=0.6)

        routing_layout, gate_routes = compute_layout_and_gate_routes_from_native(
            native_mt
        )

        arch_spec = LogicalLayoutHeuristic().arch_spec
        raw_positions = {}
        for word_id, word in enumerate(arch_spec.words):
            for site_id in range(len(word.sites)):
                loc = LocationAddress(word_id, site_id)
                pts = arch_spec.get_positions(loc)
                if pts:
                    raw_positions[(word_id, site_id)] = pts

        xs = [p[0] for pts in raw_positions.values() for p in pts]
        ys = [p[1] for pts in raw_positions.values() for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2

        target_width = FRAME_WIDTH * 0.9
        target_height = FRAME_HEIGHT * 0.75
        target_center = (0.0, -0.2, 0)

        scale = min(
            target_width / (x_max - x_min + 1e-6),
            target_height / (y_max - y_min + 1e-6),
        )

        row_spacing = 1.05

        def map_pos_full(pt):
            return (
                (pt[0] - x_mid) * scale + target_center[0],
                (pt[1] - y_mid) * scale * row_spacing + target_center[1],
                0,
            )

        routing_grid = VGroup()
        routing_grid.set_z_index(5)
        routing_positions = {}
        mapped_points = []

        # Topology lines (bus lanes)
        topo = VGroup()
        topo.set_z_index(2)

        for bus in arch_spec.site_buses:
            for word_id in arch_spec.has_site_buses:
                for src_site, dst_site in zip(bus.src, bus.dst):
                    k0 = (word_id, src_site)
                    k1 = (word_id, dst_site)
                    if k0 in raw_positions and k1 in raw_positions:
                        p0 = map_pos_full(raw_positions[k0][0])
                        p1 = map_pos_full(raw_positions[k1][0])
                        topo.add(
                            Line(p0, p1, color=GREY_B, stroke_width=1.0)
                            .set_opacity(0.18)
                            .set_z_index(2)
                        )

        for bus in arch_spec.word_buses:
            for src_word, dst_word in zip(bus.src, bus.dst):
                for site_id in arch_spec.has_word_buses:
                    k0 = (src_word, site_id)
                    k1 = (dst_word, site_id)
                    if k0 in raw_positions and k1 in raw_positions:
                        p0 = map_pos_full(raw_positions[k0][0])
                        p1 = map_pos_full(raw_positions[k1][0])
                        topo.add(
                            Line(p0, p1, color=GREY_B, stroke_width=1.6)
                            .set_opacity(0.28)
                            .set_z_index(2)
                        )

        routing_grid.add(topo)

        for (word_id, site_id), pts in raw_positions.items():
            mapped = [map_pos_full(pt) for pt in pts]
            mapped_points.extend(mapped)
            for pos in mapped:
                ring = Circle(radius=0.055, color=GREY_B, stroke_width=1.2)
                ring.move_to(pos)
                ring.set_z_index(5)
                routing_grid.add(ring)
            routing_positions[(word_id, site_id)] = mapped

        routing_label = Tex(r"\textbf{Routing topology}").scale(0.6)
        routing_label.next_to(routing_grid, UP, buff=1)
        routing_label.set_z_index(6)
        routing_group = VGroup(routing_label, routing_grid)
        self.play(FadeIn(routing_group), run_time=0.6)

        def loc_to_point(loc: LocationAddress):
            return routing_positions[(loc.word_id, loc.site_id)][0]

        colors = [ORANGE, TEAL, YELLOW]
        current_layout = list(routing_layout)
        qubit_dots = []
        for i, addr in enumerate(current_layout):
            dot = Dot(loc_to_point(addr), radius=0.075, color=colors[i % len(colors)])
            dot.set_z_index(10)
            qubit_dots.append(dot)
            self.add(dot)

        move_time = 1.4
        layer_pause = 0.4
        gate_pause = 0.4
        fire_on_time = 1.2
        fire_off_time = 1.2
        label_fade_time = 0.3
        step_text = Tex("").scale(0.55)
        step_text.next_to(routing_title, DOWN, buff=0.35)
        step_text.set_z_index(6)
        self.add(step_text)

        for gate_idx, gate_info in enumerate(gate_routes, start=1):
            gate_label = Tex(
                rf"\textbf{{Gate {gate_idx}}}: {gate_info.op}"
            ).scale(0.55)
            gate_label.next_to(routing_title, DOWN, buff=0.35)
            gate_label.set_z_index(6)
            self.play(FadeOut(step_text), FadeIn(gate_label), run_time=label_fade_time)
            step_text = gate_label
            self.wait(gate_pause)

            for layer in gate_info.move_layers:
                lines = []
                anims = []
                moved = []
                for lane in layer:
                    src, dst = arch_spec.get_endpoints(lane)
                    try:
                        qid = current_layout.index(src)
                    except ValueError:
                        continue
                    bus_kind = "site-bus" if lane.move_type == MoveType.SITE else "word-bus"
                    lines.append(
                        rf"Move q{qid} \texttt{{({src.word_id},{src.site_id}) -> ({dst.word_id},{dst.site_id})}} on {bus_kind} {lane.bus_id}"
                    )
                    anims.append(qubit_dots[qid].animate.move_to(loc_to_point(dst)))
                    moved.append((qid, dst))

                if lines:
                    shown = lines[:3]
                    if len(lines) > 3:
                        shown.append(rf"\textbf{{+{len(lines) - 3} more}}")
                    layer_label = Tex(
                        r"\textbf{Move layer:}\\ " + r"\\ ".join(shown)
                    ).scale(0.5)
                    layer_label.next_to(step_text, DOWN, buff=0.2)
                    layer_label.set_z_index(6)
                    self.play(FadeIn(layer_label), run_time=label_fade_time)
                else:
                    layer_label = None

                if anims:
                    self.play(*anims, run_time=move_time)
                    for qid, dst in moved:
                        current_layout[qid] = dst
                self.wait(layer_pause)
                if layer_label is not None:
                    self.play(FadeOut(layer_label), run_time=label_fade_time)

            # Pulse laser at gate sites after routing
            targets = []
            for loc in gate_info.sites:
                key = (loc.word_id, loc.site_id)
                if key in routing_positions:
                    targets.append(routing_positions[key][0])

            if targets:
                laser_pulses = [
                    make_laser_cone(
                        target,
                        height=0.45,
                        base_width=0.28,
                        color=GREEN,
                        opacity=0.25,
                        glow_radius=0.05,
                    )
                    for target in targets
                ]
                atom_group = VGroup(*laser_pulses).set_z_index(50)
                fire_label = Tex(r"\textbf{Gate fires}").scale(0.5)
                fire_label.next_to(step_text, DOWN, buff=0.2)
                fire_label.set_z_index(6)
                self.play(FadeIn(atom_group), FadeIn(fire_label), run_time=fire_on_time)
                self.play(FadeOut(atom_group), FadeOut(fire_label), run_time=fire_off_time)
                self.wait(0.2)

            # Ensure final positions match the placement state
            if tuple(current_layout) != gate_info.layout_after:
                anims = []
                for qid, loc in enumerate(gate_info.layout_after):
                    if current_layout[qid] != loc:
                        anims.append(
                            qubit_dots[qid].animate.move_to(loc_to_point(loc))
                        )
                if anims:
                    self.play(*anims, run_time=0.2)
                current_layout = list(gate_info.layout_after)


if __name__ == "__main__":
    print("Run with: python3 -m manimlib demo/squin_qat_smoke.py SquinQATSmokeScene")
