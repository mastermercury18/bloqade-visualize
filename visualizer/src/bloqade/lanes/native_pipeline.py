from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

from kirin import ir, rewrite

from bloqade.analysis import address
from bloqade.gemini.rewrite.initialize import __RewriteU3ToInitialize
from bloqade.lanes.analysis import layout as layout_analysis
from bloqade.lanes.analysis import placement as placement_analysis
from bloqade.lanes.dialects import place
from bloqade.lanes.heuristics.fixed import (
    LogicalLayoutHeuristic,
    LogicalPlacementStrategy,
)
from bloqade.lanes.upstream import NativeToPlace
from bloqade.native.upstream import SquinToNative
from bloqade.rewrite.passes import CallGraphPass
from bloqade.squin.rewrite.non_clifford_to_U3 import RewriteNonCliffordToU3
from bloqade.lanes.layout.encoding import LocationAddress


def emit_native(mt: ir.Method) -> ir.Method:
    """Lower a squin method to Bloqade native IR."""
    rule = rewrite.Chain(
        rewrite.Walk(RewriteNonCliffordToU3()),
        rewrite.Walk(__RewriteU3ToInitialize()),
    )
    CallGraphPass(mt.dialects, rule)(mt)
    return SquinToNative().emit(mt)


def emit_place(mt: ir.Method) -> ir.Method:
    """Lower a squin method to place dialect (native -> place)."""
    return NativeToPlace().emit(emit_native(mt))


@dataclass(frozen=True)
class GateSiteXY:
    """One placed/native gate instance with physical site + XY coordinates."""
    op: str                          # e.g. "R", "Rz", "CZ", "EndMeasure"
    logical_qubits: tuple[int, ...]  # logical/address indices used by this op
    sites: tuple[LocationAddress, ...]
    xy: tuple[tuple[float, float], ...]  # one (x,y) per site, same order as sites


def compute_layout_and_gate_sites_xy(
    mt: ir.Method,
    *,
    layout_heuristic: layout_analysis.LayoutHeuristicABC | None = None,
    placement_strategy: placement_analysis.PlacementStrategyABC | None = None,
    include_ops: Iterable[type[ir.Statement]] = (place.R, place.Rz, place.CZ, place.EndMeasure),
) -> tuple[tuple[LocationAddress, ...], list[GateSiteXY]]:
    """
    Returns:
      initial_layout: initial mapping from logical addresses -> LocationAddress
      gates_xy: list of placed ops with physical LocationAddress sites and raw (x,y) coords
    """
    place_mt = emit_place(mt)

    # Address space: how many qubit addresses exist in the placed program
    address_analysis = address.AddressAnalysis(place_mt.dialects)
    address_frame, _ = address_analysis.run_no_raise(place_mt)
    all_qubits = tuple(range(address_analysis.next_address))

    if layout_heuristic is None:
        layout_heuristic = LogicalLayoutHeuristic()
    if placement_strategy is None:
        placement_strategy = LogicalPlacementStrategy()

    # Initial layout (logical addr -> LocationAddress)
    initial_layout = layout_analysis.LayoutAnalysis(
        place_mt.dialects,
        layout_heuristic,
        address_frame.entries,
        all_qubits,
    ).get_layout_no_raise(place_mt)

    # Placement analysis gives us ConcreteState.layout snapshots keyed by op.state_after
    placement = placement_analysis.PlacementAnalysis(
        place_mt.dialects,
        initial_layout,
        address_frame.entries,
        placement_strategy,
    )
    placement_frame, _ = placement.run_no_raise(place_mt)

    include_ops = tuple(include_ops)
    arch_spec = layout_heuristic.arch_spec  # <- provides physical site coordinates

    def loc_to_xy(loc: LocationAddress) -> tuple[float, float]:
        """
        Convert a LocationAddress to a raw architecture (x,y).
        arch_spec.get_positions(loc) returns a list of points; we take the first.
        """
        pts = arch_spec.get_positions(loc)
        if not pts:
            # Shouldn't happen for valid sites; fail loudly to catch bad addresses early.
            raise RuntimeError(f"No arch_spec position for site {loc}")
        x, y = pts[0][0], pts[0][1]
        return float(x), float(y)

    gates_xy: list[GateSiteXY] = []

    # IMPORTANT: walk ALL StaticPlacement blocks; don't break after the first one
    for stmt in place_mt.callable_region.walk():
        if not isinstance(stmt, place.StaticPlacement):
            continue

        for blk in stmt.body.blocks:
            for inner in blk.stmts:
                if not isinstance(inner, include_ops):
                    continue

                state_after = placement_frame.entries.get(inner.state_after)
                if not isinstance(state_after, placement_analysis.ConcreteState):
                    # If placement isn't concrete here, we can't reliably extract sites.
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
