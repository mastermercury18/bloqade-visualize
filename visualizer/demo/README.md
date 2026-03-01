# bloqade-visualize

A **ManimGL** scene that animates the full compilation pipeline for a squin circuit: from logical gates through native placement to qubit routing on a physical topology.

## What it does

`squin_qat_smoke.py` defines a single scene, **`SquinQATSmokeScene`**, that runs in three layers:

1. **Logical** — The high-level circuit (H, CX, etc.) built with `build_squin_circuit_qat` and shown in QuEra-style QAT visuals.
2. **Native + Placement** — The same circuit is compiled to native ops (`SquinToNative` + `AggressiveUnroll`), placed via `LogicalLayoutHeuristic` and `LogicalPlacementStrategy`, and visualized as:
   - A decomposed circuit diagram (left),
   - A physical layout grid with site indices,
   - Initial qubit placement (colored dots),
   - Per-gate “laser” pulses on the corresponding physical sites.
3. **Qubit routing** — The routing topology (site and word buses) is drawn; for each gate, move layers are shown (qubits moving along buses), then “Gate fires” with laser pulses at the gate sites.

So you see: **logical circuit → native circuit + placement + gate-to-site mapping → routing moves + gate execution**.

## Requirements

- **ManimGL** (`manimlib`)
- **bloqade** (and its lanes/analysis/native/placement stack)
- **quantum-animation-toolbox** (for `BackgroundColor`, `BLACK`, and QAT circuit building)
- Optional: local **bloqade-circuit** at `bloqade-circuit/src` for squin→native decomposition (see `_CIRCUIT_SRC` in the script)

## How to run

From the repo root:

```bash
python3 -m manimlib demo/squin_qat_smoke.py SquinQATSmokeScene
```

This opens the ManimGL window and plays the three-layer animation. The circuit used is the built-in one: 3 qubits, `H(q0)`, `CX(q0,q1)`, `CX(q0,q2)` (see `demo_logical` / `demo_native` in the script).

## Circuit and pipeline details

- **Logical** uses a squin kernel with explicit qubit args (`demo_logical`) for the first view.
- **Native** uses a kernel with `qubit.qalloc(3)` and `@squin.kernel(typeinfer=True, fold=True)` (`demo_native`). That kernel is passed to `SquinToNative().emit()`, then `AggressiveUnroll` is applied; the result drives both the native circuit drawing and the placement/routing analysis.
- **Layout and placement** come from `compute_layout_and_gate_sites_xy_from_native()` and `compute_layout_and_gate_routes_from_native()` (layout heuristic and placement strategy are the fixed ones from `bloqade.lanes.heuristics.fixed`).
- **Physical topology** (site positions, site buses, word buses) is given by `LogicalLayoutHeuristic().arch_spec`; the scene draws rings at each site, optional grid lines, and bus lines for the routing layer.

## Main helpers in the script

| Name | Purpose |
|------|--------|
| `make_laser_cone(...)` | Green cone + dot used for “laser” pulses at gate sites. |
| `compute_layout_and_gate_sites_xy_from_native(native_mt, ...)` | Returns `(initial_layout, list of GateSiteXY)` for the placed native method. |
| `compute_layout_and_gate_routes_from_native(native_mt, ...)` | Returns `(initial_layout, list of GateRoute)` with move layers and layout-after per gate. |
| `GateRoute` | Dataclass: `op`, `logical_qubits`, `sites`, `xy`, `move_layers`, `layout_after`. |
| `_split_group(group)` | In-scene helper: splits a possible (background, circuit) group from `build_squin_circuit_qat`. |

## Changing the circuit

Edit the two kernels at the top of `squin_qat_smoke.py`:

- **`demo_logical`** — used for the first (logical) layer. Keep the same number of qubits as in `demo_native` if you want layout/placement to match.
- **`demo_native`** — must use `@squin.kernel(typeinfer=True, fold=True)` and `qubit.qalloc(n)`; this is what gets compiled to native and used for placement and routing.

Then run the same command again to re-render the three layers.

## Other demo files

- **`squin_scene.py`** — ManimCE scene that only draws the logical circuit (no placement/routing).
- **`squin_scene_qat.py`** — ManimGL scene that draws a single squin circuit with QAT styling (`build_squin_circuit_qat`).
