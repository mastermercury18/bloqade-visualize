# Bloqade Visual Compiler 

A **ManimGL** project that animates the full compilation pipeline for a Bloqade squin circuit: from logical gates through native placement to qubit routing on a physical topology.

![Layer 2 Visualization](bloqade-visualizer-image.png)

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
pip install bloqade
pip install bloqade.lanes
pip install bloqade.circuit
pip install quantum-animation-toolbox
python3 -m manimlib demo/squin_qat_smoke.py SquinQATSmokeScene
```

## Changing the circuit

Edit the two kernels at the top of `squin_qat_smoke.py`:

- **`demo_logical`** — used for the first (logical) layer. Keep the same number of qubits as in `demo_native` if you want layout/placement to match.
- **`demo_native`** — must use `@squin.kernel(typeinfer=True, fold=True)` and `qubit.qalloc(n)`; this is what gets compiled to native and used for placement and routing.

Then run the same command again to re-render the three layers.

