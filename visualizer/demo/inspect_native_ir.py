from bloqade import qubit, squin
from bloqade.gemini.rewrite.initialize import __RewriteU3ToInitialize
from bloqade.native.upstream import SquinToNative
from bloqade.rewrite.passes import CallGraphPass
from bloqade.squin.rewrite.non_clifford_to_U3 import RewriteNonCliffordToU3
from kirin import rewrite

from bloqade.lanes.visualize_squin import _extract_gate_ops


@squin.kernel(typeinfer=True, fold=True)
def demo_native():
    q = qubit.qalloc(3)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    squin.cx(q[0], q[2])


def emit_native(mt):
    rule = rewrite.Chain(
        rewrite.Walk(RewriteNonCliffordToU3()),
        rewrite.Walk(__RewriteU3ToInitialize()),
    )
    CallGraphPass(mt.dialects, rule)(mt)
    return SquinToNative().emit(mt)


def main():
    native_mt = emit_native(demo_native)
    print("=== Native IR ===")
    native_mt.print()
    ops = _extract_gate_ops(native_mt)
    print("=== Visualizer gate ops ===")
    for op in ops:
        qubits = [str(q) for q in op.qubits]
        controls = [str(q) for q in op.controls]
        targets = [str(q) for q in op.targets]
        print(f"{op.name} qubits={qubits} controls={controls} targets={targets}")


if __name__ == "__main__":
    main()
