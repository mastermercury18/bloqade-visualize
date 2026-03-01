from manim import Scene

from bloqade.lanes.visualize_squin import build_squin_circuit_mobject
from bloqade import squin, qubit

@squin.kernel
def demo(reg: qubit.Qubit):
    squin.sqrt_x(reg)
    squin.t(reg)

class SquinCircuitScene(Scene):
    def construct(self):
        self.add(build_squin_circuit_mobject(demo))
