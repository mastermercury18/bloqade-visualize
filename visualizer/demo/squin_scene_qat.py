from manimlib import Scene

from bloqade import qubit, squin
from bloqade.lanes.visualize_squin import build_squin_circuit_qat


@squin.kernel
def demo(reg: qubit.Qubit):
    squin.sqrt_x(reg)
    squin.t(reg)


class SquinCircuitQATScene(Scene):
    def construct(self):
        self.add(
            build_squin_circuit_qat(
                demo, use_qat_defaults=True, qat_style="quera", qat_format="demo"
            )
        )
