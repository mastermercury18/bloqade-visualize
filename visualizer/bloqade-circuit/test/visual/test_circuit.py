"""Tests for circuit visualization."""

from bloqade import squin
from bloqade.visual.circuit import circuit_drawer


def test_text_drawer():
    """Test text-based circuit drawer."""
    @squin.kernel
    def main():
        q = squin.qalloc(2)
        squin.h(q[0])
        squin.cx(q[0], q[1])

    result = circuit_drawer(main, output="text")
    assert result is not None
    circuit_str = str(result)
    assert "q0" in circuit_str
    assert "q1" in circuit_str
    # Should contain gate names
    assert "H" in circuit_str or "h" in circuit_str
    assert "CX" in circuit_str or "cx" in circuit_str


def test_matplotlib_drawer():
    """Test matplotlib-based circuit drawer."""
    try:
        import matplotlib
    except ImportError:
        # Skip if matplotlib not available
        return

    @squin.kernel
    def main():
        q = squin.qalloc(2)
        squin.h(q[0])
        squin.cx(q[0], q[1])

    result = circuit_drawer(main, output="mpl")
    assert result is not None
