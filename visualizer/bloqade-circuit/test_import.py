"""Test script to verify circuit visualization imports work."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from bloqade.visual.circuit import circuit_drawer
    print("✓ Successfully imported circuit_drawer!")
    print(f"  Location: {circuit_drawer.__module__}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nTo fix this, you need to install the package:")
    print("  uv pip install -e .")
    print("\nOr if you have Python 3.14, you may need to use Python 3.13 or earlier.")
    sys.exit(1)
