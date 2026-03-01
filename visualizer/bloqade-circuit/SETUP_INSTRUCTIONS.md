# Setup Instructions for Circuit Visualization

## The Problem
You're getting `ModuleNotFoundError: No module named 'bloqade.visual.circuit'` because the package and its dependencies aren't installed.

## Solution

### Option 1: Install in Development Mode (Recommended)

1. **Create/activate a virtual environment with Python 3.13 or earlier:**
   ```bash
   # Using uv (recommended)
   uv venv --python 3.13
   source .venv/bin/activate
   
   # Or using Python directly
   python3.13 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the package:**
   ```bash
   uv pip install -e .
   # OR
   pip install -e .
   ```

3. **Run the example:**
   ```bash
   python example_circuit_viz.py
   ```

### Option 2: Use System Python (if already installed)

If you already have the bloqade-circuit package installed system-wide:

```bash
python3 example_circuit_viz.py
```

### Option 3: Quick Test Without Full Installation

If you just want to test the imports work, you can manually install the minimal dependencies:

```bash
pip install kirin-toolchain cirq
```

Then run:
```bash
python3 -c "import sys; sys.path.insert(0, 'src'); from bloqade.visual.circuit import circuit_drawer; print('Success!')"
```

## Verify Installation

Run this to verify:
```bash
python -c "from bloqade.visual.circuit import circuit_drawer; print('✓ Import successful!')"
```

## Note on Python 3.14

Python 3.14 is too new for some dependencies (like pydantic-core). Use Python 3.13 or earlier.
