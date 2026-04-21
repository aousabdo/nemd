# N-EMD: Neural Empirical Mode Decomposition

A differentiable, physics-constrained signal decomposition method that replaces EMD's hand-crafted iterative sifting process with a learned neural operator trained end-to-end.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from nemd.utils import generate_synthetic_signal
from nemd.classical import ClassicalEMD

# Generate a synthetic multi-component signal
t, signal, components = generate_synthetic_signal(
    n_samples=1024,
    components=[
        {"type": "am-fm", "f0": 50.0, "f_mod": 2.0, "a_mod": 0.5},
        {"type": "am-fm", "f0": 15.0, "f_mod": 0.5, "a_mod": 0.3},
        {"type": "am-fm", "f0": 3.0,  "f_mod": 0.1, "a_mod": 0.2},
    ],
    noise_std=0.05,
    seed=42,
)

# Decompose with classical EMD
emd = ClassicalEMD()
imfs = emd.decompose(signal)
```

## Project Structure

```
nemd/
  model.py       - Core N-EMD architecture
  losses.py      - Physics-constrained loss functions
  layers.py      - Differentiable signal processing layers
  classical.py   - Classical EMD/EEMD/VMD baselines
  utils.py       - Signal generation, metrics, helpers
  train.py       - Training loop
```

## License

MIT
