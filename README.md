# KAN-SAs

Efficient Acceleration of Kolmogorov-Arnold Networks on Systolic Arrays.

Hardware implementation in [Amaranth HDL](https://amaranth-lang.org/).

## Installation

```bash
pip install -e .
```

## Usage

### Generate Verilog

```python
from amaranth.back import verilog
from kansas.kansa import Kansa

# 2x2 systolic array, grid_size=5, spline_degree=3
accelerator = Kansa(rows=2, cols=2, g=5, p=3)

with open("kansa.v", "w") as f:
    f.write(verilog.convert(accelerator))
```

Or from command line:

```bash
python -m kansas.kansa # generates kansa.v
python -m kansas.sa    # generates baseline systolic array sa.v
```

### Run tests

soon™

## Architecture

- `Kansa` - Top-level accelerator
- `BsplineUnit` - B-spline basis function computation
- `PE` - Weight-stationary processing element
- `Tile` - 2D array of PEs
- `SystolicDelay` - Systolic timing for data flow

## Citation

```bibtex
@INPROCEEDINGS{KAN-SAs,
  author={Sohaib Errabii and Olivier Sentieys and Marcello Traiola},
  booktitle={2026 Design, Automation & Test in Europe Conference (DATE)},
  title={KAN-SAs: Efficient Acceleration of Kolmogorov-Arnold Networks on Systolic Arrays},
  year={2026},
  url={https://arxiv.org/abs/2512.00055},
}
```
