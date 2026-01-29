"""B-spline utilities for quantized lookup table generation."""

import numpy as np


def compute_qconsts(irange, orange):
    """Compute quantization scale and zero-point for mapping irange to orange."""
    s = float((irange[1] - irange[0]) / (orange[1] - orange[0]))
    z = max(min(orange[0] - round(irange[0] / s), orange[1]), orange[0])
    return s, z

def grid_for_spline(grid_range, grid_size, spline_degree):
    """Generate extended grid points for B-spline evaluation."""
    h = (grid_range[1] - grid_range[0]) / grid_size
    return np.linspace(grid_range[0] - spline_degree * h, grid_range[1] + spline_degree * h, grid_size + 2 * spline_degree + 1)

def bsplines(x, grid, spline_degree):
    """Compute B-spline basis functions using Cox-de Boor recursion."""
    x = x[..., None]
    bases = (x >= grid[:-1]) & (x < grid[1:])
    for k in range(1, spline_degree + 1):
        bases = ((x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)]) * bases[..., :-1]) + (
        (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:-k]) * bases[..., 1:])
    return bases

def bspline_lut(p, qbits, bqmax=127):
    """Generate quantized B-spline lookup table exploiting symmetry."""
    assert p > 0 and p % 2 == 1, "Current impl assumes p is odd"
    depth = 1 << qbits
    width = (p + 1) // 2
    samples = depth * width

    # eval points selection: interval centers except boundaries
    delta = 1 / depth
    x = [0] + [3 * delta / 2 + k * delta for k in range(samples - 2)] + [(p + 1) / 2]
    bsp = bsplines(np.array(x), np.arange(p + 2), p)

    s, z = compute_qconsts((0, bsp.max()), (0, bqmax))
    bspq = np.round(1 / s * bsp + z, decimals=0).astype(np.int8) 
    bspq = bspq.squeeze().reshape(width, depth).transpose(1, 0)
    return bspq, s, z
