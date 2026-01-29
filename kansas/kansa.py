"""KAN-SAs: Top-level accelerator and B-spline computation units."""

import numpy as np

import amaranth as am
from amaranth.lib import data
from amaranth.lib.wiring import Component, In, Out
from amaranth.utils import ceil_log2, exact_log2

from kansas.bspline import bspline_lut, compute_qconsts, grid_for_spline
from kansas.pe import PE
from kansas.sa import SystolicDelay


def clip(x, min, max): 
    return am.Mux(x < min, min, am.Mux(x > max, max, x))

class Kansa(Component):
    """KAN-SAs accelerator combining systolic array with B-spline units.

    Args:
        rows: Number of rows in the systolic array.
        cols: Number of columns in the systolic array.
        g: B-spline grid size.
        p: B-spline degree (must be odd).
        a_shape: Input activation shape.
        bsp_shape: B-spline basis value shape.
        b_shape: Weight shape.
        c_shape: Accumulator shape.
        depth: B-spline LUT depth.
    """

    def __init__(self, rows, cols, g, p,
        a_shape=am.unsigned(8), bsp_shape=am.signed(8), b_shape=am.signed(8), c_shape=am.signed(32), depth=256):
        assert a_shape.width == bsp_shape.width

        pe = PE(bsp_shape, b_shape, c_shape, simd_width=p + 1, buffer_width=g + p)
        self.sa = SystolicDelay(SystolicDelay(pe.tile(rows, cols), "a", start=1), "idx", start=1)

        self.bspunits = [BsplineUnit(g, p, a_shape, b_shape, depth) for _ in range(rows)]

        grid = grid_for_spline((0, 1), g, p)
        gqmin, gqmax = 0, (1 << a_shape.width) - 1
        s, z = compute_qconsts((grid[0], grid[-1]), (gqmin, gqmax))
        gridq = np.clip(np.round(grid / s) + z, min=gqmin, max=gqmax).astype(np.int32).tolist()
        self.knots = data.ArrayLayout(a_shape, g + 2 * p).const(gridq[:-1])

        super().__init__({
            "load":   In(cols),
            "b":      In(b_shape.width * (g + p) * cols),
            "dense":  In(1),
            "a":      In(data.ArrayLayout(a_shape, rows)),
            "bypass": In(data.ArrayLayout(data.ArrayLayout(am.signed(a_shape.width), p + 1), rows)),
            "cout":   Out(self.sa.c.shape())
        })
    def elaborate(self, _):
        m = am.Module()

        m.submodules.sa = self.sa
        m.d.comb += [
            self.sa.b.eq(self.b),
            self.sa.load.eq(self.load),
            self.cout.eq(self.sa.cout),
        ]

        for i in range(len(self.bspunits)):
            m.submodules[f"bspunit_{i}"] = self.bspunits[i]
            m.d.comb += [
                self.bspunits[i].x.eq(self.a[i]),
                self.bspunits[i].t.eq(self.knots),
                self.sa.a[i].eq(am.Mux(self.dense, self.bypass[i], self.bspunits[i].b)),
                self.sa.idx[i].eq(
                    am.Mux(self.dense, self.sa.idx[i].shape().const(range(len(self.sa.idx[i]))), self.bspunits[i].k)),
            ]
        return m

class BsplineUnit(Component):
    """Computes non-zero B-spline basis values and their indices for a given input.

    Args:
        g: Grid size.
        p: Spline degree.
        a_shape: Input shape.
        bsp_shape: Output B-spline value shape.
        depth: LUT depth.
    """

    def __init__(self, g, p, a_shape=am.unsigned(8), bsp_shape=am.signed(8), depth=256):
        assert p % 2 == 1, "current impl assumes p + 1 is pair"
        
        self.a_shape = a_shape
        self.bsp_shape = bsp_shape
        self.g, self.p, self.depth = g, p, depth

        num_knots = g + 2 * p # without last know of the grid
        self.interval_search = IntervalSearch(a_shape, num_knots)

        self.lut = BSPLUT(p, a_shape, bsp_shape, depth)

        self.q = am.Signal(exact_log2(depth))
        members = {
            "t": In(data.ArrayLayout(a_shape, num_knots)),
            "x": In(a_shape),
            "b": Out(data.ArrayLayout(bsp_shape, p + 1)),
            "k": Out(data.ArrayLayout(exact_log2(g + p), p + 1)),
        }
        super().__init__(members)

    def copy(self):
        return self.__class__(self.g, self.p, self.a_shape, self.bsp_shape, self.depth)

    def elaborate(self, _):
        m = am.Module()

        m.submodules.bsplut = self.lut

        m.submodules.interval_search = self.interval_search
        m.d.comb += [
            self.interval_search.t.eq(self.t),
            self.interval_search.x.eq(self.x),
        ]

        k = self.interval_search.y

        xdiff = clip(self.x - self.t[0], min=0, max=(1 << self.x.shape().width) - 1)
        m.d.comb += [
            self.q.eq(clip((self.g + self.p * 2) * xdiff - k * (self.depth - 1), min=0, max=self.depth - 1))
        ]

        unbounded_bnz = am.Signal(data.ArrayLayout(self.b.shape().elem_shape, self.p + 1))
        m.d.comb += [
            self.lut.addr.eq(self.q),
            unbounded_bnz.eq(self.lut.bsp),
        ]

        nonzerob = am.Signal(data.ArrayLayout(self.b.shape().elem_shape, self.p + 1))
        for i, b in enumerate(unbounded_bnz):
            if i == 0:
                cond = k > self.g + self.p - 1
            elif i == self.p:
                cond = k < self.p
            else:
                cond = (k > self.g + self.p + i - 1) | (k < i)
            m.d.comb += [
                nonzerob[i].eq(am.Mux(cond, 0, b)),
                self.k[i].eq(am.Mux(cond, 0, k - i)),
            ]

        m.d.comb += self.b.eq(nonzerob)
        return m

class BSPLUT(Component):
    """B-spline lookup table exploiting symmetry to halve storage.

    Args:
        p: Spline degree.
        a_shape: Input shape.
        bsp_shape: Output basis value shape.
        depth: LUT depth (number of entries).
    """

    def __init__(self, p, a_shape=am.unsigned(8), bsp_shape=am.signed(8), depth=256):
        self.a_shape, self.bsp_shape = a_shape, bsp_shape
        self.p, self.depth = p, depth

        self.qbits = exact_log2(depth)
        bqmax = (1 << bsp_shape.width - 1) - 1 if bsp_shape.signed else (1 << bsp_shape.width) - 1

        self.bsp_table, *_ = bspline_lut(p, self.qbits, bqmax=bqmax)
        self.lut = data.ArrayLayout(data.ArrayLayout(bsp_shape, self.bsp_table.shape[1]), depth).const([
            vals.tolist() for vals in self.bsp_table])

        members = {
            "addr": In(self.qbits),
            "bsp": Out(data.ArrayLayout(bsp_shape, self.p + 1)),
        }
        super().__init__(members)

    def elaborate(self, _):
        m = am.Module()
        m.d.comb += self.bsp.eq(am.Cat(self.lut[self.addr], self.lut[(self.depth - 1 - self.addr).as_unsigned()][::-1]))
        return m

    def copy(self):
        return self.__class__(self.p, self.a_shape, self.bsp_shape, self.depth)

class IntervalSearch(Component):
    """Parallel comparator to find which grid interval contains the input.

    Args:
        width: Bit width of input and thresholds.
        num_thresholds: Number of grid thresholds to compare against.
    """

    def __init__(self, width, num_thresholds):
        self.cmps = am.Signal(num_thresholds)
        super().__init__({
            "t": In(data.ArrayLayout(width, num_thresholds)),
            "x": In(width),
            "y": In(ceil_log2(num_thresholds))
        })

    def elaborate(self, _):
        m = am.Module()
        m.d.comb += [self.cmps[i].eq(self.x >= t) for i, t in enumerate(self.t)]
        for i in range(len(self.cmps)):
            with m.If(self.cmps[i]):
                m.d.comb += self.y.eq(i)
        return m

if __name__ == '__main__':
    from amaranth.back import verilog
    with open("kansa.v", 'w') as f:
        f.write(verilog.convert(Kansa(2, 2, 5, 3)))
