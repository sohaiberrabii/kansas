"""Processing Element and Tile components for systolic arrays."""

import amaranth as am
from amaranth.lib import data
from amaranth.lib.wiring import Component, In, Out
from amaranth.utils import exact_log2


class PE(Component):
    """Weight-stationary Processing Element with SIMD support.

    Args:
        a_shape: Input activation shape.
        b_shape: Weight shape.
        c_shape: Accumulator shape.
        simd_width: Number of parallel multiply-accumulate operations.
        buffer_width: Weight buffer width for indexed access.
    """
    def __init__(self, a_shape, b_shape, c_shape, simd_width=1, buffer_width=1):
        assert simd_width <= buffer_width
        self.a_shape, self.b_shape, self.c_shape = a_shape, b_shape, c_shape
        self.buffer_width = buffer_width
        self.simd_width = simd_width
        members = {
            "a":    In(data.ArrayLayout(a_shape, simd_width)),
            "b":    In(data.ArrayLayout(b_shape, buffer_width)),
            "c":    In(c_shape),
            "load": In(1),
            "aout": Out(data.ArrayLayout(a_shape, simd_width)),
            "bout": Out(data.ArrayLayout(b_shape, buffer_width)),
            "cout": Out(c_shape),
        }
        if simd_width < buffer_width:
            members.update({
                "idx":    In(data.ArrayLayout(exact_log2(buffer_width), simd_width)),
                "idxout": Out(data.ArrayLayout(exact_log2(buffer_width), simd_width))
            })
        super().__init__(members)

    def elaborate(self, _):
        m = am.Module()

        with m.If(self.load):
            m.d.sync += self.bout.eq(self.b)

        mul_b = am.Signal(data.ArrayLayout(self.b.shape().elem_shape, self.a.shape().length))
        m.d.comb += [
            self.aout.eq(self.a),
            mul_b.eq(am.Cat(self.bout[i] for i in self.idx) if hasattr(self, "idx") else self.bout),
        ]

        if hasattr(self, "idx"):
            m.d.comb += self.idxout.eq(self.idx)

        m.d.sync += self.cout.eq(self.c + sum(a * b for a, b in zip(self.a, mul_b)))
        return m

    def copy(self):
        return self.__class__(
            self.a_shape, self.b_shape, self.c_shape, simd_width=self.simd_width, buffer_width=self.buffer_width)

    def tile(self, rows, cols):
        h_conn_ports = [("a", "aout")]
        v_gather_ports = ["a", "aout"]
        if self.simd_width < self.buffer_width:
            h_conn_ports += [("idx", "idxout")]
            v_gather_ports += ["idx", "idxout"]

        return Tile(Tile(self, cols, h_conn_ports, ["b", "bout", "c", "cout", "load"], ["load"]),
            rows, [("b", "bout"), ("c", "cout")], v_gather_ports, ["load"])

    @property
    def latency(self):
        """latency for valid cout given preloaded b and valid input a"""
        return 1

class Tile(Component):
    """Replicates a component n times with configurable port connectivity.

    Args:
        element: Component to replicate.
        n: Number of replicas.
        conn_ports: Ports to chain between adjacent elements (in, out) pairs.
        gather_ports: Ports to expose as arrays.
        broadcast_ports: Ports to fan out to all elements.
    """

    def __init__(self, element, n, conn_ports, gather_ports, broadcast_ports):
        assert n > 0
        assert all(element.signature.members[p].flow == In for p in broadcast_ports)

        self.conn_ports, self.gather_ports, self.broadcast_ports = conn_ports, gather_ports, broadcast_ports
        self.elements = [element] + [element.copy() for _ in range(n - 1)]
        super().__init__({
            k: v.flow(data.ArrayLayout(v.shape, n) if k in gather_ports else v.shape)
            for k, v in element.signature.members.items()
        })

    def elaborate(self, _):
        m = am.Module()
        m.submodules += self.elements

        # interconnect
        for pin, pout in self.conn_ports:
            m.d.comb += [getattr(eout, pin).eq(getattr(ein, pout)) for ein, eout in zip(self.elements[:-1], self.elements[1:])]
            m.d.comb += [
                getattr(self.elements[0], pin).eq(getattr(self, pin)), getattr(self, pout).eq(getattr(self.elements[-1], pout))]

        # gather
        for p in self.gather_ports:
            packed = am.Cat(getattr(e, p) for e in self.elements)
            m.d.comb += packed.eq(getattr(self, p)) if self.signature.members[p].flow == In else getattr(self, p).eq(packed)

        # broadcast
        m.d.comb += [getattr(e, p).eq(getattr(self, p)) for p in self.broadcast_ports for e in self.elements]
        return m

    def copy(self):
        return Tile(self.elements[0].copy(), len(self.elements), self.conn_ports, self.gather_ports, self.broadcast_ports)
    
    @property
    def latency(self):
        return self.elements[0].latency * len(self.elements)
