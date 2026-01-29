"""Systolic array utilities for data flow timing."""

import amaranth as am
from amaranth.lib import data
from amaranth.lib.wiring import Component, In, Out

from kansas.pe import PE


class Shifter(Component):
    """Shift register with configurable depth.

    Args:
        shape: Data shape for each register stage.
        length: Number of pipeline stages (0 = passthrough).
    """
    def __init__(self, shape, length):
        assert length >= 0
        self.shape, self.length = shape, length
        self.regs = am.Signal(data.ArrayLayout(shape, length))
        super().__init__({"en": In(1, init=1), "d": In(shape), "q": Out(shape)})

    def elaborate(self, _):
        m = am.Module()
        if self.length == 0:
            m.d.comb += self.q.eq(self.d)
        else:
            m.d.comb += self.q.eq(self.regs[-1])
            with m.If(self.en):
                m.d.sync += self.regs.eq(am.Cat(self.d, self.regs[:-1]))
        return m

class SystolicDelay(Component):
    """Wraps a component adding staggered delays to an array port for systolic timing.

    Args:
        component: Component to wrap.
        port: Name of the array port to add delays to.
        start: Delay for the first element (element i gets start+i cycles delay).
    """

    def __init__(self, component, port, start=0):
        assert start >= 0
        assert component.signature.members[port].flow == In
        port_shape = component.signature.members[port].shape
        assert isinstance(port_shape, data.ArrayLayout)
        self.port = port
        self.component = component
        self.shifters = [Shifter(port_shape.elem_shape, start + i) for i in range(port_shape.length)]
        super().__init__(component.signature)

    def elaborate(self, _):
        m = am.Module()
        m.submodules += [self.component] + self.shifters

        # add shift registers
        for i, shifter in enumerate(self.shifters):
            m.d.comb += [shifter.d.eq(getattr(self, self.port)[i]), getattr(self.component, self.port)[i].eq(shifter.q)]

        # connect other ports
        m.d.comb += [
            getattr(self.component, k).eq(getattr(self, k)) if v.flow == In else getattr(self, k).eq(getattr(self.component, k))
            for k, v in self.signature.members.items() if k != self.port]

        return m

    @property
    def latency(self):
        return self.shifters[-1].length + self.component.latency

if __name__ == '__main__':
    from amaranth.back import verilog
    with open("sa.v", 'w') as f:
        f.write(verilog.convert(SystolicDelay(PE(8, 8, 32).tile(2, 2), "a")))
