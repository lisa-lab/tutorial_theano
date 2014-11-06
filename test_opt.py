import theano

from scalmulop import ScalMulV1
from doubleop import DoubleOp
import opt

def test_scalmul_double():
    x = theano.tensor.matrix()
    y = ScalMulV1(2)(x)
    f = theano.function([x], y)

    assert not any(isinstance(n.op, ScalMulV1)
                   for n in f.maker.fgraph.toposort())
    assert any(isinstance(n.op, DoubleOp)
               for n in f.maker.fgraph.toposort())

