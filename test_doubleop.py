import numpy

from theano import function, config
from theano.tensor import matrix
from theano.tests import unittest_tools as utt
from theano.tests.test_rop import RopLop_checker

from doubleop import DoubleOp


def test_doubleop():
    utt.seed_rng()
    x = matrix()
    f = function([x], DoubleOp()(x))
    inp = numpy.asarray(numpy.random.rand(5, 4),
                        dtype=config.floatX)
    out = f(inp)
    utt.assert_allclose(inp * 2, out)


class test_Double(utt.InferShapeTester):
    def test_infer_shape(self):
        utt.seed_rng()
        x = matrix()
        self._compile_and_check(
            # function inputs (symbolic)
            [x],
            # Op instance
            [DoubleOp()(x)],
            # numeric input
            [numpy.asarray(numpy.random.rand(5, 4),
                           dtype=config.floatX)],
            # Op class that should disappear
            DoubleOp)


def test_doubleop_grad():
    utt.seed_rng()
    utt.verify_grad(
        # Op instance
        DoubleOp(),
        # Numeric inputs
        [numpy.random.rand(5, 7, 2)]
        )
