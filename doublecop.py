from theano import Apply
from theano.gof import COp
from theano.tensor import as_tensor_variable

class DoubleCOp(COp):
    __props__ = ()

    def __init__(self):
        COp.__init__(self, ["doublecop.c"],
                     "APPLY_SPECIFIC(doublecop)")

    def make_node(self, x):
        x = as_tensor_variable(x)
        if x.ndim != 1:
            raise TypeError("DoubleCOp only works with 1D")
        return Apply(self, [x], [x.type()])

    def infer_shape(self, input_shapes):
        return input_shapes

    def grad(self, inputs, g):
        return [g[0] * 2]
