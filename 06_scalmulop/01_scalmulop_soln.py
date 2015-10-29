from theano import Op, Apply
from theano.tensor import as_tensor_variable
from theano.scalar import as_scalar

class ScalMulV1(Op):
    __props__ = ('scal',)

    def __init__(self, scal):
        if not isinstance(scal, int):
            raise TypeError('expected an int')
        self.scal = scal

    def make_node(self, x):
        x = as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = x * self.scal


class ScalMulV2(Op):
    __props__ = ()

    def make_node(self, x, scal):
        x = as_tensor_variable(x)
        scal = as_scalar(scal)
        return Apply(self, [x, scal], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        scal = inputs[1]
        z = output_storage[0]
        z[0] = x * scal
