from theano import Op, Apply
from theano.tensor import as_tensor_variable
from theano.scalar import as_scalar_variable

class ScalMul(Op):
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

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def grad(self, inputs, output_grads):
        return [output_grads[0] * self.scal]
