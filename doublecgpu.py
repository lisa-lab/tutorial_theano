from theano import Apply
from theano.gpuarray.basic_ops import (as_gpuarray_variable,
                                       infer_context_name, CGpuKernelBase)


class DoubleCGpu(CGpuKernelBase):
    __props__ = ()

    def __init__(self):
        CGpuKernelBase.__init__(self, ["doublecgpu.c"],
                                "double_fn")

    def make_node(self, x):
        ctx_name = infer_context_name(x)
        x = as_gpuarray_variable(x, ctx_name)
        return Apply(self, [x], [x.type()])

    def get_params(self, node):
        return node.outputs[0].type.context

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def grad(self, inputs, output_grads):
        return [output_grads[0] * 2]
