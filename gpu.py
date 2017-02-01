from theano import Op
from theano.gpuarray.type import gpu_context_type

class GpuOp(Op):
    __props__ = ()
    params_type = gpu_context_type

    def make_node(self, ...):
        # return apply node

    def get_params(self, node):
        return node.outputs[0].type.context

    def perform(self, node, inputs, output_storage):
        # python code

    def c_code(self, node, name, input_names,
               output_names, sub):
        # return C code string

