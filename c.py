from theano import Op

class MyOp(Op):
    __props__ = ()

    def make_node(self, ...):
        # return apply node

    def c_code(self, node, name, input_names,
               output_names, sub):
        # return C code string

    def c_support_code(self):
        # return C code string

    def c_code_cache_version(self):
        # return hashable object
