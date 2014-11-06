from theano import Op

class MyOp(Op):
    __props__ = ()

    def __init__(self, ...):
        # set up parameters

    def make_node(self, ...):
        # create apply node

    def make_thunk(self, node, storage_map,
                   compute_map, no_recycling):
        # return a thunk

    def infer_shape(self, input_shapes):
        # return output shapes

    def grad(self, inputs, output_grads):
        # return gradient graph for each input
