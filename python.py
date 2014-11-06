from theano import Op

class MyOp(Op):
    __props__ = ()

    def __init__(self, ...):
        # set up parameters

    def make_node(self, ...):
        # create apply node

    def perform(self, node, inputs, outputs_storage):
        # do the computation

    def infer_shape(self, input_shapes):
        # return output shapes

    def grad(self, inputs, output_grads):
        # return gradient graph for each input

    def R_op(self, inputs, eval_points):
        # return R_op graph for each input
