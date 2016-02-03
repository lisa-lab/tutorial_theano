from theano import Op

class MyOp(Op):
    params_type = # a params type here

    def __init__(self, ...):
        # Get some params

    # signature change
    def perform(self, node, inputs, out_storage, params):
        # do something

    def get_params(self, node):
        # Return a params object
