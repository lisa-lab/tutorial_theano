from theano import Op

class MyOp(Op):
    params_type = # a params type here

    def __init__(self, ...):
        # Get some params

    def get_params(self, node):
        # Return a params object
