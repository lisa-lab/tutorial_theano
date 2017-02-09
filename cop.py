from theano.gof import COp

class MyOp(COp):
    __props__ = ()

    def __init__(self, ...):
        COp.__init__(self, c_files, func_name)
        # Other init code if needed

    def make_node(self, ...):
        # make the Apply node
