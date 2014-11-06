from doubleop import DoubleOp
from doublec import DoubleC

from theano.gof import local_optimizer

from theano.tensor.opt import register_specialize

@register_specialize
@local_optimizer([DoubleOp])
def local_scalmul_double_v1(node):
    if not (isinstance(node.op, DoubleOp)
            and node.inputs[0].ndim == 1):
        return False

    return [DoubleC()(node.inputs[0])]
