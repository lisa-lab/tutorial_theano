from scalmulop import ScalMulV1
from doubleop import DoubleOp

from theano.gof import local_optimizer

from theano.tensor.opt import register_specialize

@register_specialize
@local_optimizer([ScalMulV1])
def local_scalmul_double_v1(node):
    if not (isinstance(node.op, ScalMulV1)
            and node.op.scal == 2):
        return False

    return [DoubleOp()(node.inputs[0])]

from theano.gof.opt import OpSub

local_scalmul_double_v2 = OpSub(ScalMulV1(2), DoubleOp())

register_specialize(local_scalmul_double_v2,
                    name='local_scalmul_double_v2')
