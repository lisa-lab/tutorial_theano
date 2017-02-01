from scalmulop import ScalMulV1
from doubleop import DoubleOp
from doublecop import DoubleCOp
from doublec import DoubleC
from doublecgpu import DoubleCGpu

from theano.gof import local_optimizer
from theano.tensor.opt import register_specialize
from theano.gpuarray.opt import (register_opt, op_lifter,
                                 register_opt2)


@register_specialize
@local_optimizer([ScalMulV1])
def local_scalmul_double(node):
    if not (isinstance(node.op, ScalMulV1) and
                node.op.scal == 2):
        return False

    return [DoubleOp()(node.inputs[0])]


@register_opt('fast_compile')
@op_lifter([DoubleOp, DoubleC, DoubleCOp])
@register_opt2([DoubleOp, DoubleC, DoubleCOp],
               'fast_compile')
def local_scalmul_double_gpu(op, context_name, inputs,
                             outputs):
    return DoubleCGpu
