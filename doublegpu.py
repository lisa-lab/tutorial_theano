from theano import Op, Apply
from theano.gpuarray.basic_ops import (as_gpuarray_variable, Kernel,
                                       infer_context_name, GpuKernelBase)

try:
    from pygpu import gpuarray
except ImportError:
    pass


class DoubleGpu(Op, GpuKernelBase):
    __props__ = ()

    def make_node(self, x):
        ctx_name = infer_context_name(x)
        x = as_gpuarray_variable(x, ctx_name)
        return Apply(self, [x], [x.type()])

    def get_params(self, node):
        return node.outputs[0].type.context

    def gpu_kernels(self, node, name):
        dt = node.inputs[0].type
        code = """
KERNEL void doublek(GLOBAL_MEM %(ctype) *out,
                   GLOBAL_MEM const %(ctype)s *a,
                   ga_size n) {
  for (ga_size i = LID_0; i < n; i += LDIM_0) {
    out[i] = 2 * a[i];
  }
}
""" % dict(ctype=gpuarray.dtype_to_ctype(dt))
        return [Kernel(code=code, name="doublek",
                       params=[gpuarray.GpuArray,
                               gpuarray.GpuArray,
                               gpuarray.SIZE],
                       flags=Kernel.get_flags(dt))]

    def c_code(self, node, name, inn, outn, sub):
        return """
size_t n = 1;
Py_XDECREF(%(out)s);
%(out)s = pygpu_empty(PyGpuArray_NDIM(%(inp)s),
                      PyGpuArray_DIMS(%(inp)s),
                      GA_C_ORDER, %(ctx)s, Py_None);
if (%(out)s == NULL) %(fail)s
for (unsigned int i = 0; i < %(inp)s->ga.nd; i++)
  n *= PyGpuArray_DIM(%(inp)s, i);
if (doublek_scall(1, &n, 0, %(out)s, %(inp)s, n)) {
  PyErr_SetString(PyExc_RuntimeError,
                  "Error calling kernel");
  %(fail)s;
}
""" % dict(inp=inn[0], out=outn[0], fail=sub["fail"])

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def grad(self, inputs, output_grads):
        return [output_grads[0] * 2]
