from theano import Op, Apply
from theano.tensor import as_tensor_variable

class ScalMulC(Op):
    __props__ = ('scal',)

    def __init__(self, scal):
        if not isinstance(scal, int):
            raise TypeError('expected an int')
        self.scal = scal

    def make_node(self, x):
        x = as_tensor_variable(x)
        if x.ndim != 1:
            raise TypeError("ScalMulC only works on 1D")
        return Apply(self, [x], [x.type()])

    def c_code(self, node, name, input_names,
               output_names, sub):
        return """
Py_XDECREF(%(out)s);
%(out)s = (PyArrayObject *)PyArray_NewLikeArray(
    %(inp)s, NPY_ANYORDER, NULL, 0);
if (%(out)s == NULL) {
  %(fail)s
}
for (npy_intp i = 0; i < PyArray_DIM(%(inp)s, 0); i++) {
  *(dtype_%(out)s *)PyArray_GETPTR1(%(out)s, i) =
    (*(dtype_%(inp)s *)PyArray_GETPTR1(%(inp)s, i)) * %(scal)d;
}
""" % % dict(inp=input_names[0], out=output_names[0],
           fail=sub["fail"], scal=self.scal)

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def grad(self, inputs, output_grads):
        return [output_grads[0] * self.scal]
