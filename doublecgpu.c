#section kernels
#kernel doublek : *, *, size :

KERNEL void doublek(GLOBAL_MEM DTYPE_o0 *out,
                    GLOBAL_MEM DTYPE_i0 *a,
                    ga_size n) {
  for (ga_size i = LID_0; i < n; i += LDIM_0) {
    out[i] = 2 * a[i];
  }
}

#section support_code_struct
int double_fn(PyGpuArrayObject *inp,
              PyGpuArrayObject **out,
              PyGpuContextObject *ctx) {
  size_t n = 1;
  Py_XDECREF(*out);
  *out = pygpu_empty(PyGpuArray_NDIM(inp),
                     PyGpuArray_DIMS(inp),
                     GA_C_ORDER, ctx, Py_None);
  if (*out == NULL) return -1;
  for (unsigned int i = 0; i < inp->ga.nd; i++)
    n *= PyGpuArray_DIM(inp, i);
  if (doublek_scall(1, &n, 0, *out, inp, n)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Error calling kernel");
    return -1;
  }
}
