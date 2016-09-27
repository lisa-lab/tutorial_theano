#section support_code

int APPLY_SPECIFIC(doublecop)(PyArrayObject *x,
                              PyArrayObject **out) {
  Py_XDECREF(*out);
  *out = (PyArrayObject *)PyArray_NewLikeArray(
                           inp, NPY_ANYORDER, NULL, 0);
  if (*out == NULL)
    return -1;

  for (npy_intp i = 0; i < PyArray_DIM(x, 0); i++) {
    *(DTYPE_OUTPUT_0 *)PyArray_GETPTR1(*out, i) =
      (*(DTYPE_INPUT_0 *)PyArray_GETPTR1(x, i)) * 2;
  }
  return 0;
}
