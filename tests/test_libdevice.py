# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir import types, compiler
from numba_cuda_mlir.cuda import libdevice, libdevicefuncs

from numba_cuda_mlir.cuda.libdevice import (
    sincos as sincos_direct,
    frexp as frexp_direct,
    sad as sad_direct,
)


def mark_sync(func):
    func = pytest.mark.xdist_group("libdevice-sync")(func)
    func = pytest.mark.libdevice(func)
    return func


def use_sincos_module_attr(s, c, x):
    i = cuda.grid(1)

    if i < len(x):
        sr, cr = cuda.libdevice.sincos(x[i])
        s[i] = sr
        c[i] = cr


def use_frexp_module_attr(frac, exp, x):
    i = cuda.grid(1)

    if i < len(x):
        fracr, expr = cuda.libdevice.frexp(x[i])
        frac[i] = fracr
        exp[i] = expr


def use_sad_module_attr(r, x, y, z):
    i = cuda.grid(1)

    if i < len(x):
        r[i] = cuda.libdevice.sad(x[i], y[i], z[i])


def use_sincos_module_import(s, c, x):
    i = cuda.grid(1)

    if i < len(x):
        sr, cr = libdevice.sincos(x[i])
        s[i] = sr
        c[i] = cr


def use_frexp_module_import(frac, exp, x):
    i = cuda.grid(1)

    if i < len(x):
        fracr, expr = libdevice.frexp(x[i])
        frac[i] = fracr
        exp[i] = expr


def use_sad_module_import(r, x, y, z):
    i = cuda.grid(1)

    if i < len(x):
        r[i] = libdevice.sad(x[i], y[i], z[i])


# Kernels using direct function imports: sincos_direct()
def use_sincos_direct_import(s, c, x):
    i = cuda.grid(1)

    if i < len(x):
        sr, cr = sincos_direct(x[i])
        s[i] = sr
        c[i] = cr


def use_frexp_direct_import(frac, exp, x):
    i = cuda.grid(1)

    if i < len(x):
        fracr, expr = frexp_direct(x[i])
        frac[i] = fracr
        exp[i] = expr


def use_sad_direct_import(r, x, y, z):
    i = cuda.grid(1)

    if i < len(x):
        r[i] = sad_direct(x[i], y[i], z[i])


@mark_sync
@pytest.mark.parametrize(
    "kernel_func,import_style",
    [
        (use_sincos_module_attr, "cuda.libdevice.sincos"),
        (use_sincos_module_import, "libdevice.sincos"),
        (use_sincos_direct_import, "sincos_direct"),
    ],
)
def test_sincos(kernel_func, import_style):
    # Tests return of a UniTuple from a libdevice function
    arr = cuda.to_device(np.arange(100, dtype=np.float64))
    sres = cuda.to_device(np.zeros_like(arr))
    cres = cuda.to_device(np.zeros_like(arr))

    cufunc = cuda.jit(kernel_func)
    cufunc[4, 32](sres, cres, arr)

    sres, cres = sres.copy_to_host(), cres.copy_to_host()
    arr = arr.copy_to_host()

    np.testing.assert_allclose(np.cos(arr), cres)
    np.testing.assert_allclose(np.sin(arr), sres)


@mark_sync
@pytest.mark.parametrize(
    "kernel_func,import_style",
    [
        (use_frexp_module_attr, "cuda.libdevice.frexp"),
        (use_frexp_module_import, "libdevice.frexp"),
        (use_frexp_direct_import, "frexp_direct"),
    ],
)
def test_frexp(kernel_func, import_style):
    # Test return of a Tuple from a libdevice function
    arr = cuda.to_device(np.linspace(start=1.0, stop=10.0, num=100, dtype=np.float64))
    fracres = cuda.to_device(np.zeros_like(arr))
    expres = cuda.to_device(np.zeros(shape=arr.shape, dtype=np.int32))

    cufunc = cuda.jit(kernel_func)
    cufunc[4, 32](fracres, expres, arr)

    frac_expect, exp_expect = np.frexp(arr.copy_to_host())

    fracres, expres = fracres.copy_to_host(), expres.copy_to_host()

    np.testing.assert_array_equal(frac_expect, fracres)
    np.testing.assert_array_equal(exp_expect, expres)


@mark_sync
@pytest.mark.parametrize(
    "kernel_func,import_style",
    [
        (use_sad_module_attr, "cuda.libdevice.sad"),
        (use_sad_module_import, "libdevice.sad"),
        (use_sad_direct_import, "sad_direct"),
    ],
)
def test_sad(kernel_func, import_style):
    # Test return of a scalar from a libdevice function
    x = cuda.to_device(np.arange(0, 200, 2))
    y = cuda.to_device(np.arange(50, 150))
    z = cuda.to_device(np.arange(15, 115))
    r = cuda.to_device(np.zeros_like(x))

    cufunc = cuda.jit(kernel_func)
    cufunc[4, 32](r, x, y, z)

    x, y, z, r = x.copy_to_host(), y.copy_to_host(), z.copy_to_host(), r.copy_to_host()
    np.testing.assert_array_equal(np.abs(x - y) + z, r)


# A template for generating tests of compiling calls to libdevice functions.
# The purpose of the call and assignment of the return variables is to ensure
# the actual function implementations are not thrown away resulting in a PTX
# implementation that only contains the ret instruction - this may hide certain
# errors.
function_template = """\
def pyfunc(%(pyargs)s):
    ret = libdevice.%(func)s(%(funcargs)s)
    %(retvars)s = ret
"""


def make_test_call(libname):
    """
    Generates a test function for each libdevice function.
    """

    def _test_call_functions():
        # Strip off '__nv_' from libdevice name to get Python name
        apiname = libname[5:]
        apifunc = getattr(libdevice, apiname)  # noqa: F841
        retty, args = libdevicefuncs.functions[libname]
        sig = libdevicefuncs.create_signature(retty, args)

        # Construct arguments to the libdevice function. These are all
        # non-pointer arguments to the underlying bitcode function.
        funcargs = ", ".join(
            ["a%d" % i for i, arg in enumerate(args) if not arg.is_ptr]
        )

        # Arguments to the Python function (`pyfunc` in the template above) are
        # the arguments to the libdevice function, plus as many extra arguments
        # as there are in the return type of the libdevice function - one for
        # scalar-valued returns, or the length of the tuple for tuple-valued
        # returns.
        if isinstance(sig.return_type, (types.Tuple, types.UniTuple)):
            # Start with the parameters for the return values
            pyargs = ", ".join(["r%d" % i for i in range(len(sig.return_type))])
            # Add the parameters for the argument values
            pyargs += ", " + funcargs
            # Generate the unpacking of the return value from the libdevice
            # function into the Python function return values (`r0`, `r1`,
            # etc.).
            retvars = ", ".join(["r%d[0]" % i for i in range(len(sig.return_type))])
        else:
            # Scalar return is a more straightforward case
            pyargs = "r0, " + funcargs
            retvars = "r0[0]"

        # Create the string containing the function to compile
        d = {
            "func": apiname,
            "pyargs": pyargs,
            "funcargs": funcargs,
            "retvars": retvars,
        }
        code = function_template % d

        # Convert the string to a Python function
        locals = {}
        exec(code, globals(), locals)
        pyfunc = locals["pyfunc"]

        # Compute the signature for compilation. This mirrors the creation of
        # arguments to the Python function above.
        pyargs = [arg.ty for arg in args if not arg.is_ptr]
        if isinstance(sig.return_type, (types.Tuple, types.UniTuple)):
            pyreturns = [ret[::1] for ret in sig.return_type]
            pyargs = pyreturns + pyargs
        else:
            pyargs.insert(0, sig.return_type[::1])

        pyargs = tuple(pyargs)
        jitted = cuda.jit(pyfunc)
        compiled = compiler.compile_result(jitted, pyargs)
        ptx = compiled.ptx
        if isinstance(ptx, bytes):
            ptx = ptx.decode("utf-8")

        # If the function body was discarded by optimization (therefore making
        # the test a bit weak), there won't be any loading of parameters -
        # ensure that a load from parameters occurs somewhere in the PTX
        assert "ld.param" in ptx

        # Returning the result (through a passed-in array) should also require
        # a store to global memory, so check for at least one of those too.
        assert "st.global" in ptx

    return _test_call_functions


for libname in libdevicefuncs.functions:
    _test_name = "test_%s" % libname.replace("__nv_", "libdevice_")
    test_call = make_test_call(libname)
    test_call = mark_sync(test_call)
    globals()[_test_name] = test_call


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    test_frexp()
    test_sincos()
    test_sad()
    for libname in libdevicefuncs.functions:
        _test_name = "test_%s" % libname.replace("__nv_", "libdevice_")
        test = _test_name
        logging.debug(f"Running {_test_name}")
        locals()[test]()
