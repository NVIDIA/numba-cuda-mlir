# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Variadic (``*args``) implementations registered through the overload family."""

import numpy as np
import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda.experimental import consteval
from numba_cuda_mlir.extending import (
    overload,
    overload_method,
    register_jitable,
    typing_registry,
)
from numba_cuda_mlir.numba_cuda import types


# Implementations below consume the bundle with ``consteval``, which unrolls the
# loop before compilation so the elements may differ in type (including a bundle
# of arrays). ``for a in consteval(args)`` iterates the elements directly;
# ``for i in consteval(range(n))`` exposes the index, with ``n`` reaching the
# implementation as a freevar from the typing function. A plain ``for a in args``
# is kept where the bundle is homogeneous, so that spelling stays covered too.


def var_sum(*args):
    pass


@overload(var_sum, target="cuda", typing_registry=typing_registry)
def ol_var_sum(*args):
    def impl(*args):
        acc = 0.0
        for a in args:
            acc += a
        return acc

    return impl


def offset_sum(base, *args):
    pass


@overload(offset_sum, target="cuda", typing_registry=typing_registry)
def ol_offset_sum(base, *args):
    def impl(base, *args):
        acc = base
        for a in args:
            acc += a
        return acc

    return impl


def var_count(*args):
    pass


@overload(var_count, target="cuda", typing_registry=typing_registry)
def ol_var_count(*args):
    n = len(args)

    def impl(*args):
        return n

    return impl


def var_store(out, *args):
    pass


@overload(var_store, target="cuda", typing_registry=typing_registry)
def ol_var_store(out, *args):
    n = len(args)

    def impl(out, *args):
        for i in consteval(range(n)):
            out[i] = args[i]

    return impl


def var_from_arrays(idx, *cols):
    pass


@overload(var_from_arrays, target="cuda", typing_registry=typing_registry)
def ol_var_from_arrays(idx, *cols):
    def impl(idx, *cols):
        acc = 0.0
        for c in consteval(cols):
            acc += c[idx]
        return acc

    return impl


def var_op(*args):
    pass


@overload(var_op, target="cuda", typing_registry=typing_registry)
def ol_var_op(*args):
    """Two distinct implementations, selected by element type."""
    if all(isinstance(a, types.Integer) for a in args):

        def impl(*args):
            acc = 0
            for a in args:
                acc += a
            return acc

        return impl

    def impl(*args):
        acc = 1.0
        for a in args:
            acc *= a
        return acc

    return impl


def mixed_sum(*args):
    pass


@overload(mixed_sum, target="cuda", typing_registry=typing_registry)
def ol_mixed_sum(*args):
    def impl(*args):
        acc = 0.0
        for a in consteval(args):
            acc += a
        return acc

    return impl


def skip_none_sum(*args):
    pass


@overload(skip_none_sum, target="cuda", typing_registry=typing_registry)
def ol_skip_none_sum(*args):
    keep = [i for i, t in enumerate(args) if not isinstance(t, types.NoneType)]

    def impl(*args):
        acc = 0.0
        for i in consteval(keep):
            acc += args[i]
        return acc

    return impl


def two_way(a, *rest):
    pass


# Registered first: variadic, and declines tuple elements so typing never
# selects it for the tuple-argument call below.
@overload(two_way, target="cuda", typing_registry=typing_registry)
def ol_two_way_variadic(a, *rest):
    if any(isinstance(r, types.BaseTuple) for r in rest):
        return None
    n = len(rest)

    def impl(a, *rest):
        return a + 10.0 * n

    return impl


# Registered second: takes the tuple. Its result is distinguishable.
@overload(two_way, target="cuda", typing_registry=typing_registry)
def ol_two_way_tuple(a, t):
    if not isinstance(t, types.BaseTuple):
        return None

    def impl(a, t):
        return a + 1000.0

    return impl


@overload_method(types.Array, "var_scale", target="cuda", typing_registry=typing_registry)
def ol_array_var_scale(arr, idx, *factors):
    def impl(arr, idx, *factors):
        acc = arr[idx]
        for f in factors:
            acc *= f
        return acc

    return impl


@register_jitable(typing_registry=typing_registry)
def jitable_var_sum(*args):
    acc = 0.0
    for a in args:
        acc += a
    return acc


def _run(kernel, out, *args):
    kernel[1, out.size](out, *args)
    return out


@cuda.jit
def _sum_of_1(out, a0):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = var_sum(a0[i])


@cuda.jit
def _sum_of_2(out, a0, a1):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = var_sum(a0[i], a1[i])


@cuda.jit
def _sum_of_5(out, a0, a1, a2, a3, a4):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = var_sum(a0[i], a1[i], a2[i], a3[i], a4[i])


@pytest.mark.parametrize("kernel, arity", [(_sum_of_1, 1), (_sum_of_2, 2), (_sum_of_5, 5)])
def test_overload_varargs_arities(kernel, arity):
    """A single registration serves every arity."""
    n = 4
    arrays = [np.arange(n, dtype=np.float32) * (10**i) for i in range(arity)]
    out = np.zeros(n, dtype=np.float32)
    _run(kernel, out, *arrays)
    np.testing.assert_allclose(out, sum(arrays))


def test_overload_varargs_empty_bundle():
    """An empty ``*args`` bundle contributes no operands to the call."""

    @cuda.jit
    def kernel(out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = var_count()

    out = np.full(4, -1, dtype=np.int64)
    kernel[1, out.size](out)
    np.testing.assert_array_equal(out, 0)


def test_overload_varargs_multiple_arities_in_one_kernel():
    """Distinct arities must compile to distinct callees, not collide."""

    @cuda.jit
    def kernel(out, a, b, c):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = var_sum(a[i]) + var_sum(a[i], b[i]) + var_sum(a[i], b[i], c[i])

    n = 4
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 10
    c = np.arange(n, dtype=np.float32) * 100
    out = np.zeros(n, dtype=np.float32)
    _run(kernel, out, a, b, c)
    np.testing.assert_allclose(out, a + (a + b) + (a + b + c))


def test_overload_varargs_selects_impl_by_type():
    """Distinct implementations of one variadic overload must not cross-talk.

    Guards the ``_impl_cache`` lookup: both entries live under the same
    template, and differ in element type and arity at once.
    """

    @cuda.jit
    def kernel(int_out, flt_out, ia, fa):
        i = cuda.grid(1)
        if i < int_out.size:
            int_out[i] = var_op(ia[i], ia[i], ia[i])  # 3 ints -> sum
            flt_out[i] = var_op(fa[i], fa[i])  # 2 floats -> product

    n = 4
    ia = np.arange(1, n + 1, dtype=np.int64)
    fa = np.arange(1, n + 1, dtype=np.float64)
    int_out = np.zeros(n, dtype=np.int64)
    flt_out = np.zeros(n, dtype=np.float64)
    kernel[1, n](int_out, flt_out, ia, fa)
    np.testing.assert_array_equal(int_out, ia * 3)
    np.testing.assert_allclose(flt_out, fa * fa)


def test_overload_fixed_plus_varargs():
    @cuda.jit
    def kernel(out, a, b):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = offset_sum(100.0, a[i], b[i])

    n = 4
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 10
    out = np.zeros(n, dtype=np.float32)
    _run(kernel, out, a, b)
    np.testing.assert_allclose(out, 100.0 + a + b)


def test_overload_varargs_heterogeneous_types():
    """The bundle need not be homogeneous; each leaf gets its own operand."""

    @cuda.jit
    def kernel(out, a, b, c):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = mixed_sum(a[i], b[i], c[i])

    n = 4
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.int32) * 10
    c = np.arange(n, dtype=np.float64) * 100
    out = np.zeros(n, dtype=np.float64)
    _run(kernel, out, a, b, c)
    np.testing.assert_allclose(out, a + b + c)


def test_overload_varargs_literal_argument():
    """A literal in the bundle must still match the cached implementation."""

    @cuda.jit
    def kernel(out, a):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = mixed_sum(a[i], 2, np.float64(0.5))

    n = 4
    a = np.arange(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float64)
    _run(kernel, out, a)
    np.testing.assert_allclose(out, a + 2 + 0.5)


def test_overload_varargs_array_arguments():
    """Array (memref) values inside the ``*args`` bundle."""

    @cuda.jit
    def kernel(out, a, b, c):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = var_from_arrays(i, a, b, c)

    n = 4
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 10
    c = np.arange(n, dtype=np.float32) * 100
    out = np.zeros(n, dtype=np.float32)
    _run(kernel, out, a, b, c)
    np.testing.assert_allclose(out, a + b + c)


def test_overload_varargs_none_leaf_in_bundle():
    """A ``None`` inside the bundle has no ABI slot and must not shift operands.

    The callee signature drops ``None`` leaves; the call site has to drop them
    the same way, or every operand after the ``None`` pairs with the wrong input.
    """

    @cuda.jit
    def kernel(out, a, b):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = skip_none_sum(a[i], None, b[i])

    n = 4
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 10
    out = np.zeros(n, dtype=np.float32)
    _run(kernel, out, a, b)
    np.testing.assert_allclose(out, a + b)


def test_overload_tuple_call_prefers_exact_cache_key():
    """An exact cache key must win over an unfolded ``*args`` form.

    Unfolding the tuple call's signature ``(f64, UniTuple(f64, 2))`` yields
    ``(f64, f64, f64)`` -- the cache key of the *variadic* registration's
    scalar call. Matching that first would silently lower the tuple call
    through the wrong implementation.
    """

    @cuda.jit
    def kernel(out, x):
        if cuda.grid(1) == 0:
            out[0] = two_way(x, 2.0, 3.0)  # scalar form, populates the variadic cache
            out[1] = two_way(x, (2.0, 3.0))  # tuple form, typed to the tuple overload

    out = np.zeros(2)
    kernel[1, 1](out, 1.0)
    np.testing.assert_allclose(out, [21.0, 1001.0])


def test_overload_varargs_no_return_value():
    @cuda.jit
    def kernel(out, a, b):
        i = cuda.grid(1)
        if i == 0:
            var_store(out, a[0], b[0], 7.0)

    a = np.array([1.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float64)
    kernel[1, 1](out, a, b)
    np.testing.assert_allclose(out, [1.0, 2.0, 7.0])


def test_overload_method_varargs():
    @cuda.jit
    def kernel(out, a):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = a.var_scale(i, 2.0, 3.0)

    n = 4
    a = np.arange(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    _run(kernel, out, a)
    np.testing.assert_allclose(out, a * 6.0)


def test_register_jitable_varargs():
    @cuda.jit
    def kernel(out, a, b):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = jitable_var_sum(a[i], b[i])

    n = 4
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 10
    out = np.zeros(n, dtype=np.float32)
    _run(kernel, out, a, b)
    np.testing.assert_allclose(out, a + b)
