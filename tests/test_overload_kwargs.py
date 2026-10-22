# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keyword arguments, and ``**kwargs`` implementations, through the overload family."""

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
from numba_cuda_mlir.numba_cuda.core.errors import TypingError


# The ``**kwargs`` bundle reaches the implementation as a named tuple whose
# fields are the keyword names in call order: ``kwargs.name`` reads one value,
# and ``len``, indexing and ``consteval`` iteration treat it as a tuple.


def weighted(a, b=1.0, c=2.0):
    pass


@overload(weighted, target="cuda", typing_registry=typing_registry)
def ol_weighted(a, b=1.0, c=2.0):
    def impl(a, b=1.0, c=2.0):
        return a + 10 * b + 100 * c

    return impl


@overload_method(types.Array, "kw_affine", target="cuda", typing_registry=typing_registry)
def ol_kw_affine(arr, idx, scale=1.0, shift=0.0):
    def impl(arr, idx, scale=1.0, shift=0.0):
        return arr[idx] * scale + shift

    return impl


seen_kwargs = []


def scaled(a, **kwargs):
    pass


@overload(scaled, target="cuda", typing_registry=typing_registry)
def ol_scaled(a, **kwargs):
    # The typing function receives the keywords as a name-to-type dict.
    seen_kwargs.append(dict(kwargs))
    if "scale" in kwargs:

        def impl(a, **kwargs):
            return a * kwargs.scale + len(kwargs)

        return impl

    def impl(a, **kwargs):
        return a + len(kwargs)

    return impl


def kw_sum(**kwargs):
    pass


@overload(kw_sum, target="cuda", typing_registry=typing_registry)
def ol_kw_sum(**kwargs):
    def impl(**kwargs):
        acc = 0.0
        for v in consteval(kwargs):
            acc += v
        return acc

    return impl


def kw_first(**kwargs):
    pass


@overload(kw_first, target="cuda", typing_registry=typing_registry)
def ol_kw_first(**kwargs):
    def impl(**kwargs):
        return kwargs[0]

    return impl


def kw_pack(**kwargs):
    pass


@overload(kw_pack, target="cuda", typing_registry=typing_registry)
def ol_kw_pack(**kwargs):
    def impl(**kwargs):
        return kwargs

    return impl


def kw_count(*args, **kwargs):
    pass


@overload(kw_count, target="cuda", typing_registry=typing_registry)
def ol_kw_count(*args, **kwargs):
    n_args, n_kwargs = len(args), len(kwargs)

    def impl(*args, **kwargs):
        return 10 * n_args + n_kwargs

    return impl


def kw_columns(i, **cols):
    pass


@overload(kw_columns, target="cuda", typing_registry=typing_registry)
def ol_kw_columns(i, **cols):
    def impl(i, **cols):
        return cols.x[i] * 10 + cols.y[i]

    return impl


def kw_everything(a, *args, scale=1.0, **kwargs):
    pass


@overload(kw_everything, target="cuda", typing_registry=typing_registry)
def ol_kw_everything(a, *args, scale=1.0, **kwargs):
    def impl(a, *args, scale=1.0, **kwargs):
        acc = a
        for x in consteval(args):
            acc += x
        for v in consteval(kwargs):
            acc += v
        return acc * scale

    return impl


@overload_method(types.Array, "kw_reduce", target="cuda", typing_registry=typing_registry)
def ol_kw_reduce(arr, idx, *factors, **offsets):
    def impl(arr, idx, *factors, **offsets):
        acc = arr[idx]
        for f in consteval(factors):
            acc *= f
        for v in consteval(offsets):
            acc += v
        return acc

    return impl


@register_jitable(typing_registry=typing_registry)
def jitable_kwargs(a, **kw):
    return a + kw.b * 10 + len(kw)


def _run(kernel, out, *args):
    kernel[1, out.size](out, *args)
    return out


def test_overload_keyword_arguments():
    """Keywords bind by name, in any order, alongside omitted defaults."""

    @cuda.jit
    def kernel(out):
        out[0] = weighted(1.0, c=3.0, b=2.0)
        out[1] = weighted(1.0, c=3.0)
        out[2] = weighted(a=1.0, b=2.0)
        out[3] = weighted(1.0, 2.0, c=4.0)

    out = np.zeros(4)
    kernel[1, 1](out)
    np.testing.assert_allclose(out, [321.0, 311.0, 221.0, 421.0])


def test_overload_method_keyword_arguments():
    @cuda.jit
    def kernel(out, a):
        out[0] = a.kw_affine(1, shift=5.0)
        out[1] = a.kw_affine(1, shift=5.0, scale=2.0)
        out[2] = a.kw_affine(idx=1, scale=3.0)
        out[3] = a.kw_affine(1)

    a = np.arange(4.0)
    out = np.zeros(4)
    kernel[1, 1](out, a)
    np.testing.assert_allclose(out, [6.0, 7.0, 3.0, 1.0])


def test_overload_kwargs_attribute_access():
    """The typing function sees the keywords by name; the implementation reads them by name."""

    @cuda.jit
    def kernel(out, a):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = scaled(a[i], scale=3.0) + 10 * scaled(a[i])

    seen_kwargs.clear()
    a = np.arange(4.0)
    out = _run(kernel, np.zeros(4), a)
    np.testing.assert_allclose(out, (a * 3.0 + 1) + 10 * a)
    assert {"scale": types.float64} in seen_kwargs
    assert {} in seen_kwargs


def test_overload_kwargs_names_select_entry():
    """Bundles with the same types under different names are different overloads.

    Both calls cache an entry whose types are ``(float64, float64)``; only the
    keyword name tells them apart, so lowering must compare names too or the
    second call runs through the first call's body.
    """

    @cuda.jit
    def kernel(out, a):
        out[0] = scaled(a[0], scale=3.0)
        out[1] = scaled(a[0], other=3.0)

    out = np.zeros(2)
    kernel[1, 1](out, np.array([2.0]))
    np.testing.assert_allclose(out, [7.0, 3.0])


def test_overload_kwargs_heterogeneous_in_any_order():
    @cuda.jit
    def kernel(out, a, b):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = kw_sum(x=a[i], y=b[i], z=3) + 100 * kw_sum(y=b[i], x=a[i])

    a = np.arange(4.0)
    b = np.arange(4, dtype=np.int32) * 10
    out = _run(kernel, np.zeros(4), a, b)
    np.testing.assert_allclose(out, (a + b + 3) + 100 * (a + b))


def test_overload_kwargs_empty_bundle():
    @cuda.jit
    def kernel(out):
        out[0] = kw_sum()
        out[1] = kw_count()
        out[2] = len(kw_pack())

    out = np.full(3, -1.0)
    kernel[1, 1](out)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0])


def test_overload_kwargs_indexing():
    @cuda.jit
    def kernel(out, a):
        out[0] = kw_first(q=a[0], r=5)

    out = np.zeros(1)
    kernel[1, 1](out, np.array([1.5]))
    np.testing.assert_allclose(out, [1.5])


def test_overload_kwargs_literal_value():
    """A literal keyword value must still match the cached implementation."""

    @cuda.jit
    def kernel(out, a):
        out[0] = scaled(a[0], scale=2, other=1.0)

    out = np.zeros(1)
    kernel[1, 1](out, np.array([2.0]))
    np.testing.assert_allclose(out, [6.0])


def test_overload_kwargs_array_values():
    @cuda.jit
    def kernel(out, x, y):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = kw_columns(i, y=y, x=x)

    x = np.arange(4.0)
    y = x + 0.5
    out = _run(kernel, np.zeros(4), x, y)
    np.testing.assert_allclose(out, x * 10 + y)


def test_overload_kwargs_underscore_name():
    """A keyword with a leading underscore, which ``namedtuple`` would reject, still works."""

    def kw_private(**kw):
        pass

    @overload(kw_private, target="cuda", typing_registry=typing_registry)
    def ol_kw_private(**kw):
        def impl(**kw):
            return kw._x * 10 + kw.y

        return impl

    @cuda.jit
    def kernel(out):
        out[0] = kw_private(_x=1.0, y=2)

    out = np.zeros(1)
    kernel[1, 1](out)
    np.testing.assert_allclose(out, [12.0])


def test_overload_kwargs_with_varargs_and_keyword_only():
    """``**kwargs`` collects only keywords no other parameter claims."""

    @cuda.jit
    def kernel(out, a):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = (
                kw_everything(a[i])
                + 10 * kw_everything(a[i], 1.0, 2.0, scale=2.0)
                + 100 * kw_everything(a[i], extra=4.0, scale=3.0)
            )

    a = np.arange(4.0)
    out = _run(kernel, np.zeros(4), a)
    np.testing.assert_allclose(out, a + 10 * 2 * (a + 3.0) + 100 * 3 * (a + 4.0))


def test_overload_kwargs_returned_bundle():
    """A returned bundle is an ordinary named tuple, with ``None`` leaves in place."""

    @cuda.jit
    def kernel(out, a):
        i = cuda.grid(1)
        if i < out.size:
            t = kw_pack(x=a[i], y=None, z=2.0)
            out[i] = t.x + t.z + len(t)

    a = np.arange(4.0)
    out = _run(kernel, np.zeros(4), a)
    np.testing.assert_allclose(out, a + 2.0 + 3)


def test_overload_kwargs_returned_bundle_is_one_argument():
    """A returned bundle passed on positionally is one argument, not keywords.

    The bundle a ``**kwargs`` parameter receives is marked as such; returned as
    a value it must lose that marker, or the next call spreads its fields back
    into keywords.
    """

    @cuda.jit
    def kernel(out, a):
        t = kw_pack(x=a[0], y=2.0)
        out[0] = kw_count(t)
        out[1] = kw_count(kw_pack(x=a[0]))
        out[2] = kw_count(a[0], t, z=1)

    out = np.zeros(3)
    kernel[1, 1](out, np.array([1.0]))
    np.testing.assert_allclose(out, [10.0, 10.0, 21.0])


def test_overload_method_kwargs():
    @cuda.jit
    def kernel(out, a):
        out[0] = a.kw_reduce(1, 2.0, p=1.0, q=2)
        out[1] = a.kw_reduce(1)

    a = np.arange(4.0)
    out = np.zeros(2)
    kernel[1, 1](out, a)
    np.testing.assert_allclose(out, [5.0, 1.0])


def test_register_jitable_kwargs():
    @cuda.jit
    def kernel(out, a):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = jitable_kwargs(a[i], b=2.0)

    a = np.arange(4.0)
    out = _run(kernel, np.zeros(4), a)
    np.testing.assert_allclose(out, a + 20.0 + 1)


@pytest.mark.parametrize(
    "impl_params",
    ["a", "a, **other"],
    ids=["missing", "renamed"],
)
def test_overload_kwargs_must_match_typing_signature(impl_params):
    """``**kwargs`` must appear by the same name in both signatures."""

    def mismatch(a, **kw):
        pass

    namespace = {}
    exec(f"def impl({impl_params}):\n    return a\n", namespace)

    @overload(mismatch, target="cuda", typing_registry=typing_registry)
    def ol_mismatch(a, **kw):
        return namespace["impl"]

    @cuda.jit
    def kernel(out):
        out[0] = mismatch(1.0, z=1)

    with pytest.raises(TypingError, match="VAR_KEYWORD"):
        kernel[1, 1](np.zeros(1))
