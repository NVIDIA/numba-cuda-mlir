# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The overload body must execute at most once per argument-type set.

``_impl_cache`` keys on the active ConfigStack flags, so resolving the same overloaded
call under two flag contexts re-runs the potentially expensive overload body once per
context.  ``_OverloadFunctionTemplate`` memoizes the overload result -- which is assumed
to depend only on the overload function and the argument types, never on the flags -- to
collapse those to a single execution.  These tests pin that down.
"""

import numpy as np

from numba_cuda_mlir import cuda, extending
from numba_cuda_mlir.extending import overload, refresh_registries, typing_registry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.typing.templates import make_overload_template


def _make_template(overload_func, inline="never"):
    def target(x):
        pass

    return make_overload_template(target, overload_func, jit_options={}, strict=True, inline=inline)


def test_distinct_arg_types_run_again():
    calls = []

    def ol(x):
        calls.append(x)

        def impl(x):
            pass

        return impl

    template = _make_template(ol)(None)

    template._call_overload_func((types.int32,), {})
    template._call_overload_func((types.int64,), {})

    assert len(calls) == 2


def test_kwargs_participate_in_key():
    calls = []

    def ol(x, flag=False):
        calls.append((x, flag))

        def impl(x, flag=False):
            pass

        return impl

    template = _make_template(ol)(None)

    template._call_overload_func((types.int32,), {})
    template._call_overload_func((types.int32,), {})
    template._call_overload_func((types.int32,), {"flag": True})

    assert len(calls) == 2


def test_distinct_overloads_do_not_alias():
    calls_a = []
    calls_b = []

    def ol_a(x):
        calls_a.append(x)

        def impl(x):
            pass

        return impl

    def ol_b(x):
        calls_b.append(x)

        def impl(x):
            pass

        return impl

    template_a = _make_template(ol_a)(None)
    template_b = _make_template(ol_b)(None)
    argty = types.int32

    ra1 = template_a._call_overload_func((argty,), {})
    ra2 = template_a._call_overload_func((argty,), {})
    rb1 = template_b._call_overload_func((argty,), {})

    # Two unrelated overloads resolved at the same argument type must not hand back one
    # another's implementation.  ``_overload_func`` is part of the cache key, so this
    # holds regardless of whether the two templates share a cache dict.
    assert len(calls_a) == 1
    assert len(calls_b) == 1
    assert ra1 is ra2
    assert ra1 is not rb1


def test_subclass_does_not_reuse_parent_cache_entries():
    """A subclass overriding ``_overload_func`` must not reuse the parent's results.

    ``_overload_result_cache`` is set in the generated template's class dict, so a
    subclass shares the parent's dict object.  Including ``_overload_func`` in the key is
    what keeps their entries apart -- without it the child's overload body would never
    run and callers would silently receive the *parent's* implementation.
    """
    calls_parent = []
    calls_child = []

    def ol_parent(x):
        calls_parent.append(x)

        def impl(x):
            pass

        return impl

    def ol_child(x):
        calls_child.append(x)

        def impl(x):
            pass

        return impl

    parent_cls = _make_template(ol_parent)
    child_cls = type("ChildTemplate", (parent_cls,), {"_overload_func": staticmethod(ol_child)})
    argty = types.int32

    r_parent = parent_cls(None)._call_overload_func((argty,), {})
    r_child = child_cls(None)._call_overload_func((argty,), {})

    assert len(calls_parent) == 1
    assert len(calls_child) == 1
    assert r_parent is not r_child


def test_cache_lives_on_template_class():
    # Template instances are transient -- Numba creates a fresh one per resolution -- so
    # the cache must live on the template *class* for a second instance to reuse the
    # first's result.
    calls = []

    def ol(x):
        calls.append(x)

        def impl(x):
            pass

        return impl

    template_cls = _make_template(ol)
    argty = types.int32

    template_cls(None)._call_overload_func((argty,), {})
    template_cls(None)._call_overload_func((argty,), {})

    assert len(calls) == 1


def test_shared_device_function_across_flag_contexts():
    """End-to-end: the case the memoization actually saves work in.

    Two kernels differing only in ``lto`` share a device function that calls an
    overloaded function.  The two compilations push different ``Flags`` onto the
    ConfigStack, so ``_impl_cache`` misses on the second kernel and ``_build_impl`` runs
    again -- without the memoization the overload body would execute twice.
    """
    calls = []

    def shared_target(x):
        pass

    @overload(shared_target, target="cuda", typing_registry=typing_registry)
    def ol_shared_target(x):
        calls.append(x)

        def impl(x):
            return x + 1

        return impl

    refresh_registries()

    @cuda.jit(device=True)
    def devfn(x):
        return shared_target(x)

    @cuda.jit(lto=False)
    def kernel_no_lto(out):
        out[0] = devfn(out[0])

    @cuda.jit(lto=True)
    def kernel_lto(out):
        out[0] = devfn(out[0])

    out = np.zeros(1, dtype=np.int64)
    kernel_no_lto[1, 1](out)
    assert out[0] == 1
    kernel_lto[1, 1](out)
    assert out[0] == 2

    assert len(calls) == 1


def test_overload_method_and_attribute_across_flag_contexts():
    """``@overload_method`` / ``@overload_attribute`` inherit the same memoization.

    ``_OverloadAttributeTemplate`` has no ``_build_impl``, so the coverage is indirect:
    both decorators also register the overload function as an ``@overload`` of itself,
    and attribute/method typing resolves through that function template.  If that
    self-registration in ``numba_cuda_mlir.extending`` ever changes, these decorators
    would silently start re-running their bodies once per flag context again.
    """
    method_calls = []
    attr_calls = []

    @extending.overload_method(
        types.Array, "cached_first_doubled", typing_registry=extending.typing_registry
    )
    def arr_first_doubled(arr):
        method_calls.append(arr)

        def impl(arr):
            return arr[0] * 2

        return impl

    @extending.overload_attribute(
        types.Array,
        "cached_size_doubled",
        typing_registry=extending.typing_registry,
        lowering_registry=extending.lowering_registry,
    )
    def arr_size_doubled(arr):
        attr_calls.append(arr)

        def get(arr):
            return arr.size * 2

        return get

    extending.refresh_registries()

    @cuda.jit(device=True)
    def dev_method(arr):
        return arr.cached_first_doubled()

    @cuda.jit(device=True)
    def dev_attr(arr):
        return arr.cached_size_doubled

    @cuda.jit(lto=False)
    def method_no_lto(arr, out):
        out[0] = dev_method(arr)

    @cuda.jit(lto=True)
    def method_lto(arr, out):
        out[0] = dev_method(arr)

    @cuda.jit(lto=False)
    def attr_no_lto(arr, out):
        out[0] = dev_attr(arr)

    @cuda.jit(lto=True)
    def attr_lto(arr, out):
        out[0] = dev_attr(arr)

    arr = np.array([21.0, 1.0], dtype=np.float64)
    float_out = np.zeros(1, dtype=np.float64)
    int_out = np.zeros(1, dtype=np.int64)

    method_no_lto[1, 1](arr, float_out)
    method_lto[1, 1](arr, float_out)
    assert float_out[0] == 42.0
    assert len(method_calls) == 1

    attr_no_lto[1, 1](arr, int_out)
    attr_lto[1, 1](arr, int_out)
    assert int_out[0] == 4
    assert len(attr_calls) == 1
