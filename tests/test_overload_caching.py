# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The overload body must execute at most once per argument-type set.

``_impl_cache`` keys on the active ConfigStack flags, so resolving the same overloaded
call under two flag contexts re-runs the potentially expensive overload body once per
context.  ``_OverloadFunctionTemplate`` memoizes the overload result to collapse those
to a single execution, keyed on the overload function, the argument types, and -- for
bodies that consult the ConfigStack -- only the options they were observed to read.
These tests pin that down.
"""

import numpy as np
import pytest

from numba_cuda_mlir import cuda, extending
from numba_cuda_mlir.extending import overload, refresh_registries, typing_registry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.core.targetconfig import ConfigStack
from numba_cuda_mlir.numba_cuda.flags import CUDAFlags
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


def _flag_probe_kernels(body):
    """Register *body* as a cuda overload and return (lto=False, lto=True) kernels."""

    def target(x):
        pass

    overload(target, target="cuda", typing_registry=typing_registry)(body)
    refresh_registries()

    @cuda.jit(lto=False)
    def kernel_no_lto(out):
        out[0] = target(out[0])

    @cuda.jit(lto=True)
    def kernel_lto(out):
        out[0] = target(out[0])

    return kernel_no_lto, kernel_lto


def _read_lto_via_top_or_none():
    flags = ConfigStack.top_or_none()
    return bool(flags is not None and flags.lto)


def _read_lto_via_stack_top():
    stack = ConfigStack()
    flags = stack.top() if len(stack) else None
    return bool(flags is not None and flags.lto)


def _read_lto_via_copy():
    flags = ConfigStack.top_or_none()
    flags = flags.copy() if flags is not None else None
    return bool(flags is not None and flags.lto)


def _read_lto_via_values():
    flags = ConfigStack.top_or_none()
    return bool(flags is not None and flags.values().get("lto"))


def _read_lto_via_values_dict():
    flags = ConfigStack.top_or_none()
    return bool(flags is not None and flags._values.get("lto"))


@pytest.mark.parametrize(
    "read_lto",
    [
        _read_lto_via_top_or_none,
        _read_lto_via_stack_top,
        _read_lto_via_copy,
        _read_lto_via_values,
        pytest.param(
            _read_lto_via_values_dict,
            marks=pytest.mark.xfail(
                reason="Reads of the private `_values` dict bypass the recording "
                "properties. Intercepting it would need __getattribute__, which fires "
                "on every internal access and would widen every overload's key back "
                "to the full flag set.",
                strict=True,
            ),
        ),
    ],
    ids=["top_or_none", "stack_top", "copy", "values", "values_dict"],
)
def test_flag_reads_are_keyed_per_access_path(read_lto):
    """A body reading a flag is re-resolved per flag context, however it reads it.

    The recorder is installed by pushing recording flags onto the (thread-local)
    ConfigStack, so it is equally visible to ``top_or_none``, ``top()``, a ``copy()``
    of the flags (copies share the origin's read-set), and ``values()`` iteration
    (which reads every option through the recording getters and conservatively widens
    the key to all of them).  Each entry here was, or would be, a silent miscompile:
    an unrecorded read memoizes the first context's implementation and serves it to
    the second.
    """
    runs = []

    def body(x):
        lto = read_lto()
        runs.append(lto)

        if lto:

            def impl(x):
                return 1
        else:

            def impl(x):
                return 0

        return impl

    k_no_lto, k_lto = _flag_probe_kernels(body)
    a = np.zeros(1, dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    k_no_lto[1, 1](a)
    k_lto[1, 1](b)

    assert (a[0], b[0]) == (0, 1)
    assert len(runs) == 2


def test_unread_flags_do_not_force_re_resolution():
    """Only the options the body read may widen the key.

    Both kernels have the same ``lto``; they differ in ``debuginfo``, which the body
    never consults.  Keying on the whole flags object would re-run the body here.
    """
    runs = []

    def body(x):
        flags = ConfigStack.top_or_none()
        runs.append(bool(flags is not None and flags.lto))

        def impl(x):
            return 0

        return impl

    def target(x):
        pass

    overload(target, target="cuda", typing_registry=typing_registry)(body)
    refresh_registries()

    @cuda.jit()
    def plain(out):
        out[0] = target(out[0])

    @cuda.jit(debug=True, opt=False)
    def with_debug(out):
        out[0] = target(out[0])

    out = np.zeros(1, dtype=np.int64)
    plain[1, 1](out)
    with_debug[1, 1](out)

    assert len(runs) == 1


def test_flag_mutation_inside_overload_raises():
    """Compiler options are read-only while an overload body runs.

    The body is handed a recording copy of the flags, so a write would be silently
    discarded when the copy is popped -- and the overload is cached per observed
    option value, so mutating flags here could not affect the compiled result anyway.
    Every documented write path fails loudly instead.
    """

    def target(x):
        pass

    def make_template(mutate):
        def body(x):
            mutate(ConfigStack.top_or_none())

            def impl(x):
                pass

            return impl

        return make_overload_template(target, body, jit_options={}, strict=True, inline="never")

    writes = {
        "assign": lambda flags: setattr(flags, "lto", True),
        "delete": lambda flags: delattr(flags, "lto"),
        "discard": lambda flags: flags.discard("lto"),
    }

    for label, mutate in writes.items():
        template_cls = make_template(mutate)
        flags = CUDAFlags()
        flags.lto = False
        with ConfigStack().enter(flags):
            with pytest.raises(TypingError, match="read-only during type inference"):
                template_cls(None)._call_overload_func((types.int32,), {})
        assert flags.lto is False, f"{label} must leave the real flags untouched"
