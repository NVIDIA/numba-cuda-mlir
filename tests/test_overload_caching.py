# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overload bodies are memoized on the argument types and the compiler options they read."""

import numpy as np
import pytest

from numba_cuda_mlir import cuda, extending
from numba_cuda_mlir.descriptor import mlir_target
from numba_cuda_mlir.extending import overload, refresh_registries, typing_registry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.core.targetconfig import ConfigStack
from numba_cuda_mlir.numba_cuda.flags import CUDAFlags
from numba_cuda_mlir.numba_cuda.typing.templates import make_overload_template, signature


@pytest.fixture
def flags_on_stack():
    """Compilation always has flags on the ConfigStack; direct callers must too."""
    with ConfigStack().enter(CUDAFlags()):
        yield


def _make_template(overload_func, inline="never"):
    def target(x):
        pass

    return make_overload_template(target, overload_func, jit_options={}, strict=True, inline=inline)


def test_distinct_arg_types_run_again(flags_on_stack):
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


def test_kwargs_participate_in_key(flags_on_stack):
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


def test_subclass_does_not_reuse_parent_cache_entries(flags_on_stack):
    """A subclass shares the parent's cache dict; ``_overload_func`` in the key keeps them apart."""
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


def test_cache_lives_on_template_class(flags_on_stack):
    # Instances are transient; the cache must live on the class.
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
    """Two kernels differing in ``lto`` share an overload: the body runs once, not per context."""
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
    """``@overload_method``/``@overload_attribute`` reach the same memoization via ``@overload``."""
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
    """A body reading a flag is re-resolved per flag value, whichever accessor it uses."""
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


@pytest.mark.parametrize(
    "mutate",
    [
        lambda flags: setattr(flags, "lto", True),
        lambda flags: delattr(flags, "lto"),
        lambda flags: flags.discard("lto"),
    ],
    ids=["assign", "delete", "discard"],
)
def test_flag_mutation_inside_overload_raises(mutate):
    """Writing a flag from an overload body raises and leaves the real flags untouched."""

    def target(x):
        pass

    def body(x):
        mutate(ConfigStack.top_or_none())

        def impl(x):
            pass

        return impl

    template_cls = make_overload_template(target, body, jit_options={}, strict=True, inline="never")
    flags = CUDAFlags()
    flags.lto = False
    with ConfigStack().enter(flags):
        with pytest.raises(TypingError, match="read-only during type inference"):
            template_cls(None)._call_overload_func((types.int32,), {})
    assert flags.lto is False, "the write must leave the real flags untouched"


def test_resolution_with_empty_configstack_is_not_cached():
    """Without flags on the stack nothing can be recorded, so nothing is cached or reused."""
    runs = []

    def body(x):
        flags = ConfigStack.top_or_none()
        runs.append(None if flags is None else flags.lto)

        def impl(x):
            pass

        return impl

    template = _make_template(body)(None)
    assert len(ConfigStack()) == 0

    template._call_overload_func((types.int32,), {})
    template._call_overload_func((types.int32,), {})
    assert runs == [None, None]

    flags = CUDAFlags()
    flags.lto = True
    with ConfigStack().enter(flags):
        template._call_overload_func((types.int32,), {})
    assert runs == [None, None, True]


def test_entry_reused_when_flags_agree_on_what_that_run_read():
    """An entry is reused exactly when the flags agree on what its own run read."""
    runs = []

    def body(x):
        flags = ConfigStack.top_or_none()
        debuginfo = flags.debuginfo if flags.lto else None
        runs.append((flags.lto, debuginfo))

        def impl(x):
            pass

        return impl

    template = _make_template(body)(None)

    def resolve(**options):
        flags = CUDAFlags()
        for name, value in options.items():
            setattr(flags, name, value)
        with ConfigStack().enter(flags):
            template._call_overload_func((types.int32,), {})

    resolve(lto=False, debuginfo=False)
    resolve(lto=True, debuginfo=False)
    resolve(lto=True, debuginfo=True)  # differs on an option that run read: re-run
    resolve(lto=False, debuginfo=True)  # differs only on one the lto=False run never read

    assert runs == [(False, None), (True, False), (True, True)]


def test_overload_builder_prefers_entry_agreeing_on_observed_options():
    """Lowering prefers flags agreeing on what the body read over the first argument match."""

    class Disp:
        py_func = None

    def flags(**options):
        result = CUDAFlags()
        for name, value in options.items():
            setattr(result, name, value)
        return result

    template_cls = _make_template(lambda x: None)
    args = (types.int32,)
    first, agreeing = Disp(), Disp()
    template_cls._impl_cache[(None, args, (), flags(lto=False, debuginfo=False))] = (first, args)
    template_cls._impl_cache[(None, args, (), flags(lto=True, debuginfo=False))] = (agreeing, args)
    # Body results behind those entries; each read only `lto`.
    template_cls._overload_result_cache[(template_cls._overload_func, args, ())] = {
        (("lto", "False"),): object(),
        (("lto", "True"),): object(),
    }

    with ConfigStack().enter(flags(lto=True, debuginfo=True)):
        builder = mlir_target.target_context.get_overload_builder(
            types.Function(template_cls), signature(types.int64, *args)
        )

    assert builder.__defaults__[0] is agreeing


def test_is_set_probe_is_recorded():
    """``is_set`` bypasses the getters and answers set-ness, so it is recorded as such."""

    def body(x):
        ConfigStack.top_or_none().is_set("lto")

        def impl(x):
            pass

        return impl

    template = _make_template(body)(None)

    def resolve(flags):
        with ConfigStack().enter(flags):
            return template._call_overload_func((types.int32,), {})

    explicit_true = CUDAFlags()
    explicit_true.lto = True
    explicit_default = CUDAFlags()
    explicit_default.lto = False

    unset = resolve(CUDAFlags())
    set_true = resolve(explicit_true)
    set_default = resolve(explicit_default)

    # is_set differs between unset and explicitly-default, though the values agree...
    assert set_default is not unset
    # ...and agrees between the two explicitly-set flags, though the values differ.
    assert set_default is set_true


@pytest.mark.parametrize("kind", ["attribute", "method"])
def test_flag_reading_attribute_and_method_across_flag_contexts(kind):
    """Attribute and method overloads reading a flag get an implementation per value."""
    runs = []

    def body(arr):
        lto = bool(ConfigStack.top_or_none().lto)
        runs.append(lto)

        if lto:

            def impl(arr):
                return 1
        else:

            def impl(arr):
                return 0

        return impl

    if kind == "attribute":
        extending.overload_attribute(
            types.Array,
            "flag_probe_attribute",
            typing_registry=extending.typing_registry,
            lowering_registry=extending.lowering_registry,
        )(body)

        @cuda.jit(lto=False)
        def k_no_lto(arr, out):
            out[0] = arr.flag_probe_attribute

        @cuda.jit(lto=True)
        def k_lto(arr, out):
            out[0] = arr.flag_probe_attribute
    else:
        extending.overload_method(
            types.Array, "flag_probe_method", typing_registry=extending.typing_registry
        )(body)

        @cuda.jit(lto=False)
        def k_no_lto(arr, out):
            out[0] = arr.flag_probe_method()

        @cuda.jit(lto=True)
        def k_lto(arr, out):
            out[0] = arr.flag_probe_method()

    extending.refresh_registries()
    arr = np.zeros(2, dtype=np.float64)
    a = np.zeros(1, dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    k_no_lto[1, 1](arr, a)
    k_lto[1, 1](arr, b)

    assert (a[0], b[0]) == (0, 1)
    assert len(runs) == 2


def test_overload_builder_does_not_take_flagless_entry_over_exact_match():
    """An entry resolved with no flags is a fallback, not an exact match for any flags."""

    class Disp:
        py_func = None

    template_cls = _make_template(lambda x: None)
    args = (types.int32,)
    flags = CUDAFlags()
    flags.lto = True
    flagless, exact = Disp(), Disp()
    template_cls._impl_cache[(None, args, (), None)] = (flagless, args)
    template_cls._impl_cache[(None, args, (), flags)] = (exact, args)
    # An is_set read on the flag-less entry must be skipped, not evaluated on None.
    template_cls._overload_result_cache[(template_cls._overload_func, args, ())] = {
        ((("is_set", "lto"), "True"),): object(),
    }

    with ConfigStack().enter(flags):
        builder = mlir_target.target_context.get_overload_builder(
            types.Function(template_cls), signature(types.int64, *args)
        )

    assert builder.__defaults__[0] is exact
