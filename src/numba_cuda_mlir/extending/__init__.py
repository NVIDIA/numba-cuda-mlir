# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import MappingProxyType

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.typing.templates import (
    Registry,
    _OverloadAttributeTemplate,
    _OverloadMethodTemplate,
    make_overload_template,
    make_overload_attribute_template,
)

from numba_cuda_mlir.numba_cuda.extending import (
    intrinsic,
    _Intrinsic,
    type_callable,
)

from numba_cuda_mlir.lowering_registry import LoweringRegistry

lowering_registry = LoweringRegistry()
typing_registry = Registry()
lower_cast = lowering_registry.lower_cast

__all__ = [
    "lowering_registry",
    "lower_cast",
    "typing_registry",
]

_overload_default_jit_options = {"no_cpython_wrapper": True, "nopython": True}


def _resolve_typing_registry(registry):
    return registry if registry is not None else typing_registry


def _resolve_lowering_registry(registry):
    return registry if registry is not None else lowering_registry


def overload(
    func,
    jit_options=MappingProxyType({}),
    strict=True,
    inline="never",
    prefer_literal=False,
    typing_registry=None,
    **kwargs,
):
    selected_typing_registry = _resolve_typing_registry(typing_registry)
    jit_options = dict(jit_options)
    opts = _overload_default_jit_options.copy()
    opts.update(jit_options)

    def decorate(overload_func):
        template = make_overload_template(
            func, overload_func, opts, strict, inline, prefer_literal, **kwargs
        )
        template._typing_registry = selected_typing_registry
        selected_typing_registry.register(template)
        if callable(func):
            selected_typing_registry.register_global(func, types.Function(template))
        return overload_func

    return decorate


class _NumbaCudaMlirOverloadAttributeTemplate(_OverloadAttributeTemplate):
    """Override _init_once to register a numba_cuda_mlir-compatible lower_getattr."""

    _lowering_registered = set()

    def _init_once(self):
        cls = type(self)
        if cls in _NumbaCudaMlirOverloadAttributeTemplate._lowering_registered:
            return
        _NumbaCudaMlirOverloadAttributeTemplate._lowering_registered.add(cls)

        attr = cls._attr
        key = cls.key
        selected_lowering_registry = getattr(cls, "_attribute_lowering_registry", lowering_registry)

        @selected_lowering_registry.lower_getattr(key, attr)
        def getattr_impl(context, builder, target, value):
            value_type = builder.get_numba_type(value.name)
            disp = cls._find_overload_dispatcher(context.typing_context, value_type)
            if disp is None:
                raise NotImplementedError(
                    f"No overload_attribute dispatcher for {value_type}.{attr}"
                )
            builder.lower_overload_call(target, disp, [value])

    @classmethod
    def _find_overload_dispatcher(cls, typing_context, typ):
        """Find the cached overload Dispatcher for the given type."""
        overload_func = cls._overload_func
        fnty = typing_context.resolve_value_type(overload_func)
        for temp_cls in getattr(fnty, "templates", []):
            if not hasattr(temp_cls, "_impl_cache"):
                continue
            for cache_key, cache_value in temp_cls._impl_cache.items():
                if cache_value is None or len(cache_key) != 4:
                    continue
                _, args, _, _ = cache_key
                if args == (typ,):
                    disp, _ = cache_value
                    if hasattr(disp, "py_func"):
                        return disp
        return None


class _NumbaCudaMlirOverloadMethodTemplate(_OverloadMethodTemplate):
    """Override _init_once to skip numba's llvmlite lowering registration.

    Method lowering in numba_cuda_mlir goes through BoundFunction → get_overload_builder,
    so the registry-based lowering from numba's _init_once is never used.
    Skipping it avoids polluting numba_cuda_mlir's lowering registry with numba closures.
    """

    def _init_once(self):
        pass


def overload_attribute(typ, attr, typing_registry=None, lowering_registry=None, **kwargs):
    selected_typing_registry = _resolve_typing_registry(typing_registry)
    selected_lowering_registry = _resolve_lowering_registry(lowering_registry)

    def decorate(overload_func):
        template = make_overload_attribute_template(
            typ,
            attr,
            overload_func,
            base=_NumbaCudaMlirOverloadAttributeTemplate,
            **kwargs,
        )
        template._attribute_lowering_registry = selected_lowering_registry
        selected_typing_registry.register_attr(template)
        overload(overload_func, typing_registry=selected_typing_registry, **kwargs)(overload_func)
        return overload_func

    return decorate


def overload_method(typ, attr, typing_registry=None, lowering_registry=None, **kwargs):
    selected_typing_registry = _resolve_typing_registry(typing_registry)

    def decorate(overload_func):
        copied_kwargs = kwargs.copy()
        template = make_overload_attribute_template(
            typ,
            attr,
            overload_func,
            inline=copied_kwargs.pop("inline", "never"),
            prefer_literal=copied_kwargs.pop("prefer_literal", False),
            base=_NumbaCudaMlirOverloadMethodTemplate,
            **copied_kwargs,
        )
        selected_typing_registry.register_attr(template)
        overload(overload_func, typing_registry=selected_typing_registry, **kwargs)(overload_func)
        return overload_func

    return decorate


def register_jitable(*args, typing_registry=None, lowering_registry=None, **kwargs):
    selected_typing_registry = _resolve_typing_registry(typing_registry)

    def wrap(fn):
        inline = kwargs.pop("inline", "never")

        @overload(
            fn,
            jit_options=kwargs,
            inline=inline,
            strict=False,
            typing_registry=selected_typing_registry,
        )
        def ov_wrap(*args, **kwargs):
            return fn

        fn.__numba_cuda_mlir_jitable__ = True
        return fn

    if kwargs:
        return wrap
    else:
        return wrap(*args)
