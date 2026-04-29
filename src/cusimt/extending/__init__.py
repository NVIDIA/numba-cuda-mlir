# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import MappingProxyType

from cusimt.numba_cuda import types
from cusimt.numba_cuda.typing.templates import (
    Registry,
    _OverloadAttributeTemplate,
    _OverloadMethodTemplate,
    make_overload_template,
    make_overload_attribute_template,
)

from cusimt.numba_cuda.extending import (
    intrinsic,
    _Intrinsic,
    type_callable,
)

from cusimt.mlir_lowering_registry import MLIRLoweringRegistry

lowering_registry = MLIRLoweringRegistry()
typing_registry = Registry()
lower_cast = lowering_registry.lower_cast

__all__ = [
    "lowering_registry",
    "lower_cast",
    "typing_registry",
]

_overload_default_jit_options = {"no_cpython_wrapper": True, "nopython": True}


def overload(
    func,
    jit_options=MappingProxyType({}),
    strict=True,
    inline="never",
    prefer_literal=False,
    **kwargs,
):
    jit_options = dict(jit_options)
    opts = _overload_default_jit_options.copy()
    opts.update(jit_options)

    def decorate(overload_func):
        template = make_overload_template(
            func, overload_func, opts, strict, inline, prefer_literal, **kwargs
        )
        typing_registry.register(template)
        if callable(func):
            typing_registry.register_global(func, types.Function(template))
        return overload_func

    return decorate


class _CusimtOverloadAttributeTemplate(_OverloadAttributeTemplate):
    """Override _init_once to register a cusimt-compatible lower_getattr."""

    _lowering_registered = set()

    def _init_once(self):
        cls = type(self)
        if cls in _CusimtOverloadAttributeTemplate._lowering_registered:
            return
        _CusimtOverloadAttributeTemplate._lowering_registered.add(cls)

        attr = cls._attr
        key = cls.key

        @lowering_registry.lower_getattr(key, attr)
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


class _CusimtOverloadMethodTemplate(_OverloadMethodTemplate):
    """Override _init_once to skip numba's llvmlite lowering registration.

    Method lowering in cusimt goes through BoundFunction → get_overload_builder,
    so the registry-based lowering from numba's _init_once is never used.
    Skipping it avoids polluting cusimt's lowering registry with numba closures.
    """

    def _init_once(self):
        pass


def overload_attribute(typ, attr, **kwargs):
    def decorate(overload_func):
        template = make_overload_attribute_template(
            typ, attr, overload_func, base=_CusimtOverloadAttributeTemplate, **kwargs
        )
        typing_registry.register_attr(template)
        overload(overload_func, **kwargs)(overload_func)
        return overload_func

    return decorate


def overload_method(typ, attr, **kwargs):
    def decorate(overload_func):
        copied_kwargs = kwargs.copy()
        template = make_overload_attribute_template(
            typ,
            attr,
            overload_func,
            inline=copied_kwargs.pop("inline", "never"),
            prefer_literal=copied_kwargs.pop("prefer_literal", False),
            base=_CusimtOverloadMethodTemplate,
            **copied_kwargs,
        )
        typing_registry.register_attr(template)
        overload(overload_func, **kwargs)(overload_func)
        return overload_func

    return decorate


def register_jitable(*args, **kwargs):
    def wrap(fn):
        inline = kwargs.pop("inline", "never")

        @overload(fn, jit_options=kwargs, inline=inline, strict=False)
        def ov_wrap(*args, **kwargs):
            return fn

        fn.__cusimt_jitable__ = True
        return fn

    if kwargs:
        return wrap
    else:
        return wrap(*args)
