# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
from types import MappingProxyType

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.datamodel.registry import register
from numba_cuda_mlir.numba_cuda.typing.asnumbatype import as_numba_type
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof_impl
from numba_cuda_mlir.numba_cuda.typing.templates import (
    Registry,
    _OverloadAttributeTemplate,
    _OverloadFunctionTemplate,
    _OverloadMethodTemplate,
    make_overload_template,
    make_overload_attribute_template,
)

from numba_cuda_mlir.numba_cuda.extending import (
    intrinsic,
    _Intrinsic,
    type_callable,
)

from numba_cuda_mlir.models import mlir_data_manager

from numba_cuda_mlir.lowering_registry import LoweringRegistry
from numba_cuda_mlir.extending.argument_handler import ArgumentHandler

lowering_registry = LoweringRegistry()
typing_registry = Registry()
lower_cast = lowering_registry.lower_cast
lower_builtin = lowering_registry.lower
register_model = functools.partial(register, mlir_data_manager)

__all__ = [
    "ArgumentHandler",
    "intrinsic",
    "lowering_registry",
    "as_numba_type",
    "lower_builtin",
    "lower_cast",
    "overload",
    "overload_attribute",
    "overload_method",
    "register_jitable",
    "type_callable",
    "typing_registry",
    "typeof_impl",
]

_overload_default_jit_options = {"no_cpython_wrapper": True, "nopython": True}


def _require_typing_registry(decorator_name, registry):
    if registry is None:
        raise ValueError(f"numba_cuda_mlir.extending.{decorator_name} requires typing_registry=")
    return registry


def _require_lowering_registry(decorator_name, registry):
    if registry is None:
        raise ValueError(f"numba_cuda_mlir.extending.{decorator_name} requires lowering_registry=")
    return registry


class _NumbaCudaMlirOverloadFunctionTemplate(_OverloadFunctionTemplate):
    def _get_jit_decorator(self):
        from numba_cuda_mlir import cuda
        from numba_cuda_mlir.mlir_compiler import get_compiler_class

        def jit_with_mlir_pipeline(**jit_options):
            jit_decorator = cuda.jit(**jit_options)

            def decorate(pyfunc):
                disp = jit_decorator(pyfunc)
                fcomp = getattr(disp, "_compiler", None)
                if fcomp is not None and getattr(fcomp, "pipeline_class", None) is None:
                    fcomp.pipeline_class = get_compiler_class(disp.targetoptions)
                return disp

            return decorate

        return jit_with_mlir_pipeline


def overload(
    func,
    jit_options=MappingProxyType({}),
    strict=True,
    inline="never",
    prefer_literal=False,
    typing_registry=None,
    **kwargs,
):
    """Register an implementation for ``func``.

    The decorated function is a *typer*: it is called at compile time with the
    Numba types of the arguments and must return a Python function (the
    implementation), or ``None`` to decline the overload. The implementation
    is compiled by Numba-CUDA-MLIR's MLIR pipeline.

    Parameters
    ----------
    func
        The Python callable being overloaded.
    jit_options : Mapping
        Options forwarded to ``cuda.jit`` when compiling the implementation.
    strict : bool
        If ``True``, raise when the implementation cannot be compiled. If
        ``False``, the failure is silenced (used by :func:`register_jitable`).
    inline : str
        Inlining policy: ``"never"``, ``"always"``, or a cost-model callable.
    prefer_literal : bool
        If ``True``, prefer literal-typed arguments when resolving the
        overload.
    """
    selected_typing_registry = _require_typing_registry("overload", typing_registry)
    jit_options = dict(jit_options)
    opts = _overload_default_jit_options.copy()
    opts.update(jit_options)

    def decorate(overload_func):
        template = make_overload_template(
            func,
            overload_func,
            opts,
            strict,
            inline,
            prefer_literal,
            base=_NumbaCudaMlirOverloadFunctionTemplate,
            **kwargs,
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
        selected_lowering_registry = cls._attribute_lowering_registry

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
    """Register an implementation for a read-only attribute on a Numba type.

    The decorated function is a typer with the same contract as
    :func:`overload`. Its only parameter is the receiver, and the returned
    implementation is a function of the receiver that produces the attribute
    value.

    Parameters
    ----------
    typ
        The Numba type on which the attribute is being defined.
    attr : str
        The name of the attribute being defined.
    """

    selected_typing_registry = _require_typing_registry("overload_attribute", typing_registry)
    selected_lowering_registry = _require_lowering_registry("overload_attribute", lowering_registry)

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


def overload_method(typ, attr, typing_registry=None, **kwargs):
    """Register an implementation for a method on a Numba type.

    The decorated function is a typer with the same contract as
    :func:`overload`. Its first parameter is the receiver (``self``); any
    additional parameters become method arguments.

    Parameters
    ----------
    typ
        The Numba type on which the method is being defined.
    attr : str
        The name of the method being defined.
    """

    selected_typing_registry = _require_typing_registry("overload_method", typing_registry)

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


def overload_classmethod(typ, attr, typing_registry=None, **kwargs):
    return overload_method(types.TypeRef(typ), attr, typing_registry=typing_registry, **kwargs)


def register_jitable(*args, typing_registry=None, **kwargs):
    """Mark a plain Python function as compilable from device code.

    The function is registered as a non-strict overload of itself, so calls
    to it from inside a kernel or device function dispatch to the original
    Python source compiled by Numba-CUDA-MLIR. ``@register_jitable`` functions
    may call other ``@register_jitable`` functions, ``@cuda.jit`` device
    functions, and any built-in or overloaded operation the compiler
    understands.

    May be used with or without parentheses::

        @register_jitable
        def f(x): ...


        @register_jitable(inline="always")
        def g(x): ...
    """

    selected_typing_registry = _require_typing_registry("register_jitable", typing_registry)

    def wrap(fn):
        copied_kwargs = kwargs.copy()
        inline = copied_kwargs.pop("inline", "never")

        @overload(
            fn,
            jit_options=copied_kwargs,
            inline=inline,
            strict=False,
            typing_registry=selected_typing_registry,
        )
        def ov_wrap(*args, **kwargs):
            return fn

        fn.__numba_cuda_mlir_jitable__ = True
        return fn

    if kwargs or not args:
        return wrap
    if len(args) != 1:
        raise TypeError("register_jitable accepts at most one positional argument")
    return wrap(*args)


def make_attribute_wrapper(typeclass, struct_attr, python_attr):
    """
    Make an automatic attribute wrapper exposing member named *struct_attr*
    as a read-only attribute named *python_attr*.
    The given *typeclass*'s model must be a StructModel subclass.

    Vendored from cusimt.extending with a change to consider the CUDA data
    model manager.
    """
    from numba_cuda_mlir.numba_cuda.typing.templates import AttributeTemplate
    from numba_cuda_mlir.models import StructModel
    from numba_cuda_mlir.numba_cuda.core.imputils import impl_ret_borrowed

    from numba_cuda_mlir.typing.builtin import registry as cuda_registry
    from numba_cuda_mlir.lowering.builtins import registry as cuda_impl_registry

    data_model_manager = mlir_data_manager

    if not isinstance(typeclass, type) or not issubclass(typeclass, types.Type):
        raise TypeError("typeclass should be a Type subclass, got %s" % (typeclass,))

    def get_attr_fe_type(typ):
        """
        Get the Numba type of member *struct_attr* in *typ*.
        """
        model = data_model_manager.lookup(typ)
        if not isinstance(model, StructModel):
            raise TypeError(
                "make_struct_attribute_wrapper() needs a type "
                "with a StructModel, but got %s" % (model,)
            )
        return model.get_member_fe_type(struct_attr)

    @cuda_registry.register_attr
    class StructAttribute(AttributeTemplate):
        key = typeclass

        def generic_resolve(self, typ, attr):
            if attr == python_attr:
                return get_attr_fe_type(typ)

    @cuda_impl_registry.lower_getattr(typeclass, python_attr)
    def struct_getattr_impl(context, builder, target, value):
        from numba_cuda_mlir._mlir import ir as mlir_ir
        from numba_cuda_mlir._mlir.dialects import llvm as llvm_dialect

        value_type = builder.get_numba_type(value.name)
        model = data_model_manager.lookup(value_type)
        field_idx = model.get_field_position(struct_attr)
        field_mlir_type = model.get_model(field_idx).get_value_type()
        struct_val = builder.load_var(value)
        extracted = llvm_dialect.extractvalue(
            field_mlir_type,
            struct_val,
            position=mlir_ir.DenseI64ArrayAttr.get([field_idx]),
        )
        builder.store_var(target, extracted)
