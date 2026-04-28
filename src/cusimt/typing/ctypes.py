# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes
import operator
from cusimt.lowering_utilities.type_conversions import to_numba_type
from cusimt.logging import trace
from numba.core.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
    AbstractTemplate,
    Registry,
    signature,
)
from cusimt import types
from cusimt.type_defs.ctypes_types import CTypesType

registry = Registry()


@registry.register
class CastTemplate(AbstractTemplate):
    key = ctypes.cast

    def generic(self, args, kws):
        if len(args) != 2:
            return
        value, target_ctype = args
        if isinstance(target_ctype, CTypesType):
            target_ctype = to_numba_type(target_ctype.ctype)
        elif isinstance(target_ctype, types.NumberClass):
            target_ctype = target_ctype.dtype
        elif isinstance(target_ctype, types.Type):
            pass
        else:
            raise TypeError(
                f"ctypes.cast target must be a ctypes type, got {target_ctype=}, {type(target_ctype)=}"
            )

        return signature(target_ctype, *args)


@registry.register
class PointerFunctionTemplate(AbstractTemplate):
    key = ctypes.pointer

    def generic(self, args, kws):
        if len(args) != 1:
            return
        value = args[0]
        return signature(types.CPointer(types.none), value)


@registry.register_global(operator.isub)
@registry.register_global(operator.iadd)
@registry.register_global(operator.add)
@registry.register_global(operator.sub)
class PointerArithmeticTemplate(AbstractTemplate):
    def generic(self, args, kws):
        left, right = args
        match left, right:
            case types.CPointer() as ptr, types.Integer():
                pass
            # TODO(ajm): check what cuda c++ does here
            # case types.Integer(), types.CPointer() as ptr:
            #     pass
            case _:
                return None
        return signature(ptr, *args)


def _is_ctypes_type(obj) -> bool:
    from ctypes import _SimpleCData

    return hasattr(obj, "mro") and _SimpleCData in obj.mro()


@registry.register
class PointerTypeConstructorTemplate(AbstractTemplate):
    key = ctypes.POINTER

    def generic(self, args, kws):
        if len(args) != 1:
            raise TypeError(f"POINTER takes exactly 1 argument ({len(args)} given, {args=})")

        value = args[0]
        match value:
            case CTypesType() as ctwrapper:
                ctype = ctwrapper.ctype
            case t if _is_ctypes_type(t):
                ctype = t
            case _:
                raise TypeError(
                    f"POINTER argument must be a ctypes type, got {value=}, {type(value)=}"
                )

        numbatype = to_numba_type(ctype)
        return signature(types.CPointer(numbatype), value)


@registry.register_global(operator.getitem)
class PointerGetitemTemplate(AbstractTemplate):
    def generic(self, args, kws):
        match args:
            case types.CPointer() as ptr, types.Integer() as idx:
                # Dereferencing a pointer returns the element type
                return signature(ptr.dtype, ptr, idx)
            case _:
                return None


@registry.register_global(operator.setitem)
class PointerSetitemTemplate(AbstractTemplate):
    def generic(self, args, kws):
        match args:
            case (
                types.CPointer() as ptr,
                types.Integer() as idx,
                types.Number() as value,
            ):
                return signature(types.none, ptr, idx, value)
            case _:
                return None


@registry.register_attr
class Ctypes_stub_resolver(AttributeTemplate):
    key = types.Module(ctypes)

    def resolve(self, mod, attrname):
        from ctypes import _SimpleCData

        pymod = mod.pymod
        if hasattr(pymod, attrname):
            attr = getattr(pymod, attrname)
            if _is_ctypes_type(attr):
                return CTypesType(attr)
            elif callable(attr):
                match attr:
                    case ctypes.cast:
                        return types.Function(CastTemplate)
                    case ctypes.pointer:
                        return types.Function(PointerFunctionTemplate)
                    case ctypes.POINTER:
                        return types.Function(PointerTypeConstructorTemplate)
                    case _:
                        trace(
                            "Unknown ctypes callable attrname=%s, attr=%s",
                            attrname,
                            attr,
                        )
                        return None
            else:
                trace(f"ctypes attribute {attrname=} is not a simple data type or callable")
                return None
        else:
            trace("Unknown ctypes attribute attrname=%s, pymod=%s", attrname, pymod)
            return None
