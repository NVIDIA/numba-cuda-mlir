# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    Registry,
    signature,
)
from cusimt import types
from cusimt.type_defs.vector_types import VectorType
from cusimt.errors import ForceLiteralArg

registry = Registry()


def _extract_shape(shape_arg):
    """Extract shape from literal argument."""
    if isinstance(shape_arg, types.IntegerLiteral):
        return (shape_arg.literal_value,)
    elif isinstance(shape_arg, (types.Tuple, types.UniTuple)):
        if hasattr(shape_arg, "types"):
            shape = []
            for t in shape_arg.types:
                if isinstance(t, types.IntegerLiteral):
                    shape.append(t.literal_value)
                else:
                    return None
            return tuple(shape)
    return None


@registry.register
class VectorLoadTemplate(AbstractTemplate):
    from cusimt.cuda import vector

    key = vector.load

    def generic(self, args, kws):
        # Handle both 3 args (no alignment) and 4 args (with alignment)
        if len(args) not in (3, 4):
            return None

        array, index, shape = args[0], args[1], args[2]
        alignment = args[3] if len(args) == 4 else None

        if not isinstance(array, (types.Array, types.CPointer)):
            return None

        # CPointer only supports 1D (scalar) index
        if isinstance(array, types.CPointer) and isinstance(
            index, (types.Tuple, types.UniTuple)
        ):
            raise TypeError(
                "vector.load with CPointer only supports scalar index, not tuple. "
                "Linearize the index before passing to vector.load."
            )

        vec_shape = _extract_shape(shape)
        if vec_shape is None:
            if isinstance(shape, types.Integer):
                raise ForceLiteralArg({2})
            return None

        # If alignment is provided, check its type
        if alignment is not None:
            if isinstance(alignment, types.NoneType):
                # Explicit None passed - treat as unaligned, fall through
                pass
            elif isinstance(alignment, types.IntegerLiteral):
                # Aligned path - alignment must be a compile-time constant
                return signature(
                    VectorType(array.dtype, vec_shape), array, index, shape, alignment
                )
            elif isinstance(alignment, types.Integer):
                # Force user to provide a literal for alignment
                raise ForceLiteralArg({3})
            else:
                # Invalid alignment type
                return None

        # No alignment provided or explicit None - unaligned path
        if isinstance(array, types.CPointer):
            raise TypeError(
                "vector.load with CPointer requires an explicit alignment argument. "
                "Use vector.load(ptr, index, shape, alignment=N)."
            )
        return signature(VectorType(array.dtype, vec_shape), array, index, shape)


@registry.register
class VectorStoreTemplate(AbstractTemplate):
    from cusimt.cuda import vector

    key = vector.store

    def generic(self, args, kws):
        # Handle both 3 args (no alignment) and 4 args (with alignment)
        if len(args) not in (3, 4):
            return None

        array, index, vec = args[0], args[1], args[2]
        alignment = args[3] if len(args) == 4 else None

        if not isinstance(array, (types.Array, types.CPointer)):
            return None
        if not isinstance(vec, VectorType):
            return None

        # CPointer only supports 1D (scalar) index
        if isinstance(array, types.CPointer) and isinstance(
            index, (types.Tuple, types.UniTuple)
        ):
            raise TypeError(
                "vector.store with CPointer only supports scalar index, not tuple. "
                "Linearize the index before passing to vector.store."
            )

        # If alignment is provided, check its type
        if alignment is not None:
            if isinstance(alignment, types.NoneType):
                # Explicit None passed - treat as unaligned, fall through
                pass
            elif isinstance(alignment, types.IntegerLiteral):
                # Aligned path - alignment must be a compile-time constant
                return signature(types.none, array, index, vec, alignment)
            elif isinstance(alignment, types.Integer):
                # Force user to provide a literal for alignment
                raise ForceLiteralArg({3})
            else:
                # Invalid alignment type
                return None

        # No alignment provided or explicit None - unaligned path
        if isinstance(array, types.CPointer):
            raise TypeError(
                "vector.store with CPointer requires an explicit alignment argument. "
                "Use vector.store(ptr, index, vec, alignment=N)."
            )
        return signature(types.none, array, index, vec)


@registry.register_attr
class VectorModuleTemplate(AttributeTemplate):
    from cusimt.cuda import vector as vector_module

    key = types.Module(vector_module)

    def resolve_load(self, mod):
        return types.Function(VectorLoadTemplate)

    def resolve_store(self, mod):
        return types.Function(VectorStoreTemplate)
