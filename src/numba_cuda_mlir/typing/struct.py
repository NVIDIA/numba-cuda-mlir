# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Typing support for struct and union field access
"""

from numba_cuda_mlir.numba_cuda.typing.templates import AttributeTemplate, Registry
from numba_cuda_mlir.numba_cuda.extending import typeof_impl
from numba_cuda_mlir.type_defs.aggregate_types import AggregateType, UnionType
from numba_cuda_mlir.host.struct import StructInstance, StructTypeWrapper
from numba_cuda_mlir.host.union import UnionInstance, UnionTypeWrapper
from numba_cuda_mlir.numba_cuda import types as numba_types

registry = Registry()


@registry.register_attr
class AggregateTypeAttributeTemplate(AttributeTemplate):
    """
    Resolve attribute access on struct types (AggregateType).

    This tells Numba's type inference system what type each field has,
    enabling getattr operations like `my_struct.field_name`.
    """

    key = AggregateType

    def generic_resolve(self, typ, attr):
        """
        Resolve the type of a field access on a struct.

        Args:
            typ: The AggregateType instance (the struct type)
            attr: The attribute name being accessed (field name)

        Returns:
            The type of the field, or None if the field doesn't exist
        """
        # Look up the field in the struct's field list
        # Skip padding fields (field_name is None)
        for field_name, field_type, *_ in typ.fields:
            if field_name is not None and field_name == attr:
                return field_type

        # Field not found - return None and Numba will raise an error
        return None


@registry.register_attr
class UnionTypeAttributeTemplate(AttributeTemplate):
    """
    Resolve attribute access on union types (UnionType).

    This tells Numba's type inference system what type each variant has,
    enabling getattr operations like `my_union.variant_name`.
    """

    key = UnionType

    def generic_resolve(self, typ, attr):
        """
        Resolve the type of a variant access on a union.

        Args:
            typ: The UnionType instance (the union type)
            attr: The attribute name being accessed (variant name)

        Returns:
            The type of the variant, or None if the variant doesn't exist
        """
        # Look up the variant in the union's variant list
        for variant_name, variant_type in typ.variants:
            if variant_name == attr:
                # Unwrap StructTypeWrapper to get the underlying AggregateType
                if isinstance(variant_type, StructTypeWrapper):
                    return variant_type._type
                return variant_type

        # Variant not found - return None and Numba will raise an error
        return None


@typeof_impl.register(StructInstance)
def typeof_struct_instance(val, c):
    """
    Type inference for StructInstance objects.

    This allows passing struct instances created on the host to JIT-compiled kernels.
    """
    return val._struct_type


@typeof_impl.register(StructTypeWrapper)
def typeof_struct_type_wrapper(val, c):
    """
    Type inference for StructTypeWrapper.

    When used as a value (e.g., called in device code), we need to return
    a Function type with the registered template so the typing system can
    resolve calls to it.
    """

    # If used in device code (has a constructor template), return Function type
    if val._constructor_template is not None:
        return numba_types.Function(template=val._constructor_template)

    # Otherwise (used as type annotation), return the underlying type
    return val._type


@typeof_impl.register(UnionInstance)
def typeof_union_instance(val, c):
    """
    Type inference for UnionInstance objects.

    This allows passing union instances created on the host to JIT-compiled kernels.
    """
    return val._union_type


@typeof_impl.register(UnionTypeWrapper)
def typeof_union_type_wrapper(val, c):
    """
    Type inference for UnionTypeWrapper.

    When used as a value (e.g., called in device code), we need to return
    a Function type with the registered template so the typing system can
    resolve calls to it.
    """
    # If used in device code (has a constructor template), return Function type
    if val._constructor_template is not None:
        return numba_types.Function(template=val._constructor_template)

    # Otherwise (used as type annotation), return the underlying type
    return val._type
