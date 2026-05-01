# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MLIR lowering support for union operations (getattr, setattr)
"""

from numba_cuda_mlir.type_defs.aggregate_types import UnionType, AggregateType
from numba_cuda_mlir.descriptor import MLIRTargetContext
from numba_cuda_mlir.mlir_lowering_registry import MLIRLoweringRegistry

registry = MLIRLoweringRegistry()
from numba_cuda_mlir.logging import trace
from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir._mlir.dialects import llvm, arith
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir.numba_cuda.core import ir as numba_ir
from numba_cuda_mlir.numba_cuda import types as numba_types


def _lower_union_construction_impl(
    context,  # Unused, kept for compatibility
    builder: "MLIRLower",
    target: numba_ir.Var,
    args: list[numba_ir.Var],
    kwargs: dict,
):
    """
    Lower union construction: MyUnion() creates a zero-initialized union value.

    A union is stored as an integer of the appropriate size (matching the largest variant).
    Access to variants is done via bitcasting.
    """
    trace(
        "Lowering union construction: target=%s, args=%s, kwargs=%s",
        target,
        args,
        kwargs,
    )

    # Get the union type being constructed
    target_type = builder.get_numba_type(target.name)
    trace("Target type: %s, type class: %s", target_type, type(target_type))

    # Get the MLIR type for the union storage
    mlir_union_type = builder.get_mlir_type(target_type)

    # Create a zero-initialized union value
    zero_union = llvm.mlir_zero(res=mlir_union_type)

    builder.store_var(target, zero_union)
    trace("Created zero-initialized union of type %s in %s", target_type, target.name)


@registry.lower(UnionType)
def lower_cunion_type_call(
    builder: "MLIRLower",
    target: numba_ir.Var,
    args: list[numba_ir.Var],
    kwargs: dict,
):
    """Lower union construction when UnionType is called directly."""
    return _lower_union_construction_impl(None, builder, target, args, kwargs)


@registry.lower_getattr_generic(UnionType)
def lower_union_type_getattr(
    context: MLIRTargetContext,
    builder: "MLIRLower",
    target: numba_ir.Var,
    value: numba_ir.Var,
    attr: str,
):
    """
    Lower getattr for union variant access.

    For unions, we bitcast the storage to the appropriate variant type.
    For struct variants, we extract bits and manually construct the struct.
    """

    union_type = builder.get_numba_type(value.name)
    trace("Lowering getattr for UnionType: %s.%s", union_type.name, attr)

    # Find the variant type
    variant_type = union_type.get_variant_type(attr)
    if variant_type is None:
        raise AttributeError(
            f"Union type '{union_type.name}' has no variant '{attr}'. "
            f"Available variants: {[v[0] for v in union_type.variants]}"
        )

    # Load the union storage
    union_value = builder.load_var(value)
    trace("Union storage type: %s", union_value.type)

    # Get the MLIR type for the variant
    variant_mlir_type = builder.get_mlir_type(variant_type)

    # Check if variant is a struct (aggregate type)
    if isinstance(variant_type, AggregateType):
        # For structs, we extract bits and construct the struct manually
        # Start with a zero-initialized struct
        variant_value = llvm.mlir_zero(res=variant_mlir_type)

        # Extract each field from the union integer
        # Separate handling for bitfield vs non-bitfield structs to avoid offset confusion

        if variant_type.is_bitfield_struct:
            # Bitfield struct: struct has a single storage field at position 0
            # Just insert the union value directly as the storage field

            # Get the storage type for this bitfield struct
            storage_type = variant_type.get_bitfield_storage_type()

            storage_mlir_type = builder.get_mlir_type(storage_type)

            # Convert union value to storage type if needed
            storage_value = convert(union_value, storage_mlir_type)

            # Insert storage value at position 0 (the only field)
            variant_value = llvm.insertvalue(
                container=variant_value,
                value=storage_value,
                position=mlir_ir.DenseI64ArrayAttr.get([0]),
            )
        else:
            # Regular (non-bitfield) struct: extract fields from union storage
            # Use field_layout (single source of truth for all field information)
            for field_name, field_info in variant_type.field_layout.items():
                # Get field info from field_layout
                field_index = field_info["field_index"]
                field_type = field_info["underlying_type"]
                field_bit_offset = field_info["bit_offset"]
                field_bits = field_info["bit_width"]

                field_mlir_type = builder.get_mlir_type(field_type)
                trace(
                    "Using field_layout for '%s': offset=%s, width=%s, index=%s",
                    field_name,
                    field_bit_offset,
                    field_bits,
                    field_index,
                )

                # Shift right to get field bits at LSB
                if field_bit_offset > 0:
                    shift_amount = arith.constant(
                        builder.get_mlir_type(union_type), field_bit_offset
                    )
                    shifted = arith.shrui(union_value, shift_amount)
                else:
                    shifted = union_value

                # Mask to extract only the field bits
                mask_value = (1 << field_bits) - 1
                mask = arith.constant(builder.get_mlir_type(union_type), mask_value)
                masked = arith.andi(shifted, mask)

                # Convert to field type if needed
                field_value = convert(masked, field_mlir_type)

                # Insert into struct using field_index from field_layout
                variant_value = llvm.insertvalue(
                    container=variant_value,
                    value=field_value,
                    position=mlir_ir.DenseI64ArrayAttr.get([field_index]),
                )

                trace(
                    "Extracted field '%s' at bit offset %s",
                    field_name,
                    field_bit_offset,
                )
    elif isinstance(variant_type, numba_types.UniTuple):
        # For UniTuple (fixed-size array), extract each element from the union storage
        trace(
            "Extracting UniTuple variant: element_type=%s, count=%s",
            variant_type.dtype,
            variant_type.count,
        )

        element_type = variant_type.dtype
        element_count = variant_type.count
        element_mlir_type = builder.get_mlir_type(element_type)

        # Get element size in bits
        if hasattr(element_type, "bitwidth"):
            element_bits = element_type.bitwidth
        else:
            raise NotImplementedError(
                f"Cannot determine bit size for UniTuple element type {element_type}"
            )

        # Extract each element
        elements = []
        union_mlir_type = builder.get_mlir_type(union_type)

        for i in range(element_count):
            bit_offset = i * element_bits

            # Shift right to get element bits at LSB
            if bit_offset > 0:
                shift_amount = arith.constant(union_mlir_type, bit_offset)
                shifted = arith.shrui(union_value, shift_amount)
            else:
                shifted = union_value

            # Mask to extract only the element bits
            mask_value = (1 << element_bits) - 1
            mask = arith.constant(union_mlir_type, mask_value)
            masked = arith.andi(shifted, mask)

            # Convert to element type if needed
            element_value = convert(masked, element_mlir_type)

            elements.append(element_value)

        # Return as Python tuple - this is how UniTuple is represented in the builder
        variant_value = tuple(elements)
        trace("Built UniTuple with %s elements", len(elements))
    else:
        variant_value = convert(union_value, variant_mlir_type)

    # Store the casted value
    builder.store_var(target, variant_value)
    trace("Extracted union variant '%s' and stored in %s", attr, target.name)
