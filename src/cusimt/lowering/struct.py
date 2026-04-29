# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt.type_defs.aggregate_types import AggregateType
from cusimt.descriptor import MLIRTargetContext
from cusimt.mlir_lowering_registry import MLIRLoweringRegistry

registry = MLIRLoweringRegistry()
lower = registry.lower
from cusimt.logging import trace
from cusimt.lowering_utilities import convert
from cusimt._mlir.dialects import llvm, arith
from cusimt._mlir import ir as mlir_ir
from cusimt.numba_cuda.core import ir as numba_ir
from cusimt.numba_cuda import types


@registry.lower_getattr_generic(AggregateType)
def lower_aggregate_type_getattr(
    context: MLIRTargetContext,
    builder: "MLIRLower",
    target: numba_ir.Var,
    value: numba_ir.Var,
    attr: str,
):
    """
    Lower getattr for struct field access.

    For regular fields: Uses LLVM's extractvalue operation.
    For bitfields: Uses shift and mask operations to extract bits.
    """

    value_type = builder.get_numba_type(value.name)
    trace("Lowering getattr for AggregateType: %s.%s", value_type.name, attr)

    # Check if this is a bitfield
    if attr in value_type.field_layout and value_type.field_layout[attr].get(
        "is_bitfield", False
    ):
        # This is a bitfield - use shift/mask operations
        field_info = value_type.field_layout[attr]
        bit_offset = field_info["bit_offset"]
        bit_width = field_info["bit_width"]
        underlying_type = field_info["underlying_type"]

        trace(
            f"Bitfield access: {attr} at bit_offset={bit_offset}, bit_width={bit_width}"
        )

        # Load the struct value
        struct_value = builder.load_var(value)

        # Get the storage type for this bitfield struct
        storage_type = value_type.get_bitfield_storage_type()

        # Extract the storage field at position 0
        storage_field_value = llvm.extractvalue(
            res=builder.get_mlir_type(storage_type),
            container=struct_value,
            position=mlir_ir.DenseI64ArrayAttr.get([0]),
        )

        # Get the storage type
        storage_mlir_type = storage_field_value.type
        storage_bitwidth = storage_mlir_type.width

        # Shift right to align the bitfield at LSB
        if bit_offset > 0:
            shift_amount = arith.constant(storage_mlir_type, bit_offset)
            shifted = arith.shrui(storage_field_value, shift_amount)
        else:
            shifted = storage_field_value

        # Mask to extract only the bitfield bits
        mask_value = (1 << bit_width) - 1
        mask = arith.constant(storage_mlir_type, mask_value)
        masked = arith.andi(shifted, mask)

        # Convert to the target type size if needed (truncate or extend)
        target_mlir_type = builder.get_mlir_type(underlying_type)
        field_value = convert(masked, target_mlir_type)

        builder.store_var(target, field_value)
        trace(
            f"Extracted bitfield '{attr}' (offset={bit_offset}, width={bit_width}) into {target.name}"
        )
        return

    # Regular field (not a bitfield) - use field_layout
    if attr not in value_type.field_layout:
        available_fields = list(value_type.field_layout.keys())
        raise AttributeError(
            f"Struct type '{value_type.name}' has no field '{attr}'. "
            f"Available fields: {available_fields}"
        )

    # Get field info from field_layout (single source of truth)
    field_info = value_type.field_layout[attr]
    field_index = field_info["field_index"]
    field_type = field_info["underlying_type"]

    # Load the struct value
    struct_value = builder.load_var(value)
    trace(
        f"Struct value type: {struct_value.type}, extracting field at index {field_index}"
    )

    # Extract the field using LLVM extractvalue
    # extractvalue takes the struct and the position (field index)
    field_value = llvm.extractvalue(
        res=builder.get_mlir_type(field_type),
        container=struct_value,
        position=mlir_ir.DenseI64ArrayAttr.get([field_index]),
    )

    # Store the extracted field value
    builder.store_var(target, field_value)
    trace("Stored field '%s' (index %s) into %s", attr, field_index, target.name)


def _lower_struct_construction_impl(
    context,  # Unused, kept for compatibility
    builder: "MLIRLower",
    target: numba_ir.Var,
    args: list[numba_ir.Var],
    kwargs: dict,
):
    """
    Lower struct construction: MyStruct() creates a zero-initialized struct value (inline).

    This creates the struct as an SSA value. The compiler will decide whether to keep it
    in registers or spill to stack. Fields are set via subsequent setattr operations using
    LLVM insertvalue, which works on SSA values.
    """
    trace(
        f"Lowering struct construction: target={target}, args={args}, kwargs={kwargs}"
    )

    # Get the struct type being constructed
    target_type = builder.get_numba_type(target.name)

    # Get the MLIR type for the struct
    mlir_struct_type = builder.get_mlir_type(target_type)

    # Create a zero-initialized struct value (inline constructor)
    # This is a pure SSA value - no alloca needed since we're not taking its address
    zero_struct = llvm.mlir_zero(res=mlir_struct_type)

    builder.store_var(target, zero_struct)
    trace(
        f"Created zero-initialized struct of type {target_type} in {target.name} (inline)"
    )


@registry.lower(AggregateType)
def lower_aggregate_type_call(
    builder: "MLIRLower",
    target: numba_ir.Var,
    args: list[numba_ir.Var],
    kwargs: dict,
):
    """Lower struct construction when AggregateType is called directly."""
    return _lower_struct_construction_impl(None, builder, target, args, kwargs)
