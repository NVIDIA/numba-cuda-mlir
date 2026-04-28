# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba.core.types import Type as NumbaType
from numba import types
import math
from cusimt._mlir import ir

_map_be_type_names_to_fe_types = dict()


class AggregateType(NumbaType):
    """
    Going from a numba type to an MLIR type is easy because llvm stores the types
    for us with llvm.StructType.get_identified. Going from the MLIR type back to the
    numba type is not as easy, so we keep a mapping here.

    Supports bitfields: fields can have (name, type, bit_width) where bit_width specifies
    the number of bits for a bitfield. Bitfields are packed into the smallest possible
    integer type.
    """

    def __init__(
        self,
        name: str,
        fields: list[tuple[str, NumbaType] | tuple[str, NumbaType, int | None]],
    ):
        super().__init__(name)
        # Normalize fields to 3-tuples (name, type, bit_width)
        normalized_fields = [
            (field[0], field[1], None) if len(field) == 2 else field for field in fields
        ]

        self.fields = normalized_fields
        self.__cusimt_pointer_cast__ = True

        # Storage size in bits when this struct is used as a union variant
        # This is set by union() to match C behavior where bitfields in a union
        # struct pack into the union's storage (not the first field's type)
        self.union_storage_bits = None

        # Calculate field layout (contains ALL fields with their bit offsets/widths)
        # This is the single source of truth for field layout information
        self.field_layout = self._calculate_field_layout()

        # This is used by code that checks "if struct.bitfield_info:" to determine
        # if it's a bitfield struct (single storage) vs regular struct (multiple fields)
        self.bitfield_info = {
            name: info for name, info in self.field_layout.items() if info.get("is_bitfield", False)
        }

        AggregateType.record_named_type(name, self)

    @property
    def is_bitfield_struct(self) -> bool:
        """
        Returns True if this struct contains any bitfield fields.

        Bitfield structs are represented in MLIR with a single storage field
        and use shift/mask operations for field access.

        Regular structs have separate MLIR fields for each struct field
        and use extractvalue/insertvalue operations.
        """
        return len(self.bitfield_info) > 0

    def get_bitfield_storage_type(self):
        """
        Determine the storage type for bitfield structs.

        For bitfield structs used as union variants, use the union's storage size.
        Otherwise, default to uint64.

        Returns:
            numba type (uint8/uint16/uint32/uint64)
        """
        if self.union_storage_bits is not None:
            # Use the union's storage size
            if self.union_storage_bits <= 8:
                return types.uint8
            elif self.union_storage_bits <= 16:
                return types.uint16
            elif self.union_storage_bits <= 32:
                return types.uint32
            elif self.union_storage_bits <= 64:
                return types.uint64
            else:
                return types.uint64  # Fallback
        else:
            # Standalone bitfield struct - use uint64
            return types.uint64

    def _calculate_field_layout(self):
        """
        Calculate bit offsets and widths for ALL struct fields (both bitfields and regular).

        Returns a dict: {field_name: {'bit_offset': int, 'bit_width': int, 'underlying_type': type, 'is_bitfield': bool, 'field_index': int}}

        For bitfield structs (fields with explicit bit_width):
        - Fields are packed into a single storage unit at MLIR field index 0
        - bit_offset is relative to the start of the storage
        - field_index is always 0 (the storage field)

        For regular structs (fields without bit_width):
        - Each field is a separate MLIR struct field
        - bit_offset tracks the cumulative bit position (useful for unions)
        - field_index is the position in the MLIR struct (used by insertvalue/extractvalue)

        Supports padding: field_name can be None to indicate anonymous padding bits (bitfields only).

        Current limitation: All bitfields must be at the beginning of the struct and
        must fit within a single 64-bit storage unit.
        """
        field_layout = {}
        bit_offset = 0
        has_seen_regular_field = False
        total_bitfield_bits = 0
        has_bitfields = False
        mlir_field_index = 0  # Track MLIR struct field index for insertvalue/extractvalue

        # For bitfield structs, we always use 64 bits of storage (uint64/i64)
        BITFIELD_STORAGE_BITS = 64

        for field_name, field_type, bit_width in self.fields:
            if bit_width is not None:
                # This is a bitfield (or padding if field_name is None)
                has_bitfields = True

                # Check: No regular fields before this bitfield
                if has_seen_regular_field:
                    field_desc = "padding" if field_name is None else f"field '{field_name}'"
                    raise NotImplementedError(
                        f"TODO: Implement this. Bitfield {field_desc} appears after a regular (non-bitfield) field. "
                        f"Current implementation requires all bitfields to be at the beginning "
                        f"of the struct with no regular fields in between.\n"
                        f"Struct: {self.name}"
                    )

                # Check: Bitfields don't overflow the 64-bit storage unit
                if bit_width <= 0:
                    raise ValueError(
                        f"Bitfield '{field_name}' in struct '{self.name}' has invalid bit_width: {bit_width}. "
                        f"bit_width must be positive."
                    )

                total_bitfield_bits += bit_width
                if total_bitfield_bits > BITFIELD_STORAGE_BITS:
                    field_desc = "padding" if field_name is None else f"field '{field_name}'"
                    raise NotImplementedError(
                        f"TODO: Implement this. Bitfield {field_desc} would overflow the storage unit. "
                        f"Total bitfield size ({total_bitfield_bits} bits) exceeds the "
                        f"storage limit ({BITFIELD_STORAGE_BITS} bits). "
                        f"Current implementation does not support bitfield overflow to "
                        f"multiple storage fields.\n"
                        f"Struct: {self.name}"
                    )

                # Store info only for non-padding fields (field_name is not None)
                # Padding just advances the bit_offset without creating an accessible field
                if field_name is not None:
                    field_layout[field_name] = {
                        "bit_offset": bit_offset,
                        "bit_width": bit_width,
                        "underlying_type": field_type,
                        "is_bitfield": True,
                        "field_index": 0,  # All bitfields are in storage field at index 0
                    }

                bit_offset += bit_width
            else:
                # Regular field (not a bitfield)
                if field_name is None:
                    raise ValueError(
                        f"Regular (non-bitfield) fields cannot have None as field name. "
                        f"None field names are only allowed for bitfield padding.\n"
                        f"Struct: {self.name}"
                    )
                has_seen_regular_field = True

                # Calculate bit width for regular field
                if isinstance(field_type, AggregateType):
                    # Nested struct
                    field_bits = field_type.size_bits
                elif isinstance(field_type, types.UniTuple):
                    # Fixed-size array
                    elem_bits = (
                        field_type.dtype.bitwidth if hasattr(field_type.dtype, "bitwidth") else 64
                    )
                    field_bits = field_type.count * elem_bits
                elif hasattr(field_type, "bitwidth"):
                    field_bits = field_type.bitwidth
                else:
                    # Conservative estimate for unknown types
                    field_bits = 64

                # Track layout for regular fields too
                field_layout[field_name] = {
                    "bit_offset": bit_offset,
                    "bit_width": field_bits,
                    "underlying_type": field_type,
                    "is_bitfield": False,
                    "field_index": mlir_field_index,  # MLIR struct field index for insertvalue/extractvalue
                }

                bit_offset += field_bits
                mlir_field_index += 1  # Each regular field gets its own MLIR struct field

        # For bitfield structs, increment mlir_field_index once for the storage field
        if has_bitfields and mlir_field_index == 0:
            mlir_field_index = 1

        return field_layout

    def _get_type_size_bits(self, numba_type):
        # Map Numba integer types to their bit sizes
        type_size_map = {
            NumbaType.uint8: 8,
            NumbaType.int8: 8,
            NumbaType.uint16: 16,
            NumbaType.int16: 16,
            NumbaType.uint32: 32,
            NumbaType.int32: 32,
            NumbaType.uint64: 64,
            NumbaType.int64: 64,
        }

        if numba_type in type_size_map:
            return type_size_map[numba_type]

        # Fallback: try to get bitwidth attribute if it exists
        if hasattr(numba_type, "bitwidth"):
            return numba_type.bitwidth

        raise ValueError(
            f"Cannot determine bit size for type {numba_type}. "
            f"Bitfields require integer types (uint8, uint16, uint32, uint64, etc.)"
        )

    @property
    def size_bits(self) -> int:
        """
        Calculate the size in bits of this struct.

        For bitfield structs: returns the total bits used, rounded to power of 2 (8, 16, 32, 64).
        For regular structs: sums the sizes of all fields.
        """
        if self.is_bitfield_struct:
            # Bitfield struct: calculate the actual bits used (including padding)
            max_bit_end = sum(bit_width for _, _, bit_width in self.fields if bit_width is not None)
            # Round up to nearest power of 2 (minimum 8)
            return max(8, 2 ** math.ceil(math.log2(max_bit_end)))
        else:
            # Regular struct: sum field sizes (using field_layout)
            total_bits = sum(info["bit_width"] for info in self.field_layout.values())
            return total_bits

    @staticmethod
    def record_named_type(name: str, fe_type: NumbaType):
        """
        Record a named type in the global registry.

        Returns the registered type (either newly registered or existing).
        If a type with the same name exists, returns it (idempotent behavior).
        """
        if name in _map_be_type_names_to_fe_types:
            existing_type = _map_be_type_names_to_fe_types[name]
            # Return existing type for idempotent behavior
            # This allows modules to be imported multiple times without errors
            return existing_type
        _map_be_type_names_to_fe_types[name] = fe_type
        return fe_type

    @staticmethod
    def get_named_type(name: str) -> NumbaType | None:
        return _map_be_type_names_to_fe_types.get(name, None)


class UnionType(NumbaType):
    """
    C-style union type where all variants share the same memory location.

    A union is represented as an integer in MLIR of the size of the largest variant.
    Access to variants is done via bitcasting.

    Variants are stored as (name, type) tuples similar to struct fields.
    """

    def __init__(self, name: str, variants: list[tuple[str, NumbaType]]):
        super().__init__(name)
        self.variants = variants
        self.__cusimt_pointer_cast__ = True
        AggregateType.record_named_type(name, self)

    def get_variant_type(self, variant_name: str) -> NumbaType | None:
        """Get the type of a specific variant by name."""
        for name, vtype in self.variants:
            if name == variant_name:
                return vtype
        return None

    @property
    def size_bits(self) -> int:
        """
        Calculate the storage size in bits for this union (the maximum of all variants).

        This matches C behavior where the union size is the size of its largest member.
        """
        max_bits = 0

        for _, variant_type in self.variants:
            # Get the size of this variant
            if hasattr(variant_type, "bitwidth"):
                bits = variant_type.bitwidth
            elif isinstance(variant_type, AggregateType):
                # For structs, use their size_bits property
                bits = variant_type.size_bits
            elif isinstance(variant_type, types.UniTuple):
                # UniTuple: count * element_size
                elem_bits = (
                    variant_type.dtype.bitwidth if hasattr(variant_type.dtype, "bitwidth") else 64
                )
                bits = variant_type.count * elem_bits
            else:
                # Default for unknown types
                bits = 64

            max_bits = max(max_bits, bits)

        return max_bits
