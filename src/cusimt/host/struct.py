# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Struct support for cusimt - similar to ctypes structures
"""

from collections import namedtuple, Counter
from numba.core import types
from cusimt.type_defs.aggregate_types import AggregateType
from typing import Any
import ctypes
from cusimt.descriptor import mlir_target
from numba.core.typing.templates import ConcreteTemplate
from cusimt import typing
from cusimt.lowering.struct import registry as lowering_registry
from cusimt.lowering.struct import _lower_struct_construction_impl

# Counter for generating unique anonymous struct names if no name is provided
_anon_struct_counter = 0


class StructInstance:
    """
    This class holds the actual field values and can be passed to
    JIT-compiled kernels. It's similar to a ctypes Structure instance.
    """

    def __init__(self, struct_type: AggregateType, **kwargs):
        self._struct_type = struct_type
        self._fields = {}

        # Initialize all fields from kwargs (skip padding fields where field_name is None)
        for field_name, field_type, *_ in struct_type.fields:
            if field_name is None:
                # Skip padding fields - they're not accessible
                continue
            if field_name in kwargs:
                value = kwargs[field_name]
                # TODO: Add type checking/conversion here
                self._fields[field_name] = value
            else:
                # Initialize with default value (0 or equivalent)
                self._fields[field_name] = self._get_default_value(field_type)

    def _get_default_value(self, field_type: types.Type) -> Any:
        """Get a default value for a field type."""
        if isinstance(field_type, types.Integer):
            return 0
        elif isinstance(field_type, types.Float):
            return 0.0
        elif isinstance(field_type, types.Boolean):
            return False
        else:
            return None

    def __getattr__(self, name: str) -> Any:
        """Allow accessing fields via dot notation."""
        if name.startswith("_"):
            # Private attributes go through normal lookup
            return object.__getattribute__(self, name)

        if name in self._fields:
            return self._fields[name]

        raise AttributeError(f"Struct '{self._struct_type.name}' has no field '{name}'")

    def __setattr__(self, name: str, value: Any):
        """Allow setting fields via dot notation."""
        if name.startswith("_"):
            # Private attributes go through normal setattr
            object.__setattr__(self, name, value)
        elif hasattr(self, "_fields") and name in self._fields:
            # TODO: Add type checking here
            self._fields[name] = value
        else:
            object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        field_strs = [
            f"{name}={self._fields[name]}"
            for name, _, _ in self._struct_type.fields
            if name is not None
        ]
        return f"{self._struct_type.name}({', '.join(field_strs)})"


class StructTypeWrapper:
    """
    Wrapper around AggregateType that adds a constructor.

    This allows users to call `MyStruct(field1=val1, ...)` to create instances.

    The underlying AggregateType can be accessed via the `_type` property,
    which is needed when passing the type to compile() or similar APIs.
    """

    def __init__(self, struct_type: AggregateType):
        self._type = struct_type  # Public property: access via MyStruct._type
        # Copy type attributes so this wrapper can be used as a type annotation
        self.name = struct_type.name
        self.fields = struct_type.fields
        # Will be set by _register_struct_constructor
        self._constructor_template = None

    def __call__(self, **kwargs) -> StructInstance:
        """Create an instance of this struct."""
        return StructInstance(self._type, **kwargs)

    def __repr__(self) -> str:
        return f"StructType({self.name})"

    # Make it work as a type for annotations and isinstance checks
    def __getattr__(self, name):
        return getattr(self._type, name)

    def __instancecheck__(self, instance):
        return isinstance(instance, self._type.__class__)

    def __subclasscheck__(self, subclass):
        return issubclass(subclass, self._type.__class__)


def struct(
    fields: list[tuple[str, types.Type, int | None]], name: str | None = None
) -> StructTypeWrapper:
    """
    Create a struct type for use in CUDA kernels.

    This creates a struct type that can be:
    - Modified in device functions (mutable)
    - Accessed via getattr/setattr
    - Support bitfields with specified bit widths
    - Support explicit padding in bitfields using None as field name

    Args:
        fields: List of tuples specifying field names, types, and optional bit widths.
                Use None as field name for padding bits in bitfields.
        name: Optional name for the struct type. If not provided, generates a unique anonymous name.

    Returns:
        A StructTypeWrapper that can be called to create instances

    Example:
        >>> from cusimt import struct,
        >>> from numba import types
        >>>
        >>> # Regular fields
        >>> MyStruct = struct(
        ...     "MyStruct",
        ...     [
        ...         ("a", types.int32),
        ...         ("b", types.float32),
        ...     ],
        ... )
        >>>
        >>> # Bitfields
        >>> MyBitfields = struct(
        ...     "MyBitfields",
        ...     [
        ...         ("flag", types.uint32, 1),  # 1-bit field
        ...         ("value", types.uint32, 5),  # 5-bit field
        ...     ],
        ... )
        >>>
        >>> # Bitfields with padding
        >>> MyPaddedBitfields = struct(
        ...     "MyPaddedBitfields",
        ...     [
        ...         ("a", types.uint32, 4),  # 4 bits
        ...         (None, types.uint32, 4),  # 4 bits padding
        ...         ("b", types.uint32, 8),  # 8 bits
        ...     ],
        ... )
    """
    # Generate unique name for anonymous structs to avoid registry collisions
    if name is None:
        global _anon_struct_counter
        name = f"AnonymousStruct_{_anon_struct_counter}"
        _anon_struct_counter += 1

    # Validate that field names are unique (excluding None for padding)
    field_names = [field[0] for field in fields if field[0] is not None]
    name_counts = Counter(field_names)
    duplicates = [name for name, count in name_counts.items() if count > 1]
    if duplicates:
        raise ValueError(
            f"Struct '{name}' has duplicate field names: {duplicates}. "
            f"All field names must be unique within a struct definition."
        )

    # Convert  namedtuples to (name, type, bit_width) tuples for AggregateType
    field_tuples = [(field[0], field[1], field[2] if len(field) > 2 else None) for field in fields]

    # Create the AggregateType
    struct_type = AggregateType(name, field_tuples)

    # Wrap it so it can be called to create instances
    wrapper = StructTypeWrapper(struct_type)

    # Register the constructor as callable in device code
    _register_struct_constructor(wrapper, struct_type)

    return wrapper


def _register_struct_constructor(wrapper: StructTypeWrapper, struct_type: AggregateType):
    """
    Register a struct constructor so it can be called in device code.

    This allows `MyStruct()` to work inside kernels, where it will allocate
    the struct on the stack and return it.
    """

    typingctx = mlir_target.typing_context
    targetctx = mlir_target.target_context

    # Create a signature for the constructor: () -> struct_type
    # The constructor takes no arguments and returns the struct type
    sig = typing.signature(struct_type)

    # Create a template that makes this wrapper callable
    class struct_constructor_template(ConcreteTemplate):
        key = wrapper
        cases = [sig]

    # Store the template in the wrapper so typeof_impl can access it
    wrapper._constructor_template = struct_constructor_template

    # Register it in the typing context
    typingctx.insert_user_function(wrapper, struct_constructor_template)

    @lowering_registry.lower(wrapper)
    def lower_struct_wrapper_call(builder, target, args, kwargs):
        """Lower calls to this specific struct constructor."""
        return _lower_struct_construction_impl(None, builder, target, args, kwargs)
