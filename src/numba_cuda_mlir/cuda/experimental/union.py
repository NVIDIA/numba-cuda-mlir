# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Union support for numba_cuda_mlir - similar to ctypes structures
"""

from collections import namedtuple, Counter
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.type_defs.aggregate_types import AggregateType, UnionType
from typing import Any
import ctypes
from .struct import StructTypeWrapper, StructInstance
from numba_cuda_mlir.lowering.union import _lower_union_construction_impl
from numba_cuda_mlir.descriptor import mlir_target
from numba_cuda_mlir.numba_cuda.typing.templates import ConcreteTemplate
from numba_cuda_mlir import typing
from numba_cuda_mlir.lowering.union import registry as lowering_registry

# Counter for generating unique union names if no name is provided
_anon_union_counter = 0


class UnionInstance:
    """
    Runtime instance of a union that can be passed to kernels.

    C-style union where all variants share the same memory location.
    Any variant can be read or written at any time.
    """

    def __init__(self, union_type: UnionType, **kwargs):
        self._union_type = union_type
        self._variants = {}

        # Initialize all variants with default values
        for variant_name, variant_type in union_type.variants:
            self._variants[variant_name] = self._get_default_value(variant_type)

        # If specific variant values were provided, set them
        for variant_name, value in kwargs.items():
            if variant_name in self._variants:
                self._variants[variant_name] = value

    def _get_default_value(self, variant_type: types.Type) -> Any:
        """Get a default value for a variant type."""
        if isinstance(variant_type, types.Integer):
            return 0
        elif isinstance(variant_type, types.Float):
            return 0.0
        elif isinstance(variant_type, types.Boolean):
            return False
        elif isinstance(variant_type, AggregateType):
            # Nested struct
            return StructInstance(variant_type)
        else:
            return None

    def __getattr__(self, name: str) -> Any:
        """Allow accessing variants via dot notation."""
        if name.startswith("_"):
            # Private attributes go through normal lookup
            return object.__getattribute__(self, name)

        if name in self._variants:
            return self._variants[name]

        raise AttributeError(f"Union '{self._union_type.name}' has no variant '{name}'")

    def __setattr__(self, name: str, value: Any):
        """Allow setting variants via dot notation."""
        if name.startswith("_"):
            # Private attributes go through normal setattr
            object.__setattr__(self, name, value)
        elif hasattr(self, "_variants") and name in self._variants:
            # When setting a union variant, we update that variant
            # In C, this would overwrite the shared memory
            self._variants[name] = value
        else:
            object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        variant_strs = [f"{name}=..." for name, _ in self._union_type.variants]
        return f"{self._union_type.name}(union of: {', '.join(variant_strs)})"


class UnionTypeWrapper:
    """
    Wrapper around UnionType that adds a constructor.

    This allows users to call `MyUnion(variant1=val1)` to create instances.
    """

    def __init__(self, union_type: UnionType):
        self._type = union_type
        self.name = union_type.name
        self.variants = union_type.variants
        self._constructor_template = None

    def __call__(self, **kwargs) -> UnionInstance:
        """Create an instance of this union."""
        return UnionInstance(self._type, **kwargs)

    def __repr__(self) -> str:
        return f"UnionType({self.name})"

    def __getattr__(self, name):
        return getattr(self._type, name)


def union(variants: list[tuple[str, types.Type]], name: str | None = None) -> UnionTypeWrapper:
    """
    Create a union type for use in CUDA kernels.

    This creates a C-style union where all variants share the same memory location.
    Any variant can be read or written at any time.

    Args:
        variants: List of tuples specifying variant names and types.
        name: Optional name for the union type. If not provided, generates a unique anonymous name.

    Returns:
        A UnionTypeWrapper that can be called to create instances

    Example:
        >>> from numba_cuda_mlir import union,
        >>> from numba import types
        >>>
        >>> MyUnion = union(
        ...     "MyUnion",
        ...     [
        ...         ("as_int", types.uint32),
        ...         ("as_float", types.float32),
        ...     ],
        ... )
        >>>
        >>> # Create an instance
        >>> u = MyUnion(as_int=42)
        >>> print(u.as_int)
        42
    """
    # Generate unique name for anonymous unions to avoid registry collisions
    if name is None:
        global _anon_union_counter
        name = f"AnonymousUnion_{_anon_union_counter}"
        _anon_union_counter += 1

    # Validate that variant names are unique
    variant_names = [variant[0] for variant in variants]
    if len(set(variant_names)) != len(variant_names):
        duplicates = [name for name, count in Counter(variant_names).items() if count > 1]
        raise ValueError(
            f"Union '{name}' has duplicate variant names: {duplicates}. "
            f"All variant names must be unique within a union definition."
        )

    # Convert  namedtuples to (name, type) tuples
    # Unwrap StructTypeWrapper if present to get the underlying type
    variant_tuples = []
    for variant in variants:
        variant_type = variant[1]
        # Unwrap StructTypeWrapper to get the actual AggregateType
        if isinstance(variant_type, StructTypeWrapper):
            variant_type = variant_type._type
        variant_tuples.append((variant[0], variant_type))

    # Create the UnionType first
    union_type = UnionType(name, variant_tuples)

    # Get the union's storage size (maximum of all variants)
    # This will be used as the storage type for bitfield structs
    union_storage_bits = union_type.size_bits

    # Propagate storage size to bitfield struct variants
    # This matches C behavior where bitfields in a union struct pack into the union's storage
    for _variant_name, variant_type in variant_tuples:
        if isinstance(variant_type, AggregateType) and variant_type.is_bitfield_struct:
            # This is a bitfield struct - set its union storage size
            variant_type.union_storage_bits = union_storage_bits

    # Wrap it so it can be called to create instances
    wrapper = UnionTypeWrapper(union_type)

    # Register the constructor as callable in device code
    _register_union_constructor(wrapper, union_type)

    return wrapper


def _register_union_constructor(wrapper: UnionTypeWrapper, union_type: UnionType):
    """
    Register a union constructor so it can be called in device code.

    This allows `MyUnion()` to work inside kernels, where it will allocate
    the union and return it.
    """

    typingctx = mlir_target.typing_context
    sig = typing.signature(union_type)

    # Create a template that makes this wrapper callable
    class union_constructor_template(ConcreteTemplate):
        key = wrapper
        cases = [sig]

    wrapper._constructor_template = union_constructor_template

    typingctx.insert_user_function(wrapper, union_constructor_template)

    @lowering_registry.lower(wrapper)
    def lower_union_wrapper_call(builder, target, args, kwargs):
        """Lower calls to this specific union constructor."""
        return _lower_union_construction_impl(None, builder, target, args, kwargs)
