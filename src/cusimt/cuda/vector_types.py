# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA vector type stubs (float32x4, int32x2, etc.)

These are user-facing objects that can be used in CUDA kernels to construct
vector types. They map to cusimt's VectorType internally.
"""

import itertools
from inspect import Signature, Parameter
import numpy as np
from collections import defaultdict


class VectorTypeStub:
    """Base class for vector type stubs."""

    _base_type_name: str
    _num_elements: int
    _attr_names: tuple
    aliases: list

    def __new__(cls, *args):
        raise NotImplementedError(
            f"{cls.__name__} can only be used inside a CUDA kernel"
        )

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


def make_vector_type_stubs():
    """Create stub classes for all CUDA vector types."""
    vector_type_stubs = []
    vector_type_prefix = (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    )
    vector_type_element_counts = (1, 2, 3, 4)
    vector_type_attribute_names = ("x", "y", "z", "w")

    for prefix, nelem in itertools.product(
        vector_type_prefix, vector_type_element_counts
    ):
        type_name = f"{prefix}x{nelem}"
        attr_names = vector_type_attribute_names[:nelem]

        stub_class = type(
            type_name,
            (VectorTypeStub,),
            {
                "_base_type_name": prefix,
                "_num_elements": nelem,
                "_attr_names": attr_names,
                "_description_": f"<{type_name}>",
                "__signature__": Signature(
                    parameters=[
                        Parameter(name=attr_name, kind=Parameter.POSITIONAL_ONLY)
                        for attr_name in attr_names
                    ]
                ),
                "__doc__": f"Construct a {type_name} vector type in CUDA kernels.",
                "aliases": [],
            },
        )
        vector_type_stubs.append(stub_class)

    return vector_type_stubs


def map_vector_type_stubs_to_alias(vector_type_stubs):
    """Create C-compatible aliases for vector type stubs (e.g., float4 -> float32x4)."""
    base_type_to_alias = {
        "char": f"int{np.dtype(np.byte).itemsize * 8}",
        "short": f"int{np.dtype(np.short).itemsize * 8}",
        "int": f"int{np.dtype(np.intc).itemsize * 8}",
        "long": f"int{np.dtype(np.int_).itemsize * 8}",
        "longlong": f"int{np.dtype(np.longlong).itemsize * 8}",
        "uchar": f"uint{np.dtype(np.ubyte).itemsize * 8}",
        "ushort": f"uint{np.dtype(np.ushort).itemsize * 8}",
        "uint": f"uint{np.dtype(np.uintc).itemsize * 8}",
        "ulong": f"uint{np.dtype(np.uint).itemsize * 8}",
        "ulonglong": f"uint{np.dtype(np.ulonglong).itemsize * 8}",
        "float": f"float{np.dtype(np.single).itemsize * 8}",
        "double": f"float{np.dtype(np.double).itemsize * 8}",
    }

    base_type_to_vector_type = defaultdict(list)
    for stub in vector_type_stubs:
        base_type_to_vector_type[stub._base_type_name].append(stub)

    for alias, base_type in base_type_to_alias.items():
        stubs_for_base = base_type_to_vector_type[base_type]
        for stub in stubs_for_base:
            nelem = stub._num_elements
            stub.aliases.append(f"{alias}{nelem}")


# Create all vector type stubs
_vector_type_stubs = make_vector_type_stubs()
map_vector_type_stubs_to_alias(_vector_type_stubs)

# Build lookup dictionaries
vector_type_stubs_by_name = {stub.__name__: stub for stub in _vector_type_stubs}
vector_type_stubs_by_alias = {}
for stub in _vector_type_stubs:
    for alias in stub.aliases:
        vector_type_stubs_by_alias[alias] = stub

# Export all stubs as module-level attributes
for stub in _vector_type_stubs:
    globals()[stub.__name__] = stub
    for alias in stub.aliases:
        globals()[alias] = stub

# List of all exported names
__all__ = list(vector_type_stubs_by_name.keys()) + list(
    vector_type_stubs_by_alias.keys()
)
