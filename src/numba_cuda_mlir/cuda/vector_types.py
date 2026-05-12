# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA vector type stubs (float32x4, int32x2, etc.)

These are user-facing objects that can be used in CUDA kernels to construct
vector types. They map to numba_cuda_mlir's VectorType internally.
"""

import itertools
import numpy as np
from collections import defaultdict

from numba_cuda_mlir.type_defs.vector_types import VectorType
from numba_cuda_mlir import types

BASE_TYPE_MAP = {
    "int8": types.int8,
    "int16": types.int16,
    "int32": types.int32,
    "int64": types.int64,
    "uint8": types.uint8,
    "uint16": types.uint16,
    "uint32": types.uint32,
    "uint64": types.uint64,
    "float32": types.float32,
    "float64": types.float64,
}


def make_vector_types():
    """Create instances for all CUDA vector types."""
    vector_types = []
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

    for prefix, nelem in itertools.product(vector_type_prefix, vector_type_element_counts):
        base_type = BASE_TYPE_MAP[prefix]
        vec_type = VectorType(base_type, nelem)
        vec_type._base_type_name = prefix  # Keep for alias mapping
        vec_type._num_elements = nelem
        vec_type.__name__ = f"{prefix}x{nelem}"
        vec_type.aliases = []
        vector_types.append(vec_type)

    return vector_types


def map_vector_types_to_alias(vector_types):
    """Create C-compatible aliases for vector types (e.g., float4 -> float32x4)."""
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
    for vec_type in vector_types:
        base_type_to_vector_type[vec_type._base_type_name].append(vec_type)

    for alias, base_type in base_type_to_alias.items():
        types_for_base = base_type_to_vector_type[base_type]
        for vec_type in types_for_base:
            nelem = vec_type._num_elements
            vec_type.aliases.append(f"{alias}{nelem}")


# Create all vector types
_vector_types = make_vector_types()
map_vector_types_to_alias(_vector_types)

# Build lookup dictionaries
vector_types_by_name = {vec_type.__name__: vec_type for vec_type in _vector_types}
vector_types_by_alias = {}
for vec_type in _vector_types:
    for alias in vec_type.aliases:
        vector_types_by_alias[alias] = vec_type

# Export all types as module-level attributes
for vec_type in _vector_types:
    globals()[vec_type.__name__] = vec_type
    for alias in vec_type.aliases:
        globals()[alias] = vec_type

# List of all exported names
__all__ = list(vector_types_by_name.keys()) + list(vector_types_by_alias.keys())
