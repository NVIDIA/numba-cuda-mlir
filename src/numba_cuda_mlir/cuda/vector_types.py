# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA vector type stubs (float32x4, int32x2, etc.)

These are user-facing objects that can be used in CUDA kernels to construct
vector types. They map to numba_cuda_mlir's VectorType internally.
"""

import itertools
import numpy as np

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
    "float16": types.float16,
    "float32": types.float32,
    "float64": types.float64,
}


def make_vector_types():
    """Create instances for all CUDA vector types."""
    vector_types_by_name = {}
    vector_type_prefix = (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    )
    vector_type_element_counts = (1, 2, 3, 4)

    for prefix, nelem in itertools.product(vector_type_prefix, vector_type_element_counts):
        base_type = BASE_TYPE_MAP[prefix]
        vec_type = VectorType(base_type, nelem)
        vector_types_by_name[vec_type.name] = vec_type

    return vector_types_by_name


def make_vector_type_aliases(vector_types_by_name):
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
        "half": "float16",
        "float": f"float{np.dtype(np.single).itemsize * 8}",
        "double": f"float{np.dtype(np.double).itemsize * 8}",
    }

    vector_types_by_alias = {}
    for alias_prefix, base_type_prefix in base_type_to_alias.items():
        for nelem in (1, 2, 3, 4):
            alias_name = f"{alias_prefix}{nelem}"
            target_name = f"{base_type_prefix}x{nelem}"
            if target_name in vector_types_by_name:
                vector_types_by_alias[alias_name] = vector_types_by_name[target_name]

    return vector_types_by_alias


# Build lookup dictionaries
vector_types_by_name = make_vector_types()
vector_types_by_alias = make_vector_type_aliases(vector_types_by_name)

# List of all unique vector types
_vector_types = list(vector_types_by_name.values())

# Export all types as module-level attributes
for name, vec_type in vector_types_by_name.items():
    globals()[name] = vec_type
for alias, vec_type in vector_types_by_alias.items():
    globals()[alias] = vec_type

# List of all exported names
__all__ = list(vector_types_by_name.keys()) + list(vector_types_by_alias.keys())
