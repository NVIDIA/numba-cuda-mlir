# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NUMBA_CUDA_MLIR Universal Function (ufunc) Database

This module provides a database mapping NumPy ufuncs to their MLIR lowering
implementations. When NumPy ufuncs are called on arrays, Numba routes them through
the 'arrayexpr' code path, which requires special handling to map the ufunc to
the appropriate MLIR code generator.

The database is lazily initialized to avoid circular imports and provides a simple
lookup mechanism for the arrayexpr lowering process.
"""

import numpy as np
import operator
from typing import Callable, Dict, Any, Optional

# Lazily initialized ufunc database
_ufunc_db: Optional[Dict[Any, Callable]] = None


def _lazy_init_db():
    """Initialize the ufunc database on first access to avoid circular imports."""
    global _ufunc_db

    if _ufunc_db is None:
        from numba_cuda_mlir.lowering import (
            numpy,
            builtins,
        )

        _ufunc_db = {}
        _ufunc_db.update(numpy.ufunc_registry.registry)
        _ufunc_db.update(builtins.ufunc_registry.registry)


def get_ufunc_lowering(ufunc) -> Optional[Callable]:
    """
    Get the MLIR lowering function for a given NumPy ufunc.

    Args:
        ufunc: A NumPy universal function object (e.g., np.abs, np.sqrt)

    Returns:
        The MLIR code generation function if the ufunc is supported, None otherwise.
        The returned function has the signature:
        (builder, target, args, kwargs) -> None

    Example:
        >>> lowering = get_ufunc_lowering(np.abs)
        >>> if lowering:
        >>>     lowering(builder, target, args, {})
    """
    _lazy_init_db()
    assert _ufunc_db is not None
    return _ufunc_db.get(ufunc)


def is_supported_ufunc(ufunc) -> bool:
    """
    Check if a NumPy ufunc is supported for MLIR lowering.

    Args:
        ufunc: A NumPy universal function object

    Returns:
        True if the ufunc has a registered MLIR lowering, False otherwise.
    """
    _lazy_init_db()
    assert _ufunc_db is not None
    return ufunc in _ufunc_db


def get_supported_ufuncs():
    """
    Get a list of all supported NumPy ufuncs.

    Returns:
        A list of NumPy ufunc objects that have MLIR lowerings registered.
    """
    _lazy_init_db()
    assert _ufunc_db is not None
    return list(_ufunc_db.keys())
