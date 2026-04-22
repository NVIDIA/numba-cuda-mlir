# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vector load/store operations for device code."""


def load(
    array: "MemrefLike",
    indices: tuple[int, ...] | int,
    shape: tuple[int, ...] | int,
    alignment: int | None = None,
) -> "Vector":
    """Load a vector from array at index.

    Args:
        array: Source array (memref) to load from
        indices: Starting index - can be int for 1D arrays or tuple for N-D arrays
        shape: Vector shape - int for 1D vectors or tuple for N-D vectors
        alignment: Optional memory alignment in bytes. When provided, generates
                   llvm.load with vector type and alignment attribute, which LLVM
                   optimizes to vectorized PTX instructions (e.g., ld.global.v4.b16).
                   Common values: 4, 8, 16 bytes.
                   Default None uses vector.transfer_read (may be scalarized).

    Returns:
        Vector of the specified shape loaded from the array

    Examples:
        # 1D vector load without alignment (uses vector.transfer_read)
        vec = cuda.vector.load(arr, i, 4)

        # 1D vector load with alignment (uses llvm.load → vectorized PTX)
        vec = cuda.vector.load(arr, i, 4, alignment=8)  # 4 x fp16 = 8 bytes

        # 2D array with 1D vector load
        vec = cuda.vector.load(arr_2d, (row, col), 4, alignment=16)

        # Alignment guidelines:
        # - 4 x fp16 (half): alignment=8  (4 * 2 bytes)
        # - 8 x fp16 (half): alignment=16 (8 * 2 bytes)
        # - 4 x fp32 (float): alignment=16 (4 * 4 bytes)
    """
    ...


def store(
    array: "MemrefLike",
    index: tuple[int, ...] | int,
    vector: "Vector",
    alignment: int | None = None,
) -> None:
    """Store a vector to array starting at index.

    Args:
        array: Destination array (memref) to store into
        index: Starting index - can be int for 1D arrays or tuple for N-D arrays
        vector: Vector value to store
        alignment: Optional memory alignment in bytes. When provided, generates
                   llvm.store with alignment attribute, which LLVM optimizes to
                   vectorized PTX instructions (e.g., st.global.v4.b16).
                   Must match alignment used in corresponding load.
                   Default None uses vector.transfer_write (may be scalarized).

    Examples:
        # 1D vector store without alignment
        cuda.vector.store(arr, i, vec)

        # 1D vector store with alignment (uses llvm.store → vectorized PTX)
        cuda.vector.store(arr, i, vec, alignment=8)  # 4 x fp16 = 8 bytes

        # 2D array with 1D vector store
        cuda.vector.store(arr_2d, (row, col), vec, alignment=16)

        # Note: alignment should match the actual memory alignment of the data.
    """
    ...
