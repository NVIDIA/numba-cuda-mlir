# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np


class TensorMapSwizzle:
    """Tensor map swizzle options."""

    pass


class TensorMapL2Promotion:
    """Tensor map L2 promotion options."""

    pass


class TensorMapFloatOOBFill:
    """Tensor map out-of-bounds fill options."""

    pass


class TensorMapInterleave:
    """Tensor map interleave options."""

    pass


def _init_tma_enums():
    """Initialize TMA enum values from driver API."""
    from cuda.bindings import driver

    TensorMapSwizzle.NONE = driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
    TensorMapSwizzle.SWIZZLE_128B = driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    TensorMapSwizzle.SWIZZLE_64B = driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B
    TensorMapSwizzle.SWIZZLE_32B = driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B

    TensorMapL2Promotion.NONE = driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE

    TensorMapFloatOOBFill.NONE = driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE

    TensorMapInterleave.NONE = driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE


# Initialize enums on import (lazy to avoid import-time errors if driver unavailable)
try:
    _init_tma_enums()
except Exception:
    # If driver not available at import time, will be initialized on first access
    pass


def _get_tensor_info(tensor):
    """
    Extract pointer and dtype from GPU arrays using __cuda_array_interface__.

    Returns: (ptr, np_dtype)
    """

    if hasattr(tensor, "__cuda_array_interface__"):
        interface = tensor.__cuda_array_interface__
        ptr = interface["data"][0]
        typestr = interface["typestr"]
        np_dtype = np.dtype(typestr)
        return ptr, np_dtype

    raise TypeError(
        f"Unsupported tensor type: {type(tensor)}. "
        f"Tensor must support __cuda_array_interface__ protocol. "
        f"Supported types: torch.Tensor (CUDA), cupy.ndarray, numba.cuda.DeviceNDArray"
    )


def create_tensor_map_from_tensor(
    tensor,
    box_dim,
    swizzle=None,
    transpose=False,
    l2promo=None,
    oob=None,
    interleave=None,
):
    """
    Create a TMA descriptor from a tensor with __cuda_array_interface__

    Parameters:
    - tensor: tensor with __cuda_array_interface__
    - box_dim: Tuple of (height, width) for the TMA box dimensions
    - swizzle: TensorMapSwizzle enum (default: SWIZZLE_128B)
    - transpose: Whether to transpose the tensor (default: False)
    - l2promo: L2 promotion kind (default: L2_PROMOTION_NONE)
    - oob: Out-of-bounds fill kind (default: FLOAT_OOB_FILL_NONE)
    - interleave: Interleave kind (default: INTERLEAVE_NONE)

    Returns:
    - CUtensorMap descriptor object
    """
    from cuda.bindings import driver

    # Extract tensor information (universal)
    ptr, np_dtype = _get_tensor_info(tensor)
    element_size = np_dtype.itemsize

    # Determine layout: smaller stride indicates the contiguous (inner) dimension
    tensor_shape = tensor.shape
    tensor_strides = tensor.stride()

    if tensor_strides[0] < tensor_strides[1]:
        # Transposed view: axis 0 is contiguous
        inner_dim, outer_dim = tensor_shape[0], tensor_shape[1]
        row_stride_elements = tensor_strides[1]
    else:
        # Row-major: axis 1 is contiguous
        inner_dim, outer_dim = tensor_shape[1], tensor_shape[0]
        row_stride_elements = tensor_strides[0]

    globalDim = [driver.cuuint64_t(inner_dim), driver.cuuint64_t(outer_dim)]
    globalStrides = [driver.cuuint64_t(row_stride_elements * element_size)]
    boxDim = [driver.cuuint32_t(box_dim[0]), driver.cuuint32_t(box_dim[1])]

    elemStrides = [driver.cuuint32_t(1), driver.cuuint32_t(1)]

    # Set defaults
    if swizzle is None:
        swizzle = TensorMapSwizzle.SWIZZLE_128B
    if l2promo is None:
        l2promo = driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE
    if oob is None:
        oob = driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    if interleave is None:
        interleave = driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE

    # Map dtype name to driver dtype (only types supported by TMA)
    dtype_to_driver_map = {
        "float16": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        "float32": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        "float64": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
        "int32": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
        "int64": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT64,
        "uint32": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32,
        "uint16": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
        "uint8": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        "uint64": driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64,
    }

    dtype_name = np_dtype.name
    if dtype_name not in dtype_to_driver_map:
        raise ValueError(
            f"Unsupported dtype: {dtype_name}. Supported types: {list(dtype_to_driver_map.keys())}"
        )

    data_type = dtype_to_driver_map[dtype_name]

    # Create TMA descriptor
    err, tma_desc = driver.cuTensorMapEncodeTiled(
        data_type,
        2,  # rank
        ptr,
        globalDim,
        globalStrides,
        boxDim,
        elemStrides,
        interleave,
        swizzle,
        l2promo,
        oob,
    )

    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to create TMA descriptor: {err}")

    return tma_desc
