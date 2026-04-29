# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from cusimt.numba_cuda.cudadrv import devices, driver


def ndarray_device_allocate_data(ary):
    """
    Allocate gpu data buffer
    """
    datasize = driver.host_memory_size(ary)
    # allocate
    gpu_data = devices.get_context().memalloc(datasize)
    return gpu_data
