# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exported PTX must assemble and execute with the active CUDA toolchain."""

import re

import numpy as np
import pytest
from cuda.core import LaunchConfig, Stream, launch

from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.linker import Linker


@pytest.mark.parametrize("features", ["", "+ptx80"])
def test_exported_ptx_executes(features):
    @cuda.jit(features=features)
    def increment(x):
        x[0] = x[0] + 1

    ptx = increment.inspect_ptx(types.void(types.float32[:]))
    name = re.search(r"\.visible\s+\.entry\s+(\w+)", ptx).group(1)
    linker = Linker(lto=False)
    linker.add_ptx(ptx.encode())
    module = cuda.current_context().create_module_image(linker.complete())
    try:
        function = module.get_function(name)
        x = cuda.to_device(np.array([2.5], dtype=np.float32))
        launch(
            Stream.from_handle(0),
            LaunchConfig(grid=1, block=1),
            function.kernel,
            x.device_ctypes_pointer,
            x.device_ctypes_pointer,
            np.int64(0),
            np.int64(x.size),
            np.int64(1),
        )
        np.testing.assert_array_equal(x.copy_to_host(), [3.5])
    finally:
        module.unload()
