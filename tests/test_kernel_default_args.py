# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A launch that omits a defaulted kernel parameter gets the default.

The launch passes only the supplied arguments, so an omitted default is
compiled as ``types.Omitted``: it is not a kernel parameter, and the lowering
materializes the default inside the kernel.
"""

import importlib
import sys

import numpy as np
import pytest

from numba_cuda_mlir import cuda, types

N = 8
I = np.arange(N)

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")


def _launch(kernel, *args, dtype=np.int32):
    out = cuda.to_device(np.zeros(N, dtype=dtype))
    kernel[1, N](out, *args)
    cuda.synchronize()
    return out.copy_to_host()


def test_omitted_default_is_used():
    @cuda.jit
    def k(out, scale=3):
        i = cuda.threadIdx.x
        out[i] = i * scale

    np.testing.assert_array_equal(_launch(k), I * 3)


def test_omitted_and_passed_defaults_compile_separately():
    @cuda.jit
    def k(out, scale=3):
        i = cuda.threadIdx.x
        out[i] = i * scale

    np.testing.assert_array_equal(_launch(k), I * 3)
    np.testing.assert_array_equal(_launch(k, 5), I * 5)
    np.testing.assert_array_equal(_launch(k), I * 3)

    assert sorted(str(sig[1]) for sig in k.overloads) == ["int64", "omitted(default=3)"]


def test_trailing_defaults_are_omitted_independently():
    @cuda.jit
    def k(out, a=1, b=2):
        i = cuda.threadIdx.x
        out[i] = i * a + b

    np.testing.assert_array_equal(_launch(k), I + 2)
    np.testing.assert_array_equal(_launch(k, 5), I * 5 + 2)
    np.testing.assert_array_equal(_launch(k, 5, 7), I * 5 + 7)


def test_float_and_bool_defaults():
    @cuda.jit
    def k(out, offset=0.5, negate=False):
        i = cuda.threadIdx.x
        value = i + offset
        out[i] = -value if negate else value

    np.testing.assert_array_equal(_launch(k, dtype=np.float64), I + 0.5)
    np.testing.assert_array_equal(_launch(k, 0.25, True, dtype=np.float64), -(I + 0.25))


def test_annotated_default_is_omitted():
    @cuda.jit
    def k(out, scale: types.int32 = 3):
        i = cuda.threadIdx.x
        out[i] = i * scale

    np.testing.assert_array_equal(_launch(k), I * 3)
    np.testing.assert_array_equal(_launch(k, 4), I * 4)


def test_default_before_star_args():
    @cuda.jit
    def k(out, base=4, *rest):
        i = cuda.threadIdx.x
        out[i] = base + len(rest)

    np.testing.assert_array_equal(_launch(k), np.full(N, 4))
    np.testing.assert_array_equal(_launch(k, 1, 9, 9), np.full(N, 3))


_CACHED_KERNEL = """
from numba_cuda_mlir import cuda


@cuda.jit(cache=True)
def scaled(out, scale=3):
    i = cuda.threadIdx.x
    out[i] = i * scale
"""


def test_omitted_default_is_used_after_a_disk_cache_load(tmp_path, monkeypatch):
    module_name = "kernel_default_args_cached"
    (tmp_path / f"{module_name}.py").write_text(_CACHED_KERNEL)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        for _ in range(2):
            sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)
            np.testing.assert_array_equal(_launch(module.scaled), I * 3)
        # The second import is a fresh dispatcher, so its overload came from disk.
        assert sum(module.scaled._cache_hits.values()) == 1
    finally:
        sys.modules.pop(module_name, None)
