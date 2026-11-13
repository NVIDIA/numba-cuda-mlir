# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda import types
import pytest


def test_torch_tensors():
    torch = pytest.importorskip("torch")
    x = torch.randn(10, 10, device="cuda")

    @cuda.jit(types.void(types.float32[:, :]))
    def k(x):
        x[0, 0] = 5

    k[1, 1](x)
    x = x.cpu()
    assert x[0][0] == 5


@pytest.mark.parametrize("dtype_name", ["float16", "float32", "float64"])
def test_torch_tensors_repeated_lazy_launch(dtype_name):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)

    @cuda.jit
    def increment(x):
        i = cuda.grid(1)
        if i < x.size:
            x[i] += 1

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        configured = increment[1, 64, cuda.external_stream(stream.cuda_stream)]
        x = torch.arange(37, device="cuda", dtype=dtype)
        configured(x)
        configured(x)
    stream.synchronize()
    torch.testing.assert_close(x.cpu(), torch.arange(37, dtype=dtype) + 2)


if __name__ == "__main__":
    test_torch_tensors()
