# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.simt as cs
from cusimt.numba_cuda import types
import pytest


def test_torch_tensors():
    torch = pytest.importorskip("torch")
    x = torch.randn(10, 10, device="cuda")

    @cs.jit(types.void(types.float32[:, :]))
    def k(x):
        x[0, 0] = 5

    k[1, 1](x)
    x = x.cpu()
    assert x[0][0] == 5


if __name__ == "__main__":
    test_torch_tensors()
