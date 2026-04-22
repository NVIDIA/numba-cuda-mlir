# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that numba.cuda and cuda.simt can coexist in any import order."""

import subprocess
import sys

import pytest


def _run_snippet(code: str):
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Subprocess failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return result.stdout


COMPILE_SNIPPET = """\
import {first}
import {second}
from numba import types

@numba_cuda.jit
def kernel(a, n):
    idx = numba_cuda.blockIdx.x * numba_cuda.blockDim.x + numba_cuda.threadIdx.x
    if idx < n:
        a[idx] = a[idx] + 1.0

sig = types.void(types.float32[::1], types.int64)
kernel.compile(sig)
print("OK")
"""


@pytest.mark.parametrize(
    "first,second",
    [
        ("numba.cuda as numba_cuda", "cuda.simt as cusimt_cuda"),
        ("cuda.simt as cusimt_cuda", "numba.cuda as numba_cuda"),
    ],
    ids=["numba_first", "cusimt_first"],
)
def test_numba_cuda_compile_with_both_imports(first, second):
    out = _run_snippet(COMPILE_SNIPPET.format(first=first, second=second))
    assert "OK" in out
