# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for debugging CUDA kernels with cuda-gdb."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

import pytest

from numba_cuda_mlir import testing


def _cuda_gdb_binary() -> str:
    env = os.environ.get("NUMBA_CUDA_MLIR_GDB_BINARY", "")
    if env:
        return env
    on_path = shutil.which("cuda-gdb")
    return on_path or "cuda-gdb"


CUDA_GDB = _cuda_gdb_binary()

requires_cuda_gdb = pytest.mark.skipif(
    not os.path.isfile(CUDA_GDB),
    reason="cuda-gdb not found (set NUMBA_CUDA_MLIR_GDB_BINARY or add cuda-gdb to PATH)",
)


def _run_under_cuda_gdb(
    python_src: str,
    gdb_commands: str,
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Write *python_src* to a temporary file and invoke cuda-gdb in batch
    mode, passing each line of *gdb_commands* as a separate ``-ex`` argument.
    """
    ex_args = [
        arg
        for raw in gdb_commands.splitlines()
        if (line := raw.strip())
        for arg in ("-ex", line)
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="ncm_gdb_test_", delete=False
    ) as f:
        f.write(python_src)
        script = f.name
    try:
        return subprocess.run(
            [CUDA_GDB, "--batch", *ex_args, "--args", sys.executable, script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        os.unlink(script)


_KERNEL_SRC = textwrap.dedent("""\
    import numpy as np
    from numba_cuda_mlir import cuda

    @cuda.jit(debug=True, opt=False)
    def k(out, arg_a, arg_b):
        i = cuda.threadIdx.x
        sum = arg_a + arg_b
        out[i] = sum

    out = cuda.device_array(1, dtype=np.float32)
    k[1, 1](out, np.float32(1.5), np.float32(2.5))
    cuda.synchronize()
    assert out.copy_to_host()[0] == 4.0, out.copy_to_host()[0]
""")


@requires_cuda_gdb
def test_cuda_gdb_debug_kernel():
    """cuda-gdb hits the kernel launch breakpoint and can inspect locals."""
    gdb_commands = textwrap.dedent("""\
        set pagination off
        set cuda break_on_launch application
        run
        next
        next
        info locals
        print sum
        continue
        quit
    """)

    result = _run_under_cuda_gdb(_KERNEL_SRC, gdb_commands)
    assert result.returncode == 0, (
        f"cuda-gdb exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    testing.filecheck(
        """
        CHECK: CUDA kernel {{[0-9]+}}, grid {{[0-9]+}}, block {{[(][0-9,]+[)]}}, thread {{[(][0-9,]+[)]}}
        CHECK-DAG: arg_b = 2.5
        CHECK-DAG: arg_a = 1.5
        CHECK-DAG: sum = 4
        CHECK: $1 = 4
        """,
        result.stdout,
    )
