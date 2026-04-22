# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end DWARF tests via cuobjdump on compiled cubins.

- default jit: no .debug_line, no .debug_info
- lineinfo=True: .debug_line present
- debug=True: .debug_info present (full DWARF with types and variables)
"""

import os
import subprocess
import tempfile
from functools import lru_cache

import cuda.simt as cuda
from cuda.simt import types, compiler
import pytest


@lru_cache(maxsize=1)
def _find_cuobjdump() -> str | None:
    """Resolve cuobjdump path once; return None if not found."""
    try:
        from cusimt.tools import get_cuda_toolkit_path

        toolkit = get_cuda_toolkit_path()
        if toolkit:
            p = os.path.join(toolkit, "bin", "cuobjdump")
            if os.path.isfile(p):
                return p
    except Exception:
        pass
    return None


def _cuobjdump_path() -> str:
    """Return cached cuobjdump path or skip the test."""
    path = _find_cuobjdump()
    if path is None:
        pytest.skip(
            "cuobjdump not found (set CUDA_HOME or ensure cuobjdump is on PATH)."
        )
    return path


def _dwarf_dump(cubin: bytes) -> str:
    """Run `cuobjdump -elf` on cubin bytes; return combined stdout+stderr."""
    cuobjdump = _cuobjdump_path()
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        f.flush()
        path = f.name
    try:
        cp = subprocess.run(
            [cuobjdump, "-elf", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if cp.returncode != 0:
            pytest.fail(
                f"cuobjdump exited with {cp.returncode}\n"
                f"stdout: {cp.stdout}\nstderr: {cp.stderr}"
            )
        return cp.stdout + cp.stderr
    finally:
        os.unlink(path)


def _compile_and_dump_dwarf(kernel, sig=None) -> str:
    """Compile kernel to cubin, run cuobjdump -elf, return combined stdout+stderr."""
    sig = sig or types.void(types.float32[:])
    cubin = compiler.compile_cubin(kernel, sig)
    return _dwarf_dump(cubin)


def test_cubin_no_line_table():
    """Default jit must not emit a .debug_line section in the cubin."""

    @cuda.jit(opt_level=3)
    def k(x: cuda.DeviceNDArray):
        x[0] = x[0] + 1

    out = _compile_and_dump_dwarf(k)
    assert (
        ".debug_line" not in out
    ), f"Default jit should not emit .debug_line section; cuobjdump output:\n{out}"


def test_cubin_line_table():
    """With lineinfo=True, the cubin contains a .debug_line section."""

    @cuda.jit(lineinfo=True, opt_level=3)
    def k(x: cuda.DeviceNDArray):
        x[0] = x[0] + 1

    out = _compile_and_dump_dwarf(k)
    assert ".debug_line" in out, (
        f"cuobjdump output should contain .debug_line section when lineinfo=True; "
        f"output:\n{out}"
    )


def test_cubin_no_debug_info_default():
    """Default jit must not emit a .debug_info section."""

    @cuda.jit(opt_level=3)
    def k(x: cuda.DeviceNDArray):
        x[0] = x[0] + 1

    out = _compile_and_dump_dwarf(k)
    assert ".debug_info" not in out, (
        f"Default jit should not emit .debug_info section; " f"cuobjdump output:\n{out}"
    )


def test_cubin_debug_info():
    """With debug=True, the cubin contains a .debug_info section (full DWARF)."""

    @cuda.jit(debug=True, opt=False)
    def k(out: cuda.DeviceNDArray, a, b):
        out[0] = a + b

    out = _compile_and_dump_dwarf(
        k, types.void(types.int32[:], types.int32, types.int32)
    )
    assert ".debug_info" in out, (
        f"cuobjdump output should contain .debug_info section when debug=True; "
        f"output:\n{out}"
    )
