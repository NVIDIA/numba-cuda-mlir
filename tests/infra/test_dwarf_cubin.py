# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end DWARF tests via cuobjdump on compiled cubins.

- default jit: no .debug_line, no .debug_info
- lineinfo=True: .debug_line present
- debug=True: .debug_info present (full DWARF with types and variables)
"""

import os
import shutil
import subprocess
import tempfile
from functools import lru_cache

import numpy as np
from cuda.pathfinder import find_nvidia_binary_utility
from numba_cuda_mlir import cuda
from numba_cuda_mlir import types, compiler, testing
import pytest


@lru_cache(maxsize=1)
def _find_cuobjdump() -> str | None:
    """Resolve cuobjdump path once; return None if not found."""
    try:
        from numba_cuda_mlir.tools import get_cuda_toolkit_path

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
        pytest.skip("cuobjdump not found (set CUDA_HOME or ensure cuobjdump is on PATH).")
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
                f"cuobjdump exited with {cp.returncode}\nstdout: {cp.stdout}\nstderr: {cp.stderr}"
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
    assert ".debug_line" not in out, (
        f"Default jit should not emit .debug_line section; cuobjdump output:\n{out}"
    )


def test_cubin_line_table():
    """With lineinfo=True, the cubin contains a .debug_line section."""

    @cuda.jit(lineinfo=True, opt_level=3)
    def k(x: cuda.DeviceNDArray):
        x[0] = x[0] + 1

    out = _compile_and_dump_dwarf(k)
    assert ".debug_line" in out, (
        f"cuobjdump output should contain .debug_line section when lineinfo=True; output:\n{out}"
    )


def test_cubin_no_debug_info_default():
    """Default jit must not emit a .debug_info section."""

    @cuda.jit(opt_level=3)
    def k(x: cuda.DeviceNDArray):
        x[0] = x[0] + 1

    out = _compile_and_dump_dwarf(k)
    assert ".debug_info" not in out, (
        f"Default jit should not emit .debug_info section; cuobjdump output:\n{out}"
    )


def test_cubin_debug_info():
    """With debug=True, the cubin contains a .debug_info section (full DWARF)."""

    @cuda.jit(debug=True, opt=False)
    def k(out: cuda.DeviceNDArray, a, b):
        out[0] = a + b

    out = _compile_and_dump_dwarf(k, types.void(types.int32[:], types.int32, types.int32))
    assert ".debug_info" in out, (
        f"cuobjdump output should contain .debug_info section when debug=True; output:\n{out}"
    )


def _dwarf_dies(kernel, sig, cc) -> str:
    """Cross-compile a kernel for cc and return its decoded DWARF DIE tree.

    cuobjdump only reports which sections exist, so the DIEs come from
    llvm-dwarfdump on a cubin that ptxas assembled with debugging enabled.
    """
    # Pathfinder scans wheel layouts ahead of any system toolkit, which is what
    # this needs: a system toolkit can be far older than the one the package
    # targets, and too old to know the architectures under test.
    ptxas = find_nvidia_binary_utility("ptxas") or shutil.which("ptxas")
    if ptxas is None:
        pytest.skip("ptxas not found (set CUDA_HOME or ensure ptxas is on PATH).")
    dwarfdump = shutil.which("llvm-dwarfdump")
    if dwarfdump is None:
        pytest.skip("llvm-dwarfdump not found; needed to decode cubin DWARF DIEs.")

    ptx, _ = compiler.compile_ptx(kernel, sig, debug=True, opt=False, cc=cc)
    with tempfile.TemporaryDirectory() as tmpdir:
        ptx_path = os.path.join(tmpdir, "kernel.ptx")
        cubin_path = os.path.join(tmpdir, "kernel.cubin")
        with open(ptx_path, "w") as f:
            f.write(ptx)
        for cmd in (
            [ptxas, "-arch", f"sm_{cc[0]}{cc[1]}", "-g", "-O0", ptx_path, "-o", cubin_path],
            [dwarfdump, "--debug-info", cubin_path],
        ):
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if "is not defined for option 'gpu-name'" in cp.stderr:
                pytest.skip(f"ptxas is too old to target sm_{cc[0]}{cc[1]}.")
            if cp.returncode != 0:
                pytest.fail(
                    f"{os.path.basename(cmd[0])} exited with {cp.returncode}\n"
                    f"stdout: {cp.stdout}\nstderr: {cp.stderr}"
                )
        return cp.stdout


# sm_80 exercises the LLVM70 translator, sm_120 the upstream MLIR one.
@pytest.mark.parametrize("cc", [(8, 0), (12, 0)], ids=["sm_80", "sm_120"])
def test_cubin_dwarf_polymorphic_variant_part(cc):
    """A polymorphic local is a full DWARF variant part in the cubin.

    A debugger needs the variant part, its DW_AT_discr link to the member holding
    the active tag, and one DW_TAG_variant per type, to know which member to read
    a given value as. tests/test_debuginfo.py asserts this at the MLIR and LLVM IR
    levels; this is the assertion that it survives all the way into the cubin.
    """

    def k_poly_scalar_var(out, flag1, flag2):
        x = np.int64(1)
        if flag1:
            x = np.int32(2)
        elif flag2:
            x = np.float32(3.5)
        out[0] = x

    dump = _dwarf_dies(
        k_poly_scalar_var,
        types.void(types.float64[:], types.boolean, types.boolean),
        cc,
    )

    # DW_AT_discr has to resolve to the discriminator member DIE, so the check
    # captures the offset it points at and matches it against that DIE's own.
    testing.filecheck(
        """
        CHECK: DW_TAG_structure_type
        CHECK-NEXT: DW_AT_name ("variant_wrapper_struct")
        CHECK: DW_TAG_variant_part
        CHECK-NEXT: DW_AT_discr ([[DISCR:0x[0-9a-f]+]])
        CHECK: [[DISCR]]: DW_TAG_member
        CHECK-NEXT: DW_AT_name ("discriminator-float32.float64.int32.int64")
        CHECK: DW_TAG_variant
        CHECK-NEXT: DW_AT_discr_value (0x00)
        CHECK: DW_TAG_member
        CHECK-NEXT: DW_AT_name ("_float32")
        CHECK: DW_TAG_variant
        CHECK-NEXT: DW_AT_discr_value (0x01)
        CHECK: DW_TAG_member
        CHECK-NEXT: DW_AT_name ("_float64")
        CHECK: DW_TAG_variant
        CHECK-NEXT: DW_AT_discr_value (0x02)
        CHECK: DW_TAG_member
        CHECK-NEXT: DW_AT_name ("_int32")
        CHECK: DW_TAG_variant
        CHECK-NEXT: DW_AT_discr_value (0x03)
        CHECK: DW_TAG_member
        CHECK-NEXT: DW_AT_name ("_int64")
        """,
        dump,
    )
