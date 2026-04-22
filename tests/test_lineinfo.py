# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test that lineinfo=True produces PTX with .file and .loc directives.

By default: PTX should have no .file/.loc directives.
With lineinfo=True: PTX must contain .file referencing source file and
.loc referencing line numbers in the kernel's (or device function's) line range.
"""

import inspect
import os

import cuda.simt as cuda
from cuda.simt import types, compiler, testing


def k(x: cuda.DeviceNDArray):
    x[0] = x[0] + 1


def device_add(a, b):
    return a + b


def test_ptx_no_lineinfo():
    """Without lineinfo, PTX should not contain .file or .loc directives."""
    ptx, _ = compiler.compile_ptx(k, types.void(types.float32[:]))
    assert ptx is not None
    assert ".file" not in ptx and ".loc" not in ptx


def test_ptx_lineinfo_directives():
    """With lineinfo=True, PTX must contain .file and .loc for this source."""
    ptx, _ = compiler.compile_ptx(
        k,
        types.void(types.float32[:]),
        lineinfo=True,
    )
    assert ptx is not None

    src_path = inspect.getsourcefile(k)
    assert src_path is not None
    src_name = os.path.basename(src_path)
    _, def_line = inspect.getsourcelines(k)
    stmt_line = def_line + 1

    testing.filecheck(
        f"""
        CHECK-DAG: .file\t[[file_id:[0-9]+]] "{{{{.*}}}}{src_name}"
        CHECK-DAG: .loc\t[[file_id]] {stmt_line}
        """,
        ptx,
    )


def test_ptx_lineinfo_device_function():
    """With lineinfo=True and device=True, PTX must contain .file and .loc
    referencing the device function's source."""
    ptx, _ = compiler.compile_ptx(
        device_add,
        types.float32(types.float32, types.float32),
        device=True,
        lineinfo=True,
    )
    assert ptx is not None

    src_path = inspect.getsourcefile(device_add)
    assert src_path is not None
    src_name = os.path.basename(src_path)
    _, def_line = inspect.getsourcelines(device_add)
    stmt_line = def_line + 1

    testing.filecheck(
        f"""
        CHECK-DAG: .file\t[[file_id:[0-9]+]] "{{{{.*}}}}{src_name}"
        CHECK-DAG: .loc\t[[file_id]] {stmt_line}
        """,
        ptx,
    )
