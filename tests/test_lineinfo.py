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

from numba_cuda_mlir import cuda
from numba_cuda_mlir import types, compiler, testing

from lineinfo_usecases import helper_scale_offset


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


def kernel_calling_helper(x):
    x[0] = helper_scale_offset(x[0])


def _check_multi_file_ptx(ptx):
    kernel_name = os.path.basename(inspect.getsourcefile(kernel_calling_helper))
    helper_name = os.path.basename(inspect.getsourcefile(helper_scale_offset.py_func))

    # The exact surviving body line depends on optimization, so only
    # require that some line is attributed to the helper's file, and
    # that the kernel's own lines still reference the kernel's file.
    testing.filecheck(
        f"""
        CHECK-DAG: .file\t[[kernel_id:[0-9]+]] "{{{{.*}}}}{kernel_name}"
        CHECK-DAG: .file\t[[helper_id:[0-9]+]] "{{{{.*}}}}{helper_name}"
        CHECK-DAG: .loc\t[[helper_id]] {{{{[0-9]+}}}}
        CHECK-DAG: .loc\t[[kernel_id]] {{{{[0-9]+}}}}
        """,
        ptx,
    )


def test_ptx_lineinfo_multiple_source_files():
    """Code inlined from another file gets its own .file entry.

    Lines from an inlined device function must be attributed to that
    function's source file (via a DILexicalBlockFile scope), not
    mis-filed under the calling kernel's file.
    """
    ptx, _ = compiler.compile_ptx(
        kernel_calling_helper,
        types.void(types.float32[:]),
        lineinfo=True,
    )
    assert ptx is not None
    _check_multi_file_ptx(ptx)


def test_ptx_debug_multiple_source_files():
    """Full debug info attributes cross-file lines the same way lineinfo does."""
    ptx, _ = compiler.compile_ptx(
        kernel_calling_helper,
        types.void(types.float32[:]),
        debug=True,
        opt=False,
    )
    assert ptx is not None
    _check_multi_file_ptx(ptx)
