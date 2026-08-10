# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regression tests for #170. Subprocess-based because symbol isolation
# is a process-level property established at first-load time.
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Symbol isolation is Linux-specific (RTLD_DEEPBIND, LD_PRELOAD).",
)


def _run(code: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


# cc=(8, 0) forces the LLVM70 (sm<100) path; cc=(10, 0) forces the modern path.
# Both paths must keep bundled symbols out of RTLD_DEFAULT.
@pytest.mark.parametrize("cc", [(8, 0), (10, 0)], ids=["sm_80", "sm_100"])
def test_no_llvm_symbol_leak_to_global_scope(cc):
    code = textwrap.dedent(
        f"""
        import ctypes
        import numba_cuda_mlir  # noqa: F401
        from numba_cuda_mlir import cuda
        from numba_cuda_mlir.numba_cuda import types

        # Drive the lowering path so the launcher's dlopen runs.
        def add(x, y):
            return x + y

        ptx, _ = cuda.compile_ptx(
            add, types.int32(types.int32, types.int32),
            device=True, cc={cc},
        )
        assert ptx

        # dlsym(RTLD_DEFAULT, name) hits only globally-visible symbols.
        libdl = ctypes.CDLL("libdl.so.2", use_errno=True)
        libdl.dlsym.restype = ctypes.c_void_p
        libdl.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        RTLD_DEFAULT = None  # NULL on glibc
        leaked = []
        for sym in (b"LLVMContextCreate", b"LLVMDisposeMessage",
                    b"mlirTranslateModuleToLLVMIR",
                    b"mlirModuleCreateEmpty"):
            if libdl.dlsym(RTLD_DEFAULT, sym):
                leaked.append(sym.decode())
        assert not leaked, f"symbols leaked into RTLD_DEFAULT: {{leaked}}"
        """
    )
    result = _run(code)
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _find_shim() -> Path | None:
    # CI ships the shim next to the wheel in NUMBA_CUDA_MLIR_CUDA_ARTIFACTS_DIR;
    # locally, `make -C tests/data` drops it next to this test file.
    # NUMBA_CUDA_MLIR_TEST_BIN_DIR is an explicit override for either.
    for candidate in (
        os.environ.get("NUMBA_CUDA_MLIR_TEST_BIN_DIR"),
        os.environ.get("NUMBA_CUDA_MLIR_CUDA_ARTIFACTS_DIR"),
        str(Path(__file__).parent / "data"),
    ):
        if not candidate:
            continue
        p = Path(candidate) / "fake_mlir_shim.so"
        if p.is_file():
            return p
    return None


def test_poisoned_global_mlir_symbol_does_not_break_modern_path():
    shim_so = _find_shim()
    if shim_so is None:
        pytest.skip(
            "fake_mlir_shim.so not found; build with "
            "`make -C tests/data` or set NUMBA_CUDA_MLIR_TEST_BIN_DIR"
        )

    code = textwrap.dedent(
        """
        import numba_cuda_mlir  # noqa: F401
        from numba_cuda_mlir import cuda
        from numba_cuda_mlir.numba_cuda import types

        # cc=(10, 0) forces the modern path. Without DEEPBIND, Support's PLT
        # slot for mlirContextCreateWithThreading would bind to the shim.
        def add(x, y):
            return x + y

        ptx, resty = cuda.compile_ptx(
            add, types.int32(types.int32, types.int32),
            device=True, cc=(10, 0),
        )
        assert ptx, "empty PTX returned"
        """
    )

    env = {**os.environ, "LD_PRELOAD": str(shim_so)}
    result = _run(code, env=env)

    # Shim was called => process aborts => returncode == -SIGABRT.
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
