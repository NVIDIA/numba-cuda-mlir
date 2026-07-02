# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regression tests for #170. Subprocess-based because symbol isolation
# is a process-level property established at first-load time.
from __future__ import annotations

import shutil
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


def test_no_llvm_symbol_leak_to_global_scope():
    code = textwrap.dedent(
        """
        import ctypes
        import numba_cuda_mlir  # noqa: F401
        from numba_cuda_mlir import cuda
        from numba_cuda_mlir.numba_cuda import types

        # Drive the modern lowering path so the launcher's dlopen runs.
        def add(x, y):
            return x + y

        ptx, _ = cuda.compile_ptx(
            add, types.int32(types.int32, types.int32),
            device=True, cc=(10, 0),
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
        assert not leaked, f"symbols leaked into RTLD_DEFAULT: {leaked}"
        """
    )
    result = _run(code)
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _find_compiler() -> str | None:
    for name in ("cc", "gcc", "clang"):
        p = shutil.which(name)
        if p:
            return p
    return None


def test_poisoned_global_mlir_symbol_does_not_break_modern_path(tmp_path: Path):
    cc = _find_compiler()
    if cc is None:
        pytest.skip("no C compiler available to build the poisoning shim")

    # mlirContextCreateWithThreading picked via LD_DEBUG=bindings — hit
    # early through Support's PLT during modern lowering.
    shim_c = tmp_path / "fake_mlir.c"
    shim_c.write_text(
        textwrap.dedent(
            """
            #include <stdio.h>
            #include <stdlib.h>
            /* Return type only matters if isolation broke; abort fires first. */
            typedef struct { void *ptr; } MlirContext;
            MlirContext mlirContextCreateWithThreading(int threading_enabled) {
                (void)threading_enabled;
                fputs("FAKE mlirContextCreateWithThreading called"
                      " -- isolation broken\\n", stderr);
                abort();
            }
            """
        )
    )
    shim_so = tmp_path / "fake_mlir.so"
    subprocess.run(
        [cc, "-shared", "-fPIC", "-O0", "-o", str(shim_so), str(shim_c)],
        check=True,
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

    import os

    env = {**os.environ, "LD_PRELOAD": str(shim_so)}
    result = _run(code, env=env)

    # Shim was called => process aborts => returncode == -SIGABRT.
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
