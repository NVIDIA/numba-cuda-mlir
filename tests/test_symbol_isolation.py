# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #170: symbol isolation of the bundled LLVM/MLIR.

The bundled ``libMLIRPythonCAPI.so`` statically links a full LLVM + MLIR.
If those symbols leak into the process-global namespace, or if the library
is loaded without ``RTLD_DEEPBIND``, a differently-versioned LLVM embedded
in a sibling extension (Triton, Halide, Mojo, ...) can preempt our
internal calls and corrupt compilation state.

These tests exercise the isolation contract:

* ``test_no_llvm_symbol_leak_to_global_scope`` — the launcher's dlopen
  uses ``RTLD_LOCAL``, so no ``LLVM*`` / ``MLIR*`` symbol should be
  resolvable via ``RTLD_DEFAULT`` after ``import numba_cuda_mlir``.

* ``test_poisoned_global_mlir_symbol_does_not_break_modern_path`` — with
  a fake ``mlirContextCreateWithThreading`` in the process-global
  namespace (LD_PRELOAD'd before Python starts), a modern-path (sm_100+)
  kernel compilation must still succeed. Verifies the ``RTLD_DEEPBIND``
  preload in ``numba_cuda_mlir/__init__.py`` protects the bundled
  internals.

Both are subprocess-based because the isolation is a process-level
property established at first-load time.
"""

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
    """After exercising the modern MLIR->LLVM->NVVM path, no bundled
    LLVM/MLIR symbol may be resolvable through ``RTLD_DEFAULT``.

    We drive a modern-path (sm_100) compilation first so that
    ``cext/launcher/llvm_downgrade.cpp``'s ``load_mlir_capi`` runs and
    dlopens ``libMLIRPythonCAPI.so`` — that dlopen must not promote the
    already-loaded library to ``RTLD_GLOBAL``.
    """
    code = textwrap.dedent(
        """
        import ctypes
        import numba_cuda_mlir  # noqa: F401 -- loads MLIR/LLVM bundle
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

        # dlsym(RTLD_DEFAULT=NULL, name) on glibc: search the process-wide
        # symbol set built from executable + libraries loaded with
        # RTLD_GLOBAL. None of the bundled symbols should appear there.
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
        print("OK")
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
    """A hostile ``mlirContextCreateWithThreading`` in ``LD_PRELOAD``
    must not hijack the bundled MLIR->LLVM->NVVM lowering path.

    ``libMLIRPythonSupport-numba_cuda_mlir.so`` has hundreds of
    unresolved ``mlir*`` symbol references that get bound at load
    time to ``libMLIRPythonCAPI.so`` via DT_NEEDED. Without
    ``RTLD_DEEPBIND``, an ``RTLD_GLOBAL`` preemptor (e.g. from
    ``LD_PRELOAD``) is searched first and would win those bindings.
    We compile a shim ``.so`` that exports
    ``mlirContextCreateWithThreading`` calling ``abort()``, run a
    subprocess with that shim in ``LD_PRELOAD``, and force
    modern-path (cc=(10, 0)) compilation. If DEEPBIND isolation held,
    the bundled library's own symbol is bound and the subprocess
    exits cleanly.

    (``LLVMContextCreate`` is intentionally NOT the target: it's only
    reached via handle-based ``dlsym()`` in the launcher, which
    isn't preemptable through the global scope.)
    """
    cc = _find_compiler()
    if cc is None:
        pytest.skip("no C compiler available to build the poisoning shim")

    # mlirContextCreateWithThreading is verified via LD_DEBUG=bindings to
    # be resolved through Support's PLT during compile_ptx's MLIR IR
    # construction (nanobind -> Support -> DT_NEEDED -> CAPI).
    shim_c = tmp_path / "fake_mlir.c"
    shim_c.write_text(
        textwrap.dedent(
            """
            #include <stdio.h>
            #include <stdlib.h>
            /* MLIR C API opaque handle types are structs wrapping a
             * void*. The stub is never actually invoked if isolation
             * held, so the exact return type doesn't matter. */
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

        # cc=(10, 0) forces the modern MLIR->LLVM->NVVM path. During
        # MLIR IR construction, the nanobind bindings call into
        # libMLIRPythonSupport-numba_cuda_mlir.so, which reaches
        # mlirContextCreateWithThreading via its PLT. Without DEEPBIND
        # that PLT slot would bind to the LD_PRELOAD'd aborting shim.
        def add(x, y):
            return x + y

        ptx, resty = cuda.compile_ptx(
            add, types.int32(types.int32, types.int32),
            device=True, cc=(10, 0),
        )
        assert ptx, "empty PTX returned"
        print("OK")
        """
    )

    import os

    env = {**os.environ, "LD_PRELOAD": str(shim_so)}
    result = _run(code, env=env)

    # If Support's PLT slot for mlirContextCreateWithThreading was
    # preempted by the shim, the subprocess aborts (returncode ==
    # -SIGABRT on POSIX).
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
