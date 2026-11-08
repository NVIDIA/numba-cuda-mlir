# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every name in ``numba_cuda_mlir.__all__`` must resolve in a fresh interpreter.

These run in a subprocess on purpose. A submodule listed in ``__all__`` but never
bound still resolves once any other import has pulled it in, so an in-process
assertion passes under a full test run and only fails when a test runs alone.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_all_names_resolve_as_attributes():
    result = _run(
        """
        import numba_cuda_mlir

        missing = [
            name
            for name in numba_cuda_mlir.__all__
            if not hasattr(numba_cuda_mlir, name)
        ]
        assert not missing, f"exported but unreachable: {missing}"
        """
    )
    assert result.returncode == 0, result.stderr


def test_all_names_resolve_through_star_import():
    result = _run(
        """
        import numba_cuda_mlir

        namespace = {}
        exec("from numba_cuda_mlir import *", namespace)
        missing = [name for name in numba_cuda_mlir.__all__ if name not in namespace]
        assert not missing, f"exported but not bound by star import: {missing}"
        """
    )
    assert result.returncode == 0, result.stderr


def test_unknown_attribute_still_raises():
    result = _run(
        """
        import numba_cuda_mlir

        try:
            numba_cuda_mlir.no_such_attribute
        except AttributeError:
            pass
        else:
            raise AssertionError("expected AttributeError")
        """
    )
    assert result.returncode == 0, result.stderr
