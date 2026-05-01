# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir._mlir import ir
import contextvars
from typing import Any

_context: ir.Context | None = None
_compilation_options: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "compilation_options", default={}
)


def get_context() -> ir.Context:
    global _context
    if _context is None:
        _context = ir.Context()
    return _context


def get_compilation_options() -> dict[str, Any]:
    return _compilation_options.get()


def set_compilation_options(options: dict[str, Any]) -> contextvars.Token:
    return _compilation_options.set(options)
