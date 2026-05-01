# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Optional

from numba_cuda_mlir._mlir import ir


@dataclass
class MLIRContext:
    context: ir.Context
    module: ir.Module

    def __str__(self):
        return str(self.module)


@contextmanager
def mlir_mod(
    src: Optional[str] = None,
    location: ir.Location = None,
) -> ir.Module:
    with ExitStack() as stack:
        if location is None:
            location = ir.Location.unknown()
        stack.enter_context(location)
        if src is not None:
            module = ir.Module.parse(src)
        else:
            module = ir.Module.create()
        ip = ir.InsertionPoint(module.body)
        stack.enter_context(ip)
        yield module


@contextmanager
def mlir_mod_ctx(
    src: Optional[str] = None,
    context: ir.Context = None,
    location: ir.Location = None,
    allow_unregistered_dialects=False,
) -> MLIRContext:
    if context is None:
        context = ir.Context()
    if allow_unregistered_dialects:
        context.allow_unregistered_dialects = True
    with context, mlir_mod(src, location) as module:
        yield MLIRContext(context, module)
