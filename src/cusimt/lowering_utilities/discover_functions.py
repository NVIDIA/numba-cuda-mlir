# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt._mlir import ir
from cusimt._mlir.dialects import func, gpu, builtin
from cusimt._mlir.passmanager import PassManager
import logging

from cusimt.lowering_utilities.type_conversions import to_numba_type
from cusimt.lowering_utilities import context


class DiscoverFunctionsPass:
    def __init__(self):
        self.__name__ = DiscoverFunctionsPass.__name__
        self.functions = dict()

    def recurse(self, op: ir.Operation, _pass: PassManager):
        for region in op.regions:
            for block in region.blocks:
                for operation in block:
                    self(operation, _pass)

    def __call__(self, op: ir.Operation, _pass: PassManager):
        logging.debug(f"DiscoverFunctionsPass: {op.name}")

        if getattr(op, "name", None) == "builtin.module":
            self.recurse(op, _pass)
            return

        match op:
            case func.FuncOp():
                self.functions[op.name.value] = to_numba_type(op.type)
            case gpu.GPUFuncOp():
                self.functions[op.name.value] = to_numba_type(op.function_type.value)
            case gpu.GPUModuleOp():
                self.recurse(op, _pass)


def discover_functions(module_str: str) -> dict[str, func.FuncOp]:
    """
    Discover all functions in the module.
    """
    with context.get_context():
        try:
            module = ir.Module.parse(module_str)
        except Exception as e:
            raise ValueError(f"Failed to parse MLIR module from string:\n{module_str}\n{e}")
        pm = PassManager()
        p = DiscoverFunctionsPass()
        pm.add(p)
        pm.run(module.operation)

    return p.functions
