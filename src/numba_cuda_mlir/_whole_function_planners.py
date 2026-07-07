# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-function extension planning after device-function inlining."""

from threading import RLock

from numba_cuda_mlir.numba_cuda.core import postproc
from numba_cuda_mlir.numba_cuda.core.ir_utils import build_definitions, simplify_CFG


class WholeFunctionPlanner:
    """Base class for whole-function extension planners.

    Planners run after device-function inlining and before type inference. They
    may inspect and rewrite any block in ``state.func_ir`` and must return a
    Boolean indicating whether they changed the IR.
    """

    def __init__(self, state):
        self.state = state

    @property
    def is_device_function(self) -> bool:
        """Whether the current compilation is for a device function."""

        targetoptions = self.state.metadata.get("targetoptions", {})
        return bool(targetoptions.get("device", False))

    def run(self) -> bool:
        """Inspect or rewrite the current function and report whether it changed."""

        raise NotImplementedError


class _WholeFunctionPlannerRegistry:
    def __init__(self):
        self._planners = []
        self._lock = RLock()

    def register(self, planner_cls):
        """Register a planner class and return it for decorator use."""

        if not isinstance(planner_cls, type) or not issubclass(planner_cls, WholeFunctionPlanner):
            raise TypeError(f"{planner_cls!r} is not a WholeFunctionPlanner subclass")
        with self._lock:
            if planner_cls not in self._planners:
                self._planners.append(planner_cls)
        return planner_cls

    @property
    def has_planners(self) -> bool:
        with self._lock:
            return bool(self._planners)

    def apply(self, state) -> bool:
        """Run each planner once with coherent IR and repair every mutation."""

        modified = False
        with self._lock:
            planners = tuple(self._planners)
        if planners:
            # Inlining and CFG simplification can leave definitions and
            # postprocessing analysis describing blocks that no longer exist.
            # Repair once so the first inspection-only planner sees the same
            # coherent IR promised to every planner after a mutation.
            self._repair_ir(state.func_ir)
        for planner_cls in planners:
            result = planner_cls(state).run()
            if not isinstance(result, bool):
                raise TypeError(
                    f"{planner_cls.__name__}.run() must return bool, got {type(result)!r}"
                )
            if result:
                self._repair_ir(state.func_ir)
                modified = True
        return modified

    @staticmethod
    def _repair_ir(func_ir) -> None:
        func_ir.blocks = simplify_CFG(func_ir.blocks)
        func_ir._reset_analysis_variables()
        postproc.PostProcessor(func_ir).run()
        func_ir._definitions = build_definitions(func_ir.blocks)
        for block in func_ir.blocks.values():
            block.verify()


_planner_registry = _WholeFunctionPlannerRegistry()


def register_planner(planner_cls):
    """Register a whole-function planner class."""

    return _planner_registry.register(planner_cls)


__all__ = ["WholeFunctionPlanner", "register_planner"]
