# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for post-inline whole-function extension planning."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from numba_cuda_mlir import compiler, cuda, types
from numba_cuda_mlir._whole_function_planners import (
    _WholeFunctionPlannerRegistry,
    _planner_registry,
)
from numba_cuda_mlir.extending import WholeFunctionPlanner, register_planner
from numba_cuda_mlir.numba_cuda.compiler import run_frontend
from numba_cuda_mlir.numba_cuda.core import ir


@pytest.fixture
def isolated_global_planners():
    with _planner_registry._lock:
        original = list(_planner_registry._planners)
        _planner_registry._planners.clear()
        try:
            yield
        finally:
            _planner_registry._planners[:] = original


def test_planners_run_once_in_registration_order():
    events = []
    registry = _WholeFunctionPlannerRegistry()

    class FirstPlanner(WholeFunctionPlanner):
        def run(self):
            events.append("first")
            return False

    class SecondPlanner(WholeFunctionPlanner):
        def run(self):
            events.append("second")
            return False

    assert registry.register(FirstPlanner) is FirstPlanner
    assert registry.register(FirstPlanner) is FirstPlanner
    assert registry.register(SecondPlanner) is SecondPlanner
    assert registry.apply(SimpleNamespace()) is False
    assert events == ["first", "second"]


def test_registration_during_planning_applies_to_next_compilation():
    events = []
    registry = _WholeFunctionPlannerRegistry()

    class LatePlanner(WholeFunctionPlanner):
        def run(self):
            events.append("late")
            return False

    class RegisteringPlanner(WholeFunctionPlanner):
        def run(self):
            events.append("register")
            registry.register(LatePlanner)
            return False

    registry.register(RegisteringPlanner)
    assert registry.apply(SimpleNamespace()) is False
    assert events == ["register"]

    assert registry.apply(SimpleNamespace()) is False
    assert events == ["register", "register", "late"]


def test_concurrent_duplicate_registration_is_idempotent():
    registry = _WholeFunctionPlannerRegistry()

    class Planner(WholeFunctionPlanner):
        def run(self):
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        registered = list(executor.map(registry.register, [Planner] * 64))

    assert registered == [Planner] * 64
    assert registry._planners == [Planner]


def test_planner_registration_and_result_contracts():
    registry = _WholeFunctionPlannerRegistry()

    with pytest.raises(TypeError, match="WholeFunctionPlanner subclass"):
        registry.register(object)

    class InvalidResultPlanner(WholeFunctionPlanner):
        def run(self):
            return None

    registry.register(InvalidResultPlanner)
    with pytest.raises(TypeError, match=r"run\(\) must return bool"):
        registry.apply(SimpleNamespace())


def test_ir_is_repaired_between_modifying_planners():
    events = []
    observed_constants = []
    registry = _WholeFunctionPlannerRegistry()

    def add_one(value):
        return value + 1

    func_ir = run_frontend(add_one)

    class ReplaceConstantPlanner(WholeFunctionPlanner):
        replaced_var = None

        def run(self):
            events.append("replace")
            for block in self.state.func_ir.blocks.values():
                for index, inst in enumerate(block.body):
                    if (
                        isinstance(inst, ir.Assign)
                        and isinstance(inst.value, ir.Const)
                        and inst.value.value == 1
                    ):
                        block.body[index] = ir.Assign(ir.Const(2, inst.loc), inst.target, inst.loc)
                        type(self).replaced_var = inst.target
                        return True
            raise AssertionError("test function did not contain the expected constant")

    class ObserveDefinitionPlanner(WholeFunctionPlanner):
        def run(self):
            events.append("observe")
            definition = self.state.func_ir.get_definition(ReplaceConstantPlanner.replaced_var)
            observed_constants.append(definition.value)
            return False

    registry.register(ReplaceConstantPlanner)
    registry.register(ObserveDefinitionPlanner)

    assert registry.apply(SimpleNamespace(func_ir=func_ir)) is True
    assert events == ["replace", "observe"]
    assert observed_constants == [2]


def _post_inline_marker(value):
    return value


class _MarkerPlanner(WholeFunctionPlanner):
    kernel_runs = 0
    device_runs = 0

    def run(self):
        if self.is_device_function:
            type(self).device_runs += 1
            return False

        type(self).kernel_runs += 1
        marker_vars = set()
        for block in self.state.func_ir.blocks.values():
            for inst in block.body:
                if (
                    isinstance(inst, ir.Assign)
                    and isinstance(inst.value, ir.Global)
                    and inst.value.value is _post_inline_marker
                ):
                    marker_vars.add(inst.target.name)

        modified = False
        for block in self.state.func_ir.blocks.values():
            new_body = []
            for inst in block.body:
                if isinstance(inst, ir.Assign) and inst.target.name in marker_vars:
                    new_body.append(ir.Assign(ir.Const(None, inst.loc), inst.target, inst.loc))
                    modified = True
                    continue
                if isinstance(inst, ir.Assign):
                    call = inst.value
                    if (
                        isinstance(call, ir.Expr)
                        and call.op == "call"
                        and isinstance(call.func, ir.Var)
                        and call.func.name in marker_vars
                    ):
                        new_body.append(ir.Assign(call.args[0], inst.target, inst.loc))
                        modified = True
                        continue
                new_body.append(inst)
            block.body = new_body
        return modified


def test_planner_pass_runs_after_device_inlining_without_gpu(
    isolated_global_planners,
):
    _MarkerPlanner.kernel_runs = 0
    _MarkerPlanner.device_runs = 0
    register_planner(_MarkerPlanner)

    @cuda.jit(device=True, inline="always")
    def device_func(value):
        if value > 0:
            return _post_inline_marker(value)
        return value

    def kernel(d_in, d_out):
        d_out[0] = device_func(d_in[0])

    compiler.compile(
        kernel,
        types.void(types.int32[::1], types.int32[::1]),
        device=False,
        abi="numba",
        cc=(8, 0),
    )

    assert _MarkerPlanner.kernel_runs == 1
    assert _MarkerPlanner.device_runs == 0


def test_planner_runs_for_device_function_compilation_without_gpu(
    isolated_global_planners,
):
    _MarkerPlanner.kernel_runs = 0
    _MarkerPlanner.device_runs = 0
    register_planner(_MarkerPlanner)

    def device_func(value):
        return value + 1

    compiler.compile(
        device_func,
        types.int32(types.int32),
        device=True,
        abi="numba",
        cc=(8, 0),
    )

    assert _MarkerPlanner.kernel_runs == 0
    assert _MarkerPlanner.device_runs == 1


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_planner_sees_inline_device_function_body(isolated_global_planners):
    _MarkerPlanner.kernel_runs = 0
    _MarkerPlanner.device_runs = 0
    register_planner(_MarkerPlanner)

    @cuda.jit(device=True, inline="always")
    def device_func(value):
        if value > 0:
            return _post_inline_marker(value)
        return value

    @cuda.jit
    def kernel(d_in, d_out):
        d_out[0] = device_func(d_in[0])

    h_input = np.asarray([7], dtype=np.int32)
    h_output = np.asarray([0], dtype=np.int32)
    kernel[1, 1](h_input, h_output)

    np.testing.assert_array_equal(h_output, h_input)
    assert _MarkerPlanner.kernel_runs == 1
    assert _MarkerPlanner.device_runs == 0
