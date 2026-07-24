# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for post-inline whole-function extension planning."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from numba_cuda_mlir import compiler, cuda, types
from numba_cuda_mlir.cuda.experimental import consteval
from numba_cuda_mlir.errors import ForceLiteralArg
from numba_cuda_mlir._whole_function_planners import (
    _WholeFunctionPlannerRegistry,
    _planner_registry,
)
from numba_cuda_mlir.extending import (
    WholeFunctionPlanner,
    register_planner,
    require_launch_config,
    set_required_dynamic_shared_memory,
)
from numba_cuda_mlir.numba_cuda.compiler import run_frontend
from numba_cuda_mlir.numba_cuda.core import ir
from numba_cuda_mlir.numba_cuda.core.ir_utils import build_definitions


_RECOMPILE_VALUE = 1


@pytest.fixture
def isolated_global_planners():
    with _planner_registry._lock:
        original = list(_planner_registry._planners)
        _planner_registry._planners.clear()
    try:
        yield
    finally:
        with _planner_registry._lock:
            _planner_registry._planners[:] = original


def _planner_state():
    def empty():
        pass

    return SimpleNamespace(func_ir=run_frontend(empty))


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
    assert registry.apply(_planner_state()) is False
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
    assert registry.apply(_planner_state()) is False
    assert events == ["register"]

    assert registry.apply(_planner_state()) is False
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
        registry.apply(_planner_state())


def test_require_launch_config_contract():
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    state = SimpleNamespace(metadata={"targetoptions": {"__launch_config__": launch_config}})

    assert require_launch_config(state) is launch_config

    state.metadata["targetoptions"].pop("__launch_config__")
    with pytest.raises(RuntimeError, match="configured kernel launch"):
        require_launch_config(state)


def test_require_launch_config_requires_compiler_metadata():
    with pytest.raises(TypeError, match="metadata must be a dict"):
        require_launch_config(SimpleNamespace())

    with pytest.raises(TypeError, match="must contain targetoptions"):
        require_launch_config(SimpleNamespace(metadata={}))


def test_required_dynamic_shared_memory_is_max_accumulated():
    state = SimpleNamespace(metadata={})

    set_required_dynamic_shared_memory(state, 1024)
    set_required_dynamic_shared_memory(state, np.int64(512))
    set_required_dynamic_shared_memory(state, 4096)

    assert state.metadata["required_dynamic_shared_memory"] == 4096


@pytest.mark.parametrize("value", [True, False, 1.5, "1024", object()])
def test_required_dynamic_shared_memory_rejects_non_integer_values(value):
    with pytest.raises(TypeError, match="must be an integer"):
        set_required_dynamic_shared_memory(SimpleNamespace(metadata={}), value)


def test_required_dynamic_shared_memory_rejects_negative_values():
    with pytest.raises(ValueError, match="cannot be negative"):
        set_required_dynamic_shared_memory(SimpleNamespace(metadata={}), -1)


def test_required_dynamic_shared_memory_requires_compiler_metadata():
    with pytest.raises(TypeError, match="metadata must be a dict"):
        set_required_dynamic_shared_memory(SimpleNamespace(), 0)


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


def test_active_planners_bypass_persistent_dispatch_cache(
    isolated_global_planners,
    monkeypatch,
):
    from numba_cuda_mlir import descriptor as descriptor_mod, mlir_compiler
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    class Planner(WholeFunctionPlanner):
        def run(self):
            return False

    class UnexpectedCache:
        def load_overload(self, *args):
            pytest.fail("active planners must not load persistent cache entries")

        def save_overload(self, *args):
            pytest.fail("active planners must not save persistent cache entries")

    class CompilerResult:
        signature = cuda_typing.signature(types.none, types.int32)
        metadata = {"cubin": b"compiled", "func_name": "kernel"}

    def kernel(value):
        pass

    register_planner(Planner)
    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    dispatcher._cache = UnexpectedCache()
    monkeypatch.setattr(
        mlir_compiler,
        "mlir_compiler_entry",
        lambda *args, **kwargs: CompilerResult(),
    )
    monkeypatch.setattr(
        descriptor_mod._compile_arg_types,
        "types",
        (types.int32,),
        raising=False,
    )

    assert dispatcher._compile_impl([1]) == (b"compiled", "kernel", False)
    assert not dispatcher._cache_hits
    assert not dispatcher._cache_misses


def test_planner_registered_during_compile_prevents_persistent_cache_save(
    isolated_global_planners,
    monkeypatch,
):
    from numba_cuda_mlir import descriptor as descriptor_mod, mlir_compiler
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    class Planner(WholeFunctionPlanner):
        def run(self):
            return False

    class TrackingCache:
        load_calls = 0
        save_calls = 0

        def load_overload(self, *args):
            self.load_calls += 1
            return None

        def save_overload(self, *args):
            self.save_calls += 1

    class CompilerResult:
        signature = cuda_typing.signature(types.none, types.int32)
        metadata = {"cubin": b"compiled", "func_name": "kernel"}

    def kernel(value):
        pass

    def compile_and_register(*args, **kwargs):
        register_planner(Planner)
        return CompilerResult()

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    dispatcher._cache = TrackingCache()
    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", compile_and_register)
    monkeypatch.setattr(
        descriptor_mod._compile_arg_types,
        "types",
        (types.int32,),
        raising=False,
    )

    assert dispatcher._compile_impl([1]) == (b"compiled", "kernel", False)
    assert dispatcher._cache.load_calls == 1
    assert dispatcher._cache.save_calls == 0
    assert dispatcher._cache_misses[(types.int32,)] == 1


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_retained_generic_launcher_rebinds_after_recompile(isolated_global_planners):
    global _RECOMPILE_VALUE

    class HarmlessPlanner(WholeFunctionPlanner):
        def run(self):
            return False

    @cuda.jit
    def kernel(out):
        out[0] = _RECOMPILE_VALUE

    register_planner(HarmlessPlanner)
    retained_launcher = kernel[1, 32]
    out = np.zeros(1, dtype=np.int32)

    try:
        _RECOMPILE_VALUE = 1
        retained_launcher(out)
        _RECOMPILE_VALUE = 2
        kernel.recompile()
        retained_launcher(out)

        assert out[0] == 2
        assert retained_launcher._kernel_dispatcher is kernel._c
        assert kernel._requires_launch_config is False
    finally:
        _RECOMPILE_VALUE = 1


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_retained_launch_launcher_rebinds_after_recompile(isolated_global_planners):
    global _RECOMPILE_VALUE

    from numba_cuda_mlir import descriptor as descriptor_mod

    class HarmlessPlanner(WholeFunctionPlanner):
        def run(self):
            return False

    class LaunchConfigExtension:
        uses_launch_config = True

        def prepare_args(self, ty, val, stream=None, retr=None):
            return ty, val

    @cuda.jit(extensions=[LaunchConfigExtension()])
    def kernel(out):
        out[0] = _RECOMPILE_VALUE

    register_planner(HarmlessPlanner)
    retained_launcher = kernel[1, 32]
    out = np.zeros(1, dtype=np.int32)

    try:
        _RECOMPILE_VALUE = 1
        retained_launcher(out)
        retained_generation = retained_launcher._launch_config_generation
        kernel.extensions.clear()
        _RECOMPILE_VALUE = 2
        kernel.recompile()
        retained_launcher(out)

        assert out[0] == 2
        assert retained_launcher._launch_config_generation != retained_generation
        assert kernel._is_kernel_dispatcher_registered(
            retained_launcher._launch_config,
            retained_launcher._kernel_dispatcher,
            retained_launcher._launch_config_generation,
        )
        assert not hasattr(descriptor_mod._compile_arg_types, "launch_config")
    finally:
        _RECOMPILE_VALUE = 1


def _post_inline_marker(value):
    return value


class _InspectPostInlineStatePlanner(WholeFunctionPlanner):
    kernel_runs = 0

    def run(self):
        if self.is_device_function:
            return False

        func_ir = self.state.func_ir
        rebuilt_definitions = build_definitions(func_ir.blocks)
        assert set(func_ir._definitions) == set(rebuilt_definitions)
        for name, rebuilt in rebuilt_definitions.items():
            current = func_ir._definitions[name]
            assert len(current) == len(rebuilt)
            assert all(lhs is rhs for lhs, rhs in zip(current, rebuilt))
        assert set(func_ir.variable_lifetime.cfg.nodes()) == set(func_ir.blocks)
        assert set(func_ir.variable_lifetime.usedefs.usemap) == set(func_ir.blocks)
        type(self).kernel_runs += 1
        return False


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


class _LaunchConfigPlanner(WholeFunctionPlanner):
    attempts = 0
    launch_configs = []

    def run(self):
        if self.is_device_function:
            return False
        type(self).attempts += 1
        type(self).launch_configs.append(dict(require_launch_config(self.state)))
        return False


class _CountingPlanner(WholeFunctionPlanner):
    attempts = 0

    def run(self):
        if not self.is_device_function:
            type(self).attempts += 1
        return False


def test_planner_pass_runs_after_device_inlining_without_gpu(
    isolated_global_planners,
):
    _InspectPostInlineStatePlanner.kernel_runs = 0
    _MarkerPlanner.kernel_runs = 0
    _MarkerPlanner.device_runs = 0
    register_planner(_InspectPostInlineStatePlanner)
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

    assert _InspectPostInlineStatePlanner.kernel_runs == 1
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


def test_direct_compile_reports_missing_configured_launch(isolated_global_planners):
    _LaunchConfigPlanner.attempts = 0
    _LaunchConfigPlanner.launch_configs = []
    register_planner(_LaunchConfigPlanner)

    def kernel(output):
        output[0] = 1

    with pytest.raises(RuntimeError, match="compile through a configured kernel launch") as exc:
        compiler.compile(
            kernel,
            types.void(types.int32[::1]),
            device=False,
            abi="numba",
            cc=(8, 0),
        )

    assert type(exc.value) is RuntimeError
    assert exc.value.__cause__ is None
    assert _LaunchConfigPlanner.attempts == 1


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
    assert kernel._requires_launch_config is False
    assert kernel.overloads
    assert not kernel._launch_config_overloads


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_planner_can_request_distinct_launch_config_specializations(
    isolated_global_planners,
):
    _CountingPlanner.attempts = 0
    _LaunchConfigPlanner.attempts = 0
    _LaunchConfigPlanner.launch_configs = []
    register_planner(_CountingPlanner)
    register_planner(_LaunchConfigPlanner)

    @cuda.jit
    def kernel(output):
        output[0] = cuda.blockDim.x

    launch32 = kernel[1, 32]
    launch64 = kernel[1, 64]
    output32 = np.zeros(1, dtype=np.int32)
    output64 = np.zeros(1, dtype=np.int32)

    launch32(output32)
    launch64(output64)
    launch32(output32)

    assert output32[0] == 32
    assert output64[0] == 64
    assert _CountingPlanner.attempts == 3
    assert _LaunchConfigPlanner.attempts == 3
    assert [config["block"] for config in _LaunchConfigPlanner.launch_configs] == [
        (32, 1, 1),
        (64, 1, 1),
    ]
    assert kernel._requires_launch_config is True
    assert not kernel.overloads
    launch_blocks = {
        dict(launch_config_key)["block"]
        for _argtypes, launch_config_key in kernel._launch_config_overloads
    }
    assert launch_blocks == {(32, 1, 1), (64, 1, 1)}

    configured_after_demand = kernel.configure(1, 32)
    assert configured_after_demand._launch_config["block"] == (32, 1, 1)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_planner_dynamic_shared_memory_minimum_reaches_launch(
    isolated_global_planners,
):
    required_bytes = 32 * 1024
    last_scratch_index = required_bytes // np.dtype(np.int32).itemsize - 1

    class ScratchPlanner(WholeFunctionPlanner):
        def run(self):
            if self.is_device_function:
                return False
            set_required_dynamic_shared_memory(self.state, required_bytes)
            return False

    register_planner(ScratchPlanner)

    @cuda.jit
    def kernel(output):
        scratch = cuda.shared.array(0, dtype=types.int32)
        if cuda.threadIdx.x == 0:
            scratch[last_scratch_index] = 42
            output[0] = scratch[last_scratch_index]

    output = np.zeros(1, dtype=np.int32)
    kernel[1, 1](output)

    assert output[0] == 42
    [compile_result] = kernel.overloads.values()
    assert compile_result.metadata["required_dynamic_shared_memory"] == required_bytes


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_literal_retry_crosses_launch_config_and_dynamic_shared_memory(
    isolated_global_planners,
):
    required_bytes = 32 * 1024
    last_scratch_index = required_bytes // np.dtype(np.int32).itemsize - 1
    observed = []

    class LiteralScratchPlanner(WholeFunctionPlanner):
        def run(self):
            if self.is_device_function:
                return False
            launch_config = require_launch_config(self.state)
            set_required_dynamic_shared_memory(self.state, required_bytes)
            selector_type = self.state.args[1]
            observed.append((launch_config["block"], selector_type))
            if not isinstance(selector_type, types.Literal):
                raise ForceLiteralArg({1})
            return False

    register_planner(LiteralScratchPlanner)

    @cuda.jit
    def kernel(output, selector):
        scratch = cuda.shared.array(0, dtype=types.int32)
        type_bias = consteval(1000 if isinstance(selector, types.Boolean) else 0)
        if cuda.threadIdx.x == 0:
            scratch[last_scratch_index] = selector
            output[0] = scratch[last_scratch_index] + cuda.blockDim.x + type_bias

    # Alternate numerically equal int/bool pairs in both insertion orders. They
    # share an integer launch parameter family but must retain distinct native
    # constant-cache entries.
    selector_orders = {
        32: (1, True, 0, False, 3, 7),
        64: (True, 1, False, 0, 3, 7),
    }
    for block_size, selectors in selector_orders.items():
        for selector in selectors:
            output = np.zeros(1, dtype=np.int32)
            kernel[1, block_size](output, selector)
            type_bias = 1000 if isinstance(selector, bool) else 0
            assert output[0] == selector + block_size + type_bias

    assert kernel._requires_launch_config is True
    assert kernel._literal_arg_positions == frozenset({1})
    assert not kernel.overloads
    assert {
        (
            dict(launch_config_key)["block"],
            type(argtypes[1]),
            argtypes[1].literal_value,
        )
        for (argtypes, launch_config_key) in kernel._launch_config_overloads
    } == {
        (
            block,
            types.BooleanLiteral if isinstance(selector, bool) else types.IntegerLiteral,
            selector,
        )
        for block in ((32, 1, 1), (64, 1, 1))
        for selector in selector_orders[32]
    }
    assert all(
        compile_result.metadata["required_dynamic_shared_memory"] == required_bytes
        for compile_result in kernel._launch_config_overloads.values()
    )
    assert any(not isinstance(selector_type, types.Literal) for _, selector_type in observed)
    assert any(isinstance(selector_type, types.Literal) for _, selector_type in observed)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_literal_retry_overrides_partial_parameter_annotation(isolated_global_planners):
    class LiteralPlanner(WholeFunctionPlanner):
        def run(self):
            if self.is_device_function:
                return False
            if not isinstance(self.state.args[1], types.Literal):
                raise ForceLiteralArg({1})
            return False

    register_planner(LiteralPlanner)

    @cuda.jit
    def kernel(output, selector: types.int64):
        output[0] = selector

    output = np.zeros(1, dtype=np.int64)
    kernel[1, 1](output, 7)

    assert output[0] == 7
    assert kernel._literal_arg_positions == frozenset({1})
    [(argtypes, _)] = kernel.overloads.items()
    assert isinstance(argtypes[1], types.IntegerLiteral)
    assert argtypes[1].literal_value == 7
