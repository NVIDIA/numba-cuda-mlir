# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A compile's required dynamic shared memory reaches every launch.

The requirement comes from an after-inference rewrite, so no whole-function
planner is registered and launches start on the marshaller's fast path.
"""

import numpy as np
import pytest

from numba_cuda_mlir import cuda, types
from numba_cuda_mlir import descriptor as descriptor_mod
from numba_cuda_mlir._whole_function_planners import WholeFunctionPlanner, _planner_registry
from numba_cuda_mlir.extending import (
    require_launch_config,
    set_required_dynamic_shared_memory,
)
from numba_cuda_mlir.numba_cuda.core.rewrites import Rewrite
from numba_cuda_mlir.numba_cuda.core.rewrites.registry import rewrite_registry


THREADS = 256
REQUIRED = THREADS * np.dtype(np.int32).itemsize
EXPECTED = np.arange(THREADS - 1, -1, -1)

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")


class _RequireDynamicSharedMemory(Rewrite):
    """Require REQUIRED bytes for window_kernel when its output is int32."""

    with_launch_config = False

    def __init__(self, state=None):
        super().__init__(state)
        self.state = state

    def match(self, func_ir, block, typemap, calltypes):
        state = self.state
        if state.func_id.func_name != "window_kernel" or "test_done" in state.metadata:
            return False
        state.metadata["test_done"] = True
        if self.with_launch_config:
            require_launch_config(state)
        if state.args[0].dtype == types.int32:
            set_required_dynamic_shared_memory(state, REQUIRED)
        return False

    def apply(self):
        raise AssertionError("match never selects a block")


@pytest.fixture(params=[False, True], ids=["generic", "launch_config"])
def requirement(request):
    with _planner_registry._lock:
        planners = list(_planner_registry._planners)
        _planner_registry._planners.clear()
    _RequireDynamicSharedMemory.with_launch_config = request.param
    rewrites = rewrite_registry.rewrites["after-inference"]
    rewrites.append(_RequireDynamicSharedMemory)
    try:
        yield request.param
    finally:
        rewrites.remove(_RequireDynamicSharedMemory)
        with _planner_registry._lock:
            _planner_registry._planners[:] = planners


@pytest.fixture
def launches(monkeypatch):
    """Record the dynamic shared memory of each completed native launch.

    The kernel only touches its window when ``guard`` says the launch provides
    REQUIRED bytes, so a missing requirement gives wrong values instead of
    faulting the context for later tests.
    """

    class Launches(list):
        guard = cuda.to_device(np.zeros(1, dtype=np.int32))

    recorded = Launches()
    launch_configuration = descriptor_mod.LaunchConfiguration

    def spy(kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster):
        launcher = launch_configuration(
            kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster
        )
        requested = descriptor_mod._normalize_launch_sharedmem(sharedmem)

        def launch(*args):
            recorded.guard.copy_to_device(np.array([requested >= REQUIRED], dtype=np.int32))
            result = launcher(*args)
            recorded.append(requested)
            return result

        return launch

    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", spy)
    return recorded


@pytest.fixture
def compiles(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    recorded = []
    mlir_compiler_entry = mlir_compiler.mlir_compiler_entry

    def count(*args, **kwargs):
        recorded.append(kwargs.get("override_argtypes"))
        return mlir_compiler_entry(*args, **kwargs)

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", count)
    return recorded


def _make_kernel():
    @cuda.jit
    def window_kernel(out, guard, use_window):
        window = cuda.shared.array(0, types.int32)
        i = cuda.threadIdx.x
        if not use_window:
            out[i] = THREADS - 1 - i
        elif guard[0]:
            window[i] = i
            cuda.syncthreads()
            out[i] = window[THREADS - 1 - i]
        else:
            out[i] = -1

    return window_kernel


def _make_defaulted_kernel():
    @cuda.jit
    def window_kernel(out, guard, use_window, pad=0):
        window = cuda.shared.array(0, types.int32)
        i = cuda.threadIdx.x
        if guard[0]:
            window[i] = i
            cuda.syncthreads()
            out[i] = window[THREADS - 1 - i] + pad
        else:
            out[i] = -1

    return window_kernel


def _make_star_args_kernel():
    @cuda.jit
    def window_kernel(out, guard, use_window, *rest):
        window = cuda.shared.array(0, types.int32)
        i = cuda.threadIdx.x
        if guard[0]:
            window[i] = i
            cuda.syncthreads()
            out[i] = window[THREADS - 1 - i]
        else:
            out[i] = -1

    return window_kernel


# Omitted defaults compile as types.Omitted and *args as a StarArgTuple, so
# those launch types never match the compiled signature.
_SIGNATURE_SHAPES = {
    "default_omitted": (_make_defaulted_kernel, ()),
    "default_passed": (_make_defaulted_kernel, (0,)),
    "star_args_empty": (_make_star_args_kernel, ()),
    "star_args": (_make_star_args_kernel, (0,)),
}


class _NoOpPlanner(WholeFunctionPlanner):
    def run(self):
        return False


@pytest.fixture
def planner(requirement):
    _planner_registry.register(_NoOpPlanner)
    try:
        yield
    finally:
        with _planner_registry._lock:
            _planner_registry._planners.remove(_NoOpPlanner)


def _run(launcher, launches, dtype=np.int32, extra=()):
    out = cuda.to_device(np.zeros(THREADS, dtype=dtype))
    launcher(out, launches.guard, dtype == np.int32, *extra)
    cuda.synchronize()
    np.testing.assert_array_equal(out.copy_to_host(), EXPECTED)


def test_first_launch_gets_the_requirement(requirement, launches, compiles):
    kernel = _make_kernel()

    _run(kernel[1, THREADS], launches)

    assert launches == [REQUIRED]
    assert len(compiles) == 1
    assert kernel._requires_launch_config is requirement


def test_later_launches_get_the_requirement(requirement, launches, compiles):
    kernel = _make_kernel()
    _run(kernel[1, THREADS, 0, REQUIRED], launches)

    launcher = kernel[1, THREADS]
    _run(launcher, launches)
    _run(launcher, launches)
    _run(kernel[2, THREADS], launches)

    assert launches == [REQUIRED] * 4
    # Launch-config compiles specialize on each of the three launch configurations.
    assert len(compiles) == (3 if requirement else 1)


def test_larger_explicit_sharedmem_is_kept(requirement, launches):
    kernel = _make_kernel()
    launcher = kernel[1, THREADS, 0, 2 * REQUIRED]

    _run(launcher, launches)
    _run(launcher, launches)

    assert launches == [2 * REQUIRED] * 2


def test_smaller_explicit_sharedmem_is_raised(requirement, launches):
    kernel = _make_kernel()
    launcher = kernel[1, THREADS, 0, REQUIRED // 2]

    _run(launcher, launches)
    _run(launcher, launches)

    assert launches == [REQUIRED] * 2


def test_new_signature_gets_its_requirement(requirement, launches, compiles):
    kernel = _make_kernel()
    launcher = kernel[1, THREADS]
    _run(launcher, launches, np.float32)
    _run(launcher, launches, np.float32)

    _run(launcher, launches)
    _run(launcher, launches)

    assert launches == [0, 0, REQUIRED, REQUIRED]
    assert len(compiles) == 2


def _assert_launches_get_the_requirement(kernel, extra, requirement, launches, compiles):
    launcher = kernel[1, THREADS]
    _run(launcher, launches, extra=extra)
    _run(launcher, launches, extra=extra)
    _run(kernel[2, THREADS], launches, extra=extra)

    assert launches == [REQUIRED] * 3
    # Launch-config compiles specialize on each of the two launch configurations.
    assert len(compiles) == (2 if requirement else 1)


@pytest.mark.parametrize(
    "make_kernel, extra", list(_SIGNATURE_SHAPES.values()), ids=list(_SIGNATURE_SHAPES)
)
def test_defaulted_and_star_args_kernels_get_the_requirement(
    requirement, launches, compiles, make_kernel, extra
):
    _assert_launches_get_the_requirement(make_kernel(), extra, requirement, launches, compiles)


@pytest.mark.parametrize(
    "make_kernel, extra", list(_SIGNATURE_SHAPES.values()), ids=list(_SIGNATURE_SHAPES)
)
def test_defaulted_and_star_args_kernels_with_a_planner(
    requirement, planner, launches, compiles, make_kernel, extra
):
    _assert_launches_get_the_requirement(make_kernel(), extra, requirement, launches, compiles)


def test_kernel_without_requirement_is_unchanged(requirement, launches, monkeypatch):
    kernel = _make_kernel()
    ready_checks = []
    get_ready_launch = kernel._get_ready_launch

    def observe_ready_launch(*args):
        ready_checks.append(args[0])
        return get_ready_launch(*args)

    monkeypatch.setattr(kernel, "_get_ready_launch", observe_ready_launch)
    launcher = kernel[1, THREADS]
    _run(launcher, launches, np.float32)
    _run(launcher, launches, np.float32)
    _run(kernel[1, THREADS, 0, REQUIRED // 2], launches, np.float32)

    assert launches == [0, 0, REQUIRED // 2]
    if not requirement:
        # Without a requirement or a launch config, launches keep the native fast path.
        assert ready_checks == []
