# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A compile's required dynamic shared memory reaches every launch.

The requirement comes from an after-inference rewrite, so no whole-function
planner is registered and launches start on the marshaller's fast path.
"""

import threading

import numpy as np
import pytest

from numba_cuda_mlir import cuda, extending, types
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


@extending.intrinsic
def dynamic_smem_size(typingctx):
    """The dynamic shared memory of the running launch, in bytes."""

    def codegen(builder, target, args, kwargs):
        from numba_cuda_mlir._mlir.dialects import llvm as llvm_dialect
        from numba_cuda_mlir._mlir.ir import IntegerType

        size = llvm_dialect.inline_asm(
            IntegerType.get_signless(32),
            [],
            "mov.u32 $0, %dynamic_smem_size;",
            "=r",
            has_side_effects=True,
        )
        builder.store_var(target, size)

    return types.uint32(), codegen


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
    extending.refresh_registries()
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
def compiles(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    recorded = []
    mlir_compiler_entry = mlir_compiler.mlir_compiler_entry

    def count(*args, **kwargs):
        recorded.append(kwargs.get("override_argtypes"))
        return mlir_compiler_entry(*args, **kwargs)

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", count)
    return recorded


# Each kernel records the dynamic shared memory of its launch in size[0] and
# only touches its window when the launch provides REQUIRED bytes, so a
# missing requirement gives wrong values instead of faulting the context for
# later tests.


def _make_kernel():
    @cuda.jit
    def window_kernel(out, size, use_window):
        window = cuda.shared.array(0, types.int32)
        available = dynamic_smem_size()
        i = cuda.threadIdx.x
        if i == 0:
            size[0] = available
        if not use_window:
            out[i] = THREADS - 1 - i
        elif available >= REQUIRED:
            window[i] = i
            cuda.syncthreads()
            out[i] = window[THREADS - 1 - i]
        else:
            out[i] = -1

    return window_kernel


def _make_defaulted_kernel():
    @cuda.jit
    def window_kernel(out, size, use_window, pad=0):
        window = cuda.shared.array(0, types.int32)
        available = dynamic_smem_size()
        i = cuda.threadIdx.x
        if i == 0:
            size[0] = available
        if available >= REQUIRED:
            window[i] = i
            cuda.syncthreads()
            out[i] = window[THREADS - 1 - i] + pad
        else:
            out[i] = -1

    return window_kernel


def _make_star_args_kernel():
    @cuda.jit
    def window_kernel(out, size, use_window, *rest):
        window = cuda.shared.array(0, types.int32)
        available = dynamic_smem_size()
        i = cuda.threadIdx.x
        if i == 0:
            size[0] = available
        if available >= REQUIRED:
            window[i] = i
            cuda.syncthreads()
            out[i] = window[THREADS - 1 - i]
        else:
            out[i] = -1

    return window_kernel


# Omitting pad is not covered: the omitted argument is not marshalled
# (NVIDIA/numba-cuda-mlir#410). *args kernels are always compiled with a
# StarArgTuple, so their launch types never match the compiled signature.
_SIGNATURE_SHAPES = {
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


def _run(launcher, dtype=np.int32, extra=()):
    """Launch, check the output, and return the launch's dynamic shared memory."""
    out = cuda.to_device(np.zeros(THREADS, dtype=dtype))
    size = cuda.to_device(np.zeros(1, dtype=np.uint32))
    launcher(out, size, dtype == np.int32, *extra)
    cuda.synchronize()
    np.testing.assert_array_equal(out.copy_to_host(), EXPECTED)
    return int(size.copy_to_host()[0])


def test_first_launch_gets_the_requirement(requirement, compiles):
    kernel = _make_kernel()

    assert _run(kernel[1, THREADS]) == REQUIRED
    assert len(compiles) == 1
    assert kernel._requires_launch_config is requirement


def test_later_launches_get_the_requirement(requirement, compiles):
    kernel = _make_kernel()
    sizes = [_run(kernel[1, THREADS, 0, REQUIRED])]

    launcher = kernel[1, THREADS]
    sizes.append(_run(launcher))
    sizes.append(_run(launcher))
    sizes.append(_run(kernel[2, THREADS]))

    assert sizes == [REQUIRED] * 4
    # Launch-config compiles specialize on each of the three launch configurations.
    assert len(compiles) == (3 if requirement else 1)


def test_larger_explicit_sharedmem_is_kept(requirement):
    kernel = _make_kernel()
    launcher = kernel[1, THREADS, 0, 2 * REQUIRED]

    assert [_run(launcher), _run(launcher)] == [2 * REQUIRED] * 2


def test_smaller_explicit_sharedmem_is_raised(requirement):
    kernel = _make_kernel()
    launcher = kernel[1, THREADS, 0, REQUIRED // 2]

    assert [_run(launcher), _run(launcher)] == [REQUIRED] * 2


def test_new_signature_gets_its_requirement(requirement, compiles):
    kernel = _make_kernel()
    launcher = kernel[1, THREADS]
    sizes = [_run(launcher, np.float32), _run(launcher, np.float32)]

    sizes += [_run(launcher), _run(launcher)]

    assert sizes == [0, 0, REQUIRED, REQUIRED]
    assert len(compiles) == 2


def _assert_launches_get_the_requirement(kernel, extra, requirement, compiles):
    launcher = kernel[1, THREADS]
    sizes = [
        _run(launcher, extra=extra),
        _run(launcher, extra=extra),
        _run(kernel[2, THREADS], extra=extra),
    ]

    assert sizes == [REQUIRED] * 3
    # Launch-config compiles specialize on each of the two launch configurations.
    assert len(compiles) == (2 if requirement else 1)


@pytest.mark.parametrize(
    "make_kernel, extra", list(_SIGNATURE_SHAPES.values()), ids=list(_SIGNATURE_SHAPES)
)
def test_defaulted_and_star_args_kernels_get_the_requirement(
    requirement, compiles, make_kernel, extra
):
    _assert_launches_get_the_requirement(make_kernel(), extra, requirement, compiles)


@pytest.mark.parametrize(
    "make_kernel, extra", list(_SIGNATURE_SHAPES.values()), ids=list(_SIGNATURE_SHAPES)
)
def test_defaulted_and_star_args_kernels_with_a_planner(
    requirement, planner, compiles, make_kernel, extra
):
    _assert_launches_get_the_requirement(make_kernel(), extra, requirement, compiles)


def test_kernel_without_requirement_is_unchanged(requirement, monkeypatch):
    kernel = _make_kernel()
    ready_checks = []
    get_ready_launch = kernel._get_ready_launch

    def observe_ready_launch(*args):
        ready_checks.append(args[0])
        return get_ready_launch(*args)

    monkeypatch.setattr(kernel, "_get_ready_launch", observe_ready_launch)
    launcher = kernel[1, THREADS]
    sizes = [_run(launcher, np.float32), _run(launcher, np.float32)]
    sizes.append(_run(kernel[1, THREADS, 0, REQUIRED // 2], np.float32))

    assert sizes == [0, 0, REQUIRED // 2]
    if not requirement:
        # Without a requirement or a launch config, launches keep the native fast path.
        assert ready_checks == []


@pytest.mark.parametrize("other", ["launch", "compile"])
def test_launch_started_before_the_requirement_gets_it(requirement, other):
    """A launch paused just before its native launcher call, while another
    caller compiles the kernel and records the requirement, still gets it.

    After another launch the kernel is in the native launcher's cache, so the
    paused launch never calls back into Python; after a compile it is not.
    """
    if other == "compile" and requirement:
        pytest.skip("compile() cannot compile a kernel that requires a launch config")
    kernel = _make_kernel()
    paused_launcher = kernel[1, THREADS]
    native_launcher = paused_launcher._launcher
    paused = threading.Event()
    resume = threading.Event()

    def pause_then_launch(*args):
        paused.set()
        assert resume.wait(60)
        return native_launcher(*args)

    paused_launcher._launcher = pause_then_launch
    sizes = {}
    errors = []

    def launch_paused():
        try:
            sizes["paused"] = _run(paused_launcher)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=launch_paused)
    thread.start()
    try:
        assert paused.wait(60)
        if other == "launch":
            sizes["other"] = _run(kernel[2, THREADS])
        else:
            kernel.compile((types.int32[::1], types.uint32[::1], types.boolean))
    finally:
        resume.set()
        thread.join(60)
    assert not thread.is_alive()
    if errors:
        raise errors[0]

    expected = {"paused": REQUIRED}
    if other == "launch":
        expected["other"] = REQUIRED
    assert sizes == expected
