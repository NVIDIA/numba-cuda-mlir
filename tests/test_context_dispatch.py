# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device/context partitioning without requiring a CUDA driver."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
from uuid import uuid4

import pytest

from numba_cuda_mlir import descriptor, mlir_compiler, tools, types
from numba_cuda_mlir._context_cache import current_context_token
from numba_cuda_mlir.compiler import CodeLibrary
from numba_cuda_mlir.numba_cuda import typing
from numba_cuda_mlir.numba_cuda.cudadrv import devices


@pytest.fixture
def contexts(monkeypatch):
    def context(device, cc, handle):
        return SimpleNamespace(
            device=SimpleNamespace(id=device, compute_capability=cc), handle=handle, extras={}
        )

    a = context(0, (9, 0), 100)
    b = context(1, (12, 0), 200)
    local = threading.local()
    local.context = a
    monkeypatch.setattr(devices, "get_context", lambda: local.context)
    return a, b, local


@pytest.fixture
def compiler_stub(monkeypatch):
    calls = []

    def compile_mlir(pyfunc, return_type, args, targetoptions):
        target = tools.resolve_gpu_target(targetoptions)
        metadata = {
            "targetoptions": targetoptions.copy(),
            "gpu_target": target,
            "cubin": target["chip"].encode(),
            "func_name": pyfunc.__name__,
            "required_dynamic_shared_memory": target["host_cc"][0] * 128,
        }
        calls.append(metadata)
        return SimpleNamespace(
            signature=typing.signature(types.none, *args),
            metadata=metadata,
            target_context=SimpleNamespace(insert_user_function=lambda *args: None),
            entry_point=None,
            fndesc=None,
            library=None,
            objectmode=False,
        )

    monkeypatch.setattr(mlir_compiler, "compile_mlir", compile_mlir)
    monkeypatch.setattr(
        mlir_compiler,
        "mlir_compiler_entry",
        lambda pyfunc, func_args, targetoptions, override_argtypes: compile_mlir(
            pyfunc, types.none, override_argtypes, targetoptions
        ),
    )
    from numba_cuda_mlir import mlir_optimization

    monkeypatch.setattr(mlir_optimization, "optimize", lambda cres: None)
    monkeypatch.setattr(descriptor.mlir_target, "ensure_initialized", lambda: None)
    return calls


def make_dispatcher(**options):
    def kernel(x):
        pass

    return descriptor.MLIRDispatcher(kernel, targetoptions=options)


def test_capability_and_target_follow_current_device(contexts, monkeypatch):
    a, b, local = contexts
    assert tools.get_gpu_compute_capability(tuple) == (9, 0)
    local.context = b
    assert tools.get_gpu_compute_capability() == "sm_120"
    assert tools.resolve_gpu_target()["linker_arch"] == "sm_120"
    assert tools.resolve_gpu_target({"chip": "sm_90a"})["chip"] == "sm_90a"
    local.context = a
    assert tools.get_gpu_compute_capability() == "sm_90"

    calls = []
    monkeypatch.setattr(
        tools, "get_gpu_compute_capability", lambda as_type: calls.append(as_type) or (9, 0)
    )
    target = tools.resolve_gpu_target()
    assert target["chip"] == target["linker_arch"] == "sm_90"
    assert calls == [tuple]


@pytest.mark.parametrize("same_arch", [False, True])
def test_overloads_and_native_state_follow_context(contexts, compiler_stub, same_arch):
    a, b, local = contexts
    if same_arch:
        b.device.compute_capability = a.device.compute_capability
    dispatch = make_dispatcher()
    dispatch._get_context_dispatcher()
    first = dispatch.compile((types.int32,))
    native_a = dispatch._get_context_dispatcher()._c
    local.context = b
    second = dispatch.compile((types.int32,))
    native_b = dispatch._get_context_dispatcher()._c
    assert second is not first
    assert native_b is not native_a
    assert dispatch.overloads[(types.int32,)] is second
    assert dispatch.get_metadata((types.int32,)) is second.metadata
    assert "chip" not in dispatch.targetoptions
    local.context = a
    assert dispatch.compile((types.int32,)) is first
    assert dispatch._get_context_dispatcher()._c is native_a
    assert len(compiler_stub) == 2


def test_explicit_chip_is_preserved_with_device_specific_linker(contexts, compiler_stub):
    a, b, local = contexts
    dispatch = make_dispatcher(chip="sm_90a")
    assert dispatch.compile((types.int32,)).metadata["gpu_target"]["linker_arch"] == "sm_90a"
    local.context = b
    result = dispatch.compile((types.int32,))
    assert result.metadata["gpu_target"]["linker_arch"] == "sm_120"
    assert dispatch.targetoptions["chip"] == "sm_90a"
    assert all(call["targetoptions"]["chip"] == "sm_90a" for call in compiler_stub)


def test_context_reset_and_handle_reuse_get_fresh_native_state(contexts, compiler_stub):
    a, b, local = contexts
    dispatch = make_dispatcher()
    dispatch._get_context_dispatcher()
    first = dispatch.compile((types.int32,))
    native = dispatch._get_context_dispatcher()._c
    token = current_context_token()
    # The reset hook removes this entry even when the Context and handle survive.
    a.extras.pop("numba_cuda_mlir.context_token")
    assert current_context_token() is not token
    second = dispatch.compile((types.int32,))
    assert second is not first
    assert dispatch._get_context_dispatcher()._c is not native
    local.context = SimpleNamespace(device=a.device, handle=a.handle, extras={})
    assert dispatch.compile((types.int32,)) is not second


def test_fixed_signatures_recompile_for_each_device(contexts, compiler_stub):
    a, b, local = contexts
    dispatch = make_dispatcher()
    first = dispatch.compile((types.float32,))
    dispatch.disable_compile()
    local.context = b
    target = dispatch._get_context_dispatcher()
    assert not target._can_compile
    assert target.overloads[(types.float32,)] is not first
    assert target.overloads[(types.float32,)].metadata["gpu_target"]["chip"] == "sm_120"
    assert target.signatures == [(types.float32,)]


def test_retained_configuration_preserves_per_device_smem(contexts, compiler_stub, monkeypatch):
    a, b, local = contexts
    launches = []

    class Extension:
        uses_launch_config = True

        def prepare_args(self, ty, val, stream=None, retr=None):
            return ty, val

    def launch_config(native, grid, block, stream, sharedmem, cluster):
        return lambda *args: launches.append((native, sharedmem, args))

    monkeypatch.setattr(descriptor, "LaunchConfiguration", launch_config)
    dispatch = make_dispatcher(extensions=[Extension()])
    configured = dispatch[1, 32]
    configured(1)
    first = launches[-1]
    local.context = b
    configured(1)
    second = launches[-1]
    assert first[0] is not second[0]
    assert (first[1], second[1]) == (9 * 128, 12 * 128)
    local.context = a
    configured(1)
    assert launches[-1] == first
    assert len(compiler_stub) == 2
    dispatch.recompile()
    configured(1)
    assert launches[-1][0] is not first[0]
    local.context = b
    configured(1)
    assert launches[-1][0] is not second[0]
    assert len(compiler_stub) == 4


def test_serialization_keeps_portable_signatures_only(contexts, compiler_stub):
    a, b, local = contexts
    dispatch = make_dispatcher()
    dispatch.compile((types.int32,))
    dispatch.disable_compile()
    state = dispatch._reduce_states()
    assert "chip" not in state["targetoptions"]
    assert all("context" not in key for key in state)
    local.context = b
    state["uuid"] = str(uuid4())
    rebuilt = descriptor.MLIRDispatcher._rebuild(**state)
    assert rebuilt.overloads[(types.int32,)].metadata["gpu_target"]["chip"] == "sm_120"
    assert not rebuilt._can_compile


def test_concurrent_device_selection_keeps_state_separate(contexts, compiler_stub):
    a, b, local = contexts
    dispatch = make_dispatcher()
    barrier = threading.Barrier(2)

    def compile_on(context):
        local.context = context
        barrier.wait(timeout=10)
        dispatch._get_context_dispatcher()
        result = dispatch.compile((types.int32,))
        return result, dispatch._get_context_dispatcher()._c

    with ThreadPoolExecutor(max_workers=2) as pool:
        fa, fb = (pool.submit(compile_on, context) for context in (a, b))
        (ra, na), (rb, nb) = fa.result(timeout=10), fb.result(timeout=10)
    assert ra.metadata["gpu_target"]["chip"] == "sm_90"
    assert rb.metadata["gpu_target"]["chip"] == "sm_120"
    assert na is not nb
    assert len(compiler_stub) == 2


def test_code_library_reload_after_reset(contexts, monkeypatch):
    from cuda.bindings import driver

    a, b, local = contexts
    modules = []
    ok = driver.CUresult.CUDA_SUCCESS

    def load(data):
        module = object()
        modules.append(module)
        return ok, module

    monkeypatch.setattr(driver, "cuModuleLoadData", load)
    monkeypatch.setattr(driver, "cuModuleGetFunction", lambda module, name: (ok, module))
    library = CodeLibrary(b"cubin", "kernel")
    fa = library.get_cufunc()
    local.context = b
    fb = library.get_cufunc()
    local.context = a
    assert library.get_cufunc() is fa
    assert fb is not fa
    a.extras.pop("numba_cuda_mlir.context_token")
    assert library.get_cufunc() is not fa
    assert len(modules) == 3


def test_explicit_target_compile_does_not_acquire_runtime_context(monkeypatch, compiler_stub):
    cc = [(9, 0)]
    monkeypatch.setattr(
        tools,
        "get_gpu_compute_capability",
        lambda as_type=str: cc[0] if as_type is tuple else tools.format_arch(cc[0]),
    )
    monkeypatch.setattr(
        descriptor,
        "current_context_token",
        lambda: pytest.fail("compile-only path acquired a CUDA context"),
    )
    dispatch = make_dispatcher(chip="sm_90")
    first = dispatch.compile((types.int32,))
    cc[0] = (12, 0)
    second = dispatch.compile((types.int32,))
    assert first is not second
    assert second.metadata["gpu_target"]["linker_arch"] == "sm_120"
    cc[0] = (9, 0)
    assert dispatch.compile((types.int32,)) is first
    assert len(compiler_stub) == 2
    assert not dispatch._context_dispatchers


def test_nrt_allocator_state_belongs_to_context(contexts):
    from numba_cuda_mlir.memory_management import rtsys

    a, b, local = contexts
    allocation_a, library_a = object(), object()
    rtsys._memsys = allocation_a
    rtsys._memsys_library = library_a
    rtsys._initialized = True
    local.context = b
    assert rtsys._memsys is None
    assert rtsys._memsys_library is None
    assert not rtsys._initialized
    allocation_b = object()
    rtsys._memsys = allocation_b
    local.context = a
    assert rtsys._memsys is allocation_a
    assert rtsys._memsys_library is library_a
    assert rtsys._initialized
    # A context can be reset while a different context is current.
    rtsys.close(b)
    assert rtsys._memsys is allocation_a
    local.context = b
    assert rtsys._memsys is None


def test_explicit_target_launch_compile_does_not_acquire_context(monkeypatch, compiler_stub):
    monkeypatch.setattr(
        tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    monkeypatch.setattr(
        descriptor,
        "current_context_token",
        lambda: pytest.fail("launch-config compilation acquired a CUDA context"),
    )
    dispatch = make_dispatcher(chip="sm_90")
    key = descriptor._launch_config_key(
        {"grid": (1, 1, 1), "block": (32, 1, 1), "sharedmem": 0, "cluster": None}
    )
    result = dispatch._compile_launch_config_signature((types.int32,), key)
    assert result.metadata["gpu_target"]["chip"] == "sm_90"
    assert dispatch._compile_launch_config_signature((types.int32,), key) is result
    assert not dispatch._context_dispatchers


def test_context_reset_expires_dispatch_and_allocator_state(contexts):
    from numba_cuda_mlir.memory_management import rtsys
    from numba_cuda_mlir.numba_cuda.cudadrv.driver import Context

    a, b, local = contexts
    old_token = current_context_token()
    rtsys._memsys = object()
    a.memory_manager = SimpleNamespace(reset=lambda: None)
    a.modules = {}
    a.deallocations = SimpleNamespace(clear=lambda: None)
    Context.reset(a)
    assert current_context_token() is not old_token
    assert rtsys._memsys is None


def test_production_explicit_target_compile_stays_offline(monkeypatch):
    from numba_cuda_mlir import cuda

    monkeypatch.setattr(
        tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    monkeypatch.setattr(
        descriptor,
        "current_context_token",
        lambda: pytest.fail("offline compilation acquired a CUDA context"),
    )

    @cuda.jit(chip="sm_90")
    def write(out, value):
        out[0] = value

    signature = (types.int32[::1], types.int32)
    result = write.compile(signature)
    assert result.metadata["cubin"]
    assert result.metadata["gpu_target"]["chip"] == "sm_90"
    key = descriptor._launch_config_key(
        {"grid": (1, 1, 1), "block": (32, 1, 1), "sharedmem": 0, "cluster": None}
    )
    launch_result = write._compile_launch_config_signature(signature, key)
    assert launch_result.metadata["cubin"]
    assert write.targetoptions["chip"] == "sm_90"


def test_lookup_and_compilation_use_one_capability_snapshot(monkeypatch, compiler_stub):
    calls = []

    def capability(as_type=str):
        calls.append(as_type)
        return (9, 0) if len(calls) == 1 else (12, 0)

    monkeypatch.setattr(tools, "get_gpu_compute_capability", capability)
    dispatch = make_dispatcher()
    result = dispatch.compile((types.int32,))
    assert len(calls) == 1
    assert result.metadata["gpu_target"]["chip"] == "sm_90"
    assert result.metadata["gpu_target"]["linker_arch"] == "sm_90"
    assert dict(next(iter(dispatch._compile_dispatchers)))["chip"] == "sm_90"


def test_compile_after_all_contexts_expire_uses_new_target(contexts, compiler_stub):
    a, b, local = contexts
    dispatch = make_dispatcher()
    dispatch._get_context_dispatcher()
    first = dispatch.compile((types.int32,))
    a.extras.clear()
    assert not dispatch._context_dispatchers
    local.context = b
    second = dispatch.compile((types.int32,))
    assert second is not first
    assert second.metadata["gpu_target"]["chip"] == "sm_120"


def test_fixed_literal_signatures_keep_native_literal_flags(contexts, compiler_stub, monkeypatch):
    a, b, local = contexts
    monkeypatch.setattr(
        descriptor.MLIRDispatcher,
        "_new_kernel_dispatcher",
        lambda self: SimpleNamespace(literal_positions=self._literal_arg_positions),
    )
    dispatch = make_dispatcher()
    dispatch._get_context_dispatcher()
    dispatch._literal_arg_positions = frozenset({0})
    dispatch._requires_launch_config = True
    key = descriptor._launch_config_key(
        {"grid": (1, 1, 1), "block": (32, 1, 1), "sharedmem": 0, "cluster": None}
    )
    dispatch._compile_launch_config_signature((descriptor.types.literal(7),), key)
    dispatch.disable_compile()
    local.context = b
    state = dispatch._get_context_dispatcher()
    assert state._c.literal_positions == frozenset({0})
    assert not state._can_compile


def test_copied_context_cannot_inherit_a_compile_target(contexts, compiler_stub):
    from contextvars import copy_context

    a, b, local = contexts
    with tools._gpu_target_scope():
        inherited = copy_context()

    def compile_on_b():
        local.context = b
        return inherited.run(make_dispatcher().compile, (types.int32,))

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(compile_on_b).result(timeout=10)
    assert result.metadata["gpu_target"]["chip"] == "sm_120"
