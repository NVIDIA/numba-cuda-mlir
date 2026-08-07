# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for lazy compiler-extension initialization."""

from __future__ import annotations

import importlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import pytest


@pytest.fixture
def bootstrap_module(monkeypatch):
    module = importlib.import_module("numba_cuda_mlir._extension_bootstrap")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "_snapshot_registrations", lambda: object())
    monkeypatch.setattr(module, "_restore_registrations", lambda _snapshot: None)
    return module


@dataclass(frozen=True)
class _Distribution:
    name: str


class _EntryPoint:
    name = "init"

    def __init__(self, distribution, module, initializer):
        self.dist = _Distribution(distribution)
        self.module = module
        self.value = f"{module}:init"
        self._initializer = initializer

    def load(self):
        return self._initializer


class _EntryPoints(tuple):
    def select(self, *, group, name):
        assert group == "numba_cuda_mlir_extensions"
        return tuple(entry_point for entry_point in self if entry_point.name == name)


def _install_entry_points(monkeypatch, bootstrap_module, entry_points):
    discoveries = []

    def discover():
        discoveries.append(None)
        return _EntryPoints(entry_points)

    monkeypatch.setattr(bootstrap_module.importlib_metadata, "entry_points", discover)
    return discoveries


def test_import_does_not_discover_extensions(monkeypatch):
    from importlib import metadata

    module_name = "numba_cuda_mlir._extension_bootstrap"
    sys.modules.pop(module_name, None)

    def unexpected_discovery():
        raise AssertionError("extension metadata was discovered during import")

    monkeypatch.setattr(metadata, "entry_points", unexpected_discovery)
    try:
        assert importlib.import_module(module_name) is not None
    finally:
        sys.modules.pop(module_name, None)


def test_no_entry_points_is_cached(bootstrap_module, monkeypatch):
    discoveries = _install_entry_points(monkeypatch, bootstrap_module, ())

    assert bootstrap_module.initialize_extensions() == ()
    assert bootstrap_module.initialize_extensions() == ()
    assert len(discoveries) == 1


def test_entry_point_discovery_failure_is_cached(bootstrap_module, monkeypatch):
    discoveries = []

    def discover():
        discoveries.append(None)
        raise ValueError("metadata is unavailable")

    monkeypatch.setattr(bootstrap_module.importlib_metadata, "entry_points", discover)

    with pytest.raises(
        bootstrap_module.ExtensionInitializationError,
        match="entry-point discovery failed with ValueError: metadata is unavailable",
    ) as first:
        bootstrap_module.initialize_extensions()
    with pytest.raises(bootstrap_module.ExtensionInitializationError) as second:
        bootstrap_module.initialize_extensions()

    assert first.value is second.value
    assert len(discoveries) == 1


def test_registration_snapshot_failure_is_cached_and_clears_recursion_state(
    bootstrap_module,
    monkeypatch,
):
    attempts = []

    def snapshot():
        attempts.append(None)
        raise ValueError("registry is unavailable")

    monkeypatch.setattr(bootstrap_module, "_snapshot_registrations", snapshot)

    with pytest.raises(
        bootstrap_module.ExtensionInitializationError,
        match="registration snapshot failed with ValueError: registry is unavailable",
    ) as first:
        bootstrap_module.initialize_extensions()
    with pytest.raises(bootstrap_module.ExtensionInitializationError) as second:
        bootstrap_module.initialize_extensions()

    assert first.value is second.value
    assert bootstrap_module._initializing_thread is None
    assert attempts == [None]


def test_entry_points_initialize_once_in_deterministic_order(
    bootstrap_module,
    monkeypatch,
):
    initialized = []
    entry_points = (
        _EntryPoint("zeta", "zeta_extension", lambda: initialized.append("zeta")),
        _EntryPoint("Alpha", "alpha_extension", lambda: initialized.append("alpha")),
    )
    discoveries = _install_entry_points(monkeypatch, bootstrap_module, entry_points)

    result = bootstrap_module.initialize_extensions()

    assert initialized == ["alpha", "zeta"]
    assert result == (
        "'Alpha' (alpha_extension:init)",
        "'zeta' (zeta_extension:init)",
    )
    assert bootstrap_module.initialize_extensions() is result
    assert len(discoveries) == 1


def test_concurrent_initialization_waits_for_one_initializer(
    bootstrap_module,
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def initialize():
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(timeout=5)

    _install_entry_points(
        monkeypatch,
        bootstrap_module,
        (_EntryPoint("extension", "extension", initialize),),
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(bootstrap_module.initialize_extensions) for _ in range(4)]
        assert entered.wait(timeout=5)
        release.set()
        results = [future.result(timeout=5) for future in futures]

    assert len(calls) == 1
    assert results == [("'extension' (extension:init)",)] * 4


def test_recursive_initialization_is_rejected_and_failure_is_cached(
    bootstrap_module,
    monkeypatch,
):
    calls = []

    def initialize():
        calls.append(None)
        bootstrap_module.initialize_extensions()

    _install_entry_points(
        monkeypatch,
        bootstrap_module,
        (_EntryPoint("recursive", "recursive_extension", initialize),),
    )

    with pytest.raises(
        bootstrap_module.ExtensionInitializationError,
        match="recursive Numba-CUDA-MLIR extension initialization",
    ) as first:
        bootstrap_module.initialize_extensions()
    with pytest.raises(bootstrap_module.ExtensionInitializationError) as second:
        bootstrap_module.initialize_extensions()

    assert calls == [None]
    assert first.value is second.value
    assert isinstance(first.value.__cause__, RuntimeError)


def test_failure_rolls_back_registrations_and_is_cached(monkeypatch):
    bootstrap = importlib.reload(importlib.import_module("numba_cuda_mlir._extension_bootstrap"))
    from numba_cuda_mlir._whole_function_planners import (
        WholeFunctionPlanner,
        _planner_registry,
    )
    from numba_cuda_mlir.numba_cuda.core.rewrites import Rewrite, rewrite_registry

    with _planner_registry._lock:
        planners_before = tuple(_planner_registry._planners)
    rewrites_before = {
        kind: tuple(rewrites) for kind, rewrites in rewrite_registry.rewrites.items()
    }
    calls = []

    def initialize():
        calls.append(None)

        class FailedPlanner(WholeFunctionPlanner):
            def run(self):
                return False

        class FailedRewrite(Rewrite):
            pass

        _planner_registry.register(FailedPlanner)
        rewrite_registry.register("before-inference")(FailedRewrite)
        raise ValueError("registration failed")

    monkeypatch.setattr(
        bootstrap.importlib_metadata,
        "entry_points",
        lambda: _EntryPoints((_EntryPoint("broken", "broken_extension", initialize),)),
    )

    with pytest.raises(
        bootstrap.ExtensionInitializationError,
        match="broken_extension:init.*ValueError: registration failed",
    ) as first:
        bootstrap.initialize_extensions()
    with pytest.raises(bootstrap.ExtensionInitializationError) as second:
        bootstrap.initialize_extensions()

    with _planner_registry._lock:
        assert tuple(_planner_registry._planners) == planners_before
    assert {
        kind: tuple(rewrites) for kind, rewrites in rewrite_registry.rewrites.items()
    } == rewrites_before
    assert calls == [None]
    assert first.value is second.value
    importlib.reload(bootstrap)


def test_legacy_entry_point_loader_delegates(monkeypatch):
    from numba_cuda_mlir import _extension_bootstrap
    from numba_cuda_mlir.numba_cuda.core import entrypoints

    report = object()
    monkeypatch.setattr(_extension_bootstrap, "initialize_extensions", lambda: report)

    assert entrypoints.init_all() is report


def test_cuda_coop_extension_protocol_is_public():
    import numba_cuda_mlir
    from numba_cuda_mlir import extending

    assert numba_cuda_mlir.CUDA_COOP_EXTENSION_PROTOCOL == 1
    assert "CUDA_COOP_EXTENSION_PROTOCOL" in numba_cuda_mlir.__all__
    assert extending.CUDA_COOP_EXTENSION_PROTOCOL == 1
    assert "CUDA_COOP_EXTENSION_PROTOCOL" in extending.__all__


class _BootstrapReached(Exception):
    pass


@pytest.mark.parametrize(
    "invoke",
    [
        lambda dispatcher: dispatcher._compile_launch_config_signature(None, None),
        lambda dispatcher: dispatcher._prepare_for_launch(None, None, None, None, None, None, None),
        lambda dispatcher: dispatcher._compile(None),
        lambda dispatcher: dispatcher._compile_impl([]),
        lambda dispatcher: dispatcher.compile(None),
        lambda dispatcher: dispatcher._compile_public(None),
        lambda dispatcher: dispatcher._compile_device_callee(None),
        lambda dispatcher: dispatcher._compile_as_device_callee(None),
        lambda dispatcher: dispatcher.compile_device(None),
        lambda dispatcher: dispatcher.compile_for(None),
        lambda dispatcher: dispatcher.specialize(None),
        lambda dispatcher: dispatcher.get_call_template(None, None),
    ],
)
def test_dispatcher_compile_and_cache_roots_bootstrap_first(monkeypatch, invoke):
    from numba_cuda_mlir import descriptor

    def bootstrap():
        raise _BootstrapReached

    monkeypatch.setattr(descriptor, "initialize_extensions", bootstrap)
    dispatcher = object.__new__(descriptor.MLIRDispatcher)

    with pytest.raises(_BootstrapReached):
        invoke(dispatcher)


def test_launch_marshalling_bootstraps_before_type_discovery(monkeypatch):
    from numba_cuda_mlir import descriptor

    monkeypatch.setattr(
        descriptor,
        "initialize_extensions",
        lambda: (_ for _ in ()).throw(_BootstrapReached()),
    )
    marshaller = object.__new__(descriptor._ArgMarshaller)

    with pytest.raises(_BootstrapReached):
        marshaller._call_impl(object())


@pytest.mark.parametrize(
    "invoke",
    [
        lambda compiler: compiler._compile_and_optimize(None),
        lambda compiler: compiler._compile_only(None),
        lambda compiler: compiler._compile(None),
        lambda compiler: compiler.compile_for(None, object()),
        lambda compiler: compiler.compile(None, None),
        lambda compiler: compiler.compile_for_current_device(None, None),
        lambda compiler: compiler.compile_ptx_for_current_device(None, None),
        lambda compiler: compiler.compile_result(None),
        lambda compiler: compiler.compile_mlir(None, None),
        lambda compiler: compiler.compile_cubin(None, None),
    ],
)
def test_direct_compiler_apis_bootstrap_first(monkeypatch, invoke):
    from numba_cuda_mlir import compiler

    monkeypatch.setattr(
        compiler,
        "initialize_extensions",
        lambda: (_ for _ in ()).throw(_BootstrapReached()),
    )

    with pytest.raises(_BootstrapReached):
        invoke(compiler)


@pytest.mark.parametrize(
    "invoke",
    [
        lambda compiler: compiler.compile_mlir(None, None, (), {}),
        lambda compiler: compiler.mlir_compiler_entry(None, [], {}),
    ],
)
def test_low_level_compile_roots_bootstrap_first(monkeypatch, invoke):
    from numba_cuda_mlir import mlir_compiler

    monkeypatch.setattr(
        mlir_compiler,
        "initialize_extensions",
        lambda: (_ for _ in ()).throw(_BootstrapReached()),
    )

    with pytest.raises(_BootstrapReached):
        invoke(mlir_compiler)


def test_inherited_compilation_hook_uses_shared_bootstrap(monkeypatch):
    from numba_cuda_mlir.numba_cuda import dispatcher

    calls = []
    monkeypatch.setattr(dispatcher, "initialize_extensions", lambda: calls.append(None))

    dispatcher._DispatcherBase._compilation_chain_init_hook(object())

    assert calls == [None]


@pytest.mark.parametrize("explicit_signature", [False, True])
def test_eager_annotation_paths_bootstrap_before_extraction(
    monkeypatch,
    explicit_signature,
):
    from numba_cuda_mlir import decorators

    events = []

    def extract(_func):
        events.append("extract")
        raise _BootstrapReached

    monkeypatch.setattr(decorators, "initialize_extensions", lambda: events.append("bootstrap"))
    monkeypatch.setattr(decorators, "_extract_signature_from_annotations", extract)

    if explicit_signature:
        signature = decorators.typing.signature(
            decorators.numba_types.none,
            decorators.numba_types.int32,
        )
        decorate = decorators.mlir_jit(signature)
    else:
        decorate = decorators.mlir_jit()

    def annotated(value: int):
        pass

    with pytest.raises(_BootstrapReached):
        decorate(annotated)

    assert events == ["bootstrap", "extract"]


def test_lazy_unannotated_decorator_does_not_initialize_extensions(monkeypatch):
    from numba_cuda_mlir import decorators

    calls = []
    monkeypatch.setattr(decorators, "initialize_extensions", lambda: calls.append(None))

    dispatcher = decorators.mlir_jit(lambda value: None)

    assert dispatcher.signatures == []
    assert calls == []


def test_cuda_current_device_helpers_use_mlir_compiler_wrappers():
    from numba_cuda_mlir import compiler, cuda

    assert cuda.compile_for_current_device is compiler.compile_for_current_device
    assert cuda.compile_ptx_for_current_device is compiler.compile_ptx_for_current_device
