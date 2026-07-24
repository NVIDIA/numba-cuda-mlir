# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dispatcher launch metadata plumbing."""

import threading
from uuid import uuid4

import pytest
import numpy as np

from numba_cuda_mlir import descriptor as descriptor_mod
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda.experimental import consteval, current_target_options
from numba_cuda_mlir.descriptor import _ArgMarshaller
from numba_cuda_mlir._whole_function_planners import _RequireLaunchConfig
from numba_cuda_mlir.numba_cuda import types, typing as cuda_typing


class _Dispatcher:
    def __init__(self, targetoptions=None):
        self.targetoptions = {} if targetoptions is None else dict(targetoptions)
        self._can_compile = True
        self.overloads = {}
        self._launch_config_lock = threading.RLock()
        self.remembered_dispatchers = []

    def _remember_kernel_dispatcher(
        self,
        launch_config,
        kernel_dispatcher,
        launch_config_generation=None,
        replace_existing=True,
    ):
        active_launch_config = getattr(descriptor_mod._compile_arg_types, "launch_config", None)
        self.remembered_dispatchers.append(
            (
                launch_config,
                kernel_dispatcher,
                active_launch_config,
                launch_config_generation,
            )
        )
        return True


class _LaunchConfigExtension:
    uses_launch_config = True

    def prepare_args(self, ty, val, stream=None, retr=None):
        return ty, val


class _NonLaunchConfigExtension:
    uses_launch_config = False

    def prepare_args(self, ty, val, stream=None, retr=None):
        return ty, val


class _UnhashableLaunchConfigExtension(_LaunchConfigExtension):
    __hash__ = None


class _CompileResult:
    objectmode = False

    def __init__(self, sig_args, ptx="ptx"):
        self.signature = cuda_typing.signature(types.none, *sig_args)
        self.metadata = {"ptx": ptx}


@pytest.fixture(autouse=True)
def restore_compile_arg_types():
    # The launch metadata thread-local is only mutated from the test thread in
    # this file, so restoring the current thread's local dict is sufficient.
    state = descriptor_mod._compile_arg_types.__dict__.copy()
    yield
    descriptor_mod._compile_arg_types.__dict__.clear()
    descriptor_mod._compile_arg_types.__dict__.update(state)


def test_arg_marshaller_exposes_launch_config_during_launch():
    dispatcher = _Dispatcher()
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 128,
        "cluster": None,
    }
    observed = []

    def launcher():
        observed.append(getattr(descriptor_mod._compile_arg_types, "launch_config", None))
        return "launched"

    marshaller = _ArgMarshaller(
        launcher,
        dispatcher=dispatcher,
        launch_config=launch_config,
    )

    assert marshaller() == "launched"
    assert observed == [launch_config]
    assert "__launch_config__" not in dispatcher.targetoptions
    assert not hasattr(descriptor_mod._compile_arg_types, "launch_config")


def test_arg_marshaller_restores_launch_config_after_error():
    original_launch_config = {"block": (16, 1, 1)}
    dispatcher = _Dispatcher()
    descriptor_mod._compile_arg_types.launch_config = original_launch_config
    launch_config = {
        "grid": (1, 1, 1),
        "block": (64, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    observed = []

    def launcher():
        observed.append(getattr(descriptor_mod._compile_arg_types, "launch_config", None))
        raise ValueError("launch failed")

    marshaller = _ArgMarshaller(
        launcher,
        dispatcher=dispatcher,
        launch_config=launch_config,
    )

    with pytest.raises(ValueError, match="launch failed"):
        marshaller()

    assert observed == [launch_config]
    assert descriptor_mod._compile_arg_types.launch_config == original_launch_config


def test_arg_marshaller_clears_launch_config_after_error():
    dispatcher = _Dispatcher()
    launch_config = {
        "grid": (1, 1, 1),
        "block": (64, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }

    def launcher():
        raise ValueError("launch failed")

    marshaller = _ArgMarshaller(
        launcher,
        dispatcher=dispatcher,
        launch_config=launch_config,
    )

    with pytest.raises(ValueError, match="launch failed"):
        marshaller()

    assert not hasattr(descriptor_mod._compile_arg_types, "launch_config")


def test_arg_marshaller_without_launch_config_leaves_thread_local_absent():
    observed = []

    def launcher():
        observed.append(hasattr(descriptor_mod._compile_arg_types, "launch_config"))
        return "launched"

    marshaller = _ArgMarshaller(launcher)

    assert marshaller() == "launched"
    assert observed == [False]
    assert not hasattr(descriptor_mod._compile_arg_types, "launch_config")


def test_arg_marshaller_without_launch_config_clears_outer_thread_local_temporarily():
    outer_launch_config = {"block": (16, 1, 1)}
    descriptor_mod._compile_arg_types.launch_config = outer_launch_config
    observed = []

    def launcher():
        observed.append(hasattr(descriptor_mod._compile_arg_types, "launch_config"))
        return "launched"

    marshaller = _ArgMarshaller(launcher)

    assert marshaller() == "launched"
    assert observed == [False]
    assert descriptor_mod._compile_arg_types.launch_config is outer_launch_config


def test_arg_marshaller_exposes_available_config_without_activating_it():
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": None,
        "cluster": None,
    }
    observed = []

    def launcher():
        observed.append(
            (
                getattr(descriptor_mod._compile_arg_types, "launch_config", None),
                getattr(
                    descriptor_mod._compile_arg_types,
                    "available_launch_config",
                    None,
                ),
            )
        )
        return "launched"

    marshaller = _ArgMarshaller(
        launcher,
        available_launch_config=launch_config,
    )

    assert marshaller() == "launched"
    assert observed == [(None, launch_config)]
    assert not hasattr(descriptor_mod._compile_arg_types, "launch_config")
    assert not hasattr(descriptor_mod._compile_arg_types, "available_launch_config")


def test_arg_marshaller_rebinds_to_requested_launch_configuration(monkeypatch):
    generic_kernel_dispatcher = object()
    launch_kernel_dispatcher = object()
    available_launch_config = {
        "grid": (2, 1, 1),
        "block": (64, 1, 1),
        "sharedmem": None,
        "cluster": (2, 1, 1),
    }
    active_launch_config = {**available_launch_config, "sharedmem": 256}
    dispatcher = _Dispatcher()
    dispatcher._requires_launch_config = True
    prepare_calls = []
    launches = []

    def prepare_for_launch(
        args,
        argtypes,
        launch_config,
        configured_launch_config,
        configured_kernel_dispatcher,
        configured_launch_config_generation,
    ):
        prepare_calls.append(
            (
                args,
                argtypes,
                launch_config,
                configured_launch_config,
                configured_kernel_dispatcher,
                configured_launch_config_generation,
            )
        )
        return launch_kernel_dispatcher, 7, active_launch_config, 0

    def launch_configuration(
        kernel_dispatcher,
        griddim,
        blockdim,
        stream,
        sharedmem,
        cluster,
    ):
        launches.append(
            (
                kernel_dispatcher,
                griddim,
                blockdim,
                stream,
                sharedmem,
                cluster,
            )
        )

        def launch():
            assert descriptor_mod._compile_arg_types.launch_config == active_launch_config
            return "qualified"

        return launch

    dispatcher._prepare_for_launch = prepare_for_launch
    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    marshaller = _ArgMarshaller(
        lambda: pytest.fail("generic launcher should not run"),
        dispatcher=dispatcher,
        available_launch_config=available_launch_config,
        kernel_dispatcher=generic_kernel_dispatcher,
        launch_stream=123,
    )

    assert marshaller() == "qualified"
    assert marshaller() == "qualified"
    assert prepare_calls == [
        ((), (), available_launch_config, None, generic_kernel_dispatcher, None),
        (
            (),
            (),
            available_launch_config,
            active_launch_config,
            launch_kernel_dispatcher,
            7,
        ),
    ]
    assert launches == [
        (
            launch_kernel_dispatcher,
            (2, 1, 1),
            (64, 1, 1),
            123,
            256,
            (2, 1, 1),
        )
    ]
    assert marshaller._launch_config == active_launch_config
    assert marshaller._launch_config_generation == 7
    assert dispatcher.remembered_dispatchers == [
        (active_launch_config, launch_kernel_dispatcher, active_launch_config, 7)
    ]
    assert not hasattr(descriptor_mod._compile_arg_types, "launch_config")
    assert not hasattr(descriptor_mod._compile_arg_types, "available_launch_config")


def test_arg_marshaller_rebinds_for_retained_launch_extension_snapshot(monkeypatch):
    configured_kernel_dispatcher = object()
    refreshed_kernel_dispatcher = object()
    launch_config = {
        "grid": (2, 1, 1),
        "block": (64, 1, 1),
        "sharedmem": 256,
        "cluster": None,
    }
    dispatcher = _Dispatcher()
    dispatcher.extensions = []
    dispatcher._launch_config_generation = 8
    prepare_calls = []

    def prepare_for_launch(*args):
        prepare_calls.append(args)
        return refreshed_kernel_dispatcher, 8, launch_config, 0

    dispatcher._prepare_for_launch = prepare_for_launch
    monkeypatch.setattr(
        descriptor_mod,
        "LaunchConfiguration",
        lambda *args: lambda: "refreshed",
    )

    marshaller = _ArgMarshaller(
        lambda: pytest.fail("stale launcher should not run"),
        extensions=[_LaunchConfigExtension()],
        dispatcher=dispatcher,
        launch_config=launch_config,
        available_launch_config=launch_config,
        kernel_dispatcher=configured_kernel_dispatcher,
        launch_config_generation=7,
    )

    assert marshaller() == "refreshed"
    assert prepare_calls == [
        (
            (),
            (),
            launch_config,
            launch_config,
            configured_kernel_dispatcher,
            7,
        )
    ]
    assert marshaller._kernel_dispatcher is refreshed_kernel_dispatcher
    assert marshaller._launch_config_generation == 8


def test_arg_marshaller_refreshes_launch_config_dispatcher():
    dispatcher = _Dispatcher()
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    kernel_dispatcher = object()

    marshaller = _ArgMarshaller(
        lambda: None,
        dispatcher=dispatcher,
        launch_config=launch_config,
        kernel_dispatcher=kernel_dispatcher,
    )

    marshaller()
    marshaller()

    assert dispatcher.remembered_dispatchers == [
        (launch_config, kernel_dispatcher, launch_config, None)
    ]


def test_arg_marshaller_raises_and_caches_dynamic_shared_memory_minimum(monkeypatch):
    kernel_dispatcher = object()
    launch_config = {
        "grid": (3, 1, 1),
        "block": (64, 1, 1),
        "sharedmem": 128,
        "cluster": (2, 1, 1),
    }
    dispatcher = _Dispatcher()
    dispatcher._requires_launch_config = True
    launches = []
    original_launches = []

    def prepare_for_launch(*args):
        return kernel_dispatcher, 7, launch_config, 4096

    def launch_configuration(
        native_dispatcher,
        griddim,
        blockdim,
        stream,
        sharedmem,
        cluster,
    ):
        launches.append(
            (
                native_dispatcher,
                griddim,
                blockdim,
                stream,
                sharedmem,
                cluster,
            )
        )
        return lambda value: ("adjusted", value)

    dispatcher._prepare_for_launch = prepare_for_launch
    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)
    marshaller = _ArgMarshaller(
        lambda value: original_launches.append(value),
        dispatcher=dispatcher,
        launch_config=launch_config,
        available_launch_config=launch_config,
        kernel_dispatcher=kernel_dispatcher,
        launch_config_generation=7,
        launch_stream=123,
    )

    assert marshaller(1) == ("adjusted", 1)
    assert marshaller(2) == ("adjusted", 2)
    assert launches == [
        (
            kernel_dispatcher,
            (3, 1, 1),
            (64, 1, 1),
            123,
            4096,
            (2, 1, 1),
        )
    ]
    assert not original_launches
    assert marshaller._launch_config is launch_config
    assert marshaller._available_launch_config is launch_config
    assert launch_config["sharedmem"] == 128


def test_arg_marshaller_preserves_larger_user_shared_memory(monkeypatch):
    kernel_dispatcher = object()
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 8192,
        "cluster": None,
    }
    dispatcher = _Dispatcher()
    dispatcher._requires_launch_config = True
    dispatcher._prepare_for_launch = lambda *args: (
        kernel_dispatcher,
        3,
        launch_config,
        4096,
    )
    monkeypatch.setattr(
        descriptor_mod,
        "LaunchConfiguration",
        lambda *args: pytest.fail("a larger configured sharedmem must be preserved"),
    )
    launches = []
    marshaller = _ArgMarshaller(
        lambda value: launches.append(value) or "original",
        dispatcher=dispatcher,
        launch_config=launch_config,
        available_launch_config=launch_config,
        kernel_dispatcher=kernel_dispatcher,
        launch_config_generation=3,
    )

    assert marshaller(7) == "original"
    assert launches == [7]


def test_arg_marshaller_preserves_raw_sharedmem_without_required_minimum(monkeypatch):
    class RegisteredPlannerRegistry:
        has_planners = True

    kernel_dispatcher = object()
    available_launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": "dynamic",
        "cluster": None,
    }
    dispatcher = _Dispatcher()
    dispatcher._prepare_for_launch = lambda *args: (
        kernel_dispatcher,
        None,
        None,
        0,
    )
    monkeypatch.setattr(
        descriptor_mod,
        "_planner_registry",
        RegisteredPlannerRegistry(),
    )
    launches = []
    marshaller = _ArgMarshaller(
        lambda value: launches.append(value) or "original",
        dispatcher=dispatcher,
        available_launch_config=available_launch_config,
        kernel_dispatcher=kernel_dispatcher,
    )

    assert marshaller(7) == "original"
    assert launches == [7]


def test_arg_marshaller_rejects_raw_sharedmem_when_minimum_requires_adjustment(
    monkeypatch,
):
    class RegisteredPlannerRegistry:
        has_planners = True

    kernel_dispatcher = object()
    available_launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": "dynamic",
        "cluster": None,
    }
    dispatcher = _Dispatcher()
    dispatcher._prepare_for_launch = lambda *args: (
        kernel_dispatcher,
        None,
        None,
        4096,
    )
    monkeypatch.setattr(
        descriptor_mod,
        "_planner_registry",
        RegisteredPlannerRegistry(),
    )
    marshaller = _ArgMarshaller(
        lambda value: pytest.fail("invalid sharedmem must fail before launch"),
        dispatcher=dispatcher,
        available_launch_config=available_launch_config,
        kernel_dispatcher=kernel_dispatcher,
    )

    with pytest.raises(TypeError, match="sharedmem.*integer-convertible"):
        marshaller(7)


def test_configure_records_normalized_launch_config(monkeypatch):
    captured = []

    def launch_configuration(kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster):
        captured.append(
            {
                "griddim": griddim,
                "blockdim": blockdim,
                "stream": stream,
                "sharedmem": sharedmem,
                "cluster": cluster,
            }
        )
        return lambda *args: None

    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    marshaller = dispatcher.configure(2, 32)

    assert captured == [
        {
            "griddim": (2, 1, 1),
            "blockdim": (32, 1, 1),
            "stream": None,
            "sharedmem": 0,
            "cluster": None,
        }
    ]
    assert marshaller._launch_config == {
        "grid": (2, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    assert dispatcher.configure(2, 32, None, 0) is marshaller

    explicit = dispatcher.configure(2, 32, None, 4096)
    assert captured[-1] == {
        "griddim": (2, 1, 1),
        "blockdim": (32, 1, 1),
        "stream": None,
        "sharedmem": 4096,
        "cluster": None,
    }
    assert explicit._launch_config["sharedmem"] == 4096


def test_configure_retains_python_stream_object(monkeypatch):
    captured_streams = []

    class FakeStream:
        def __init__(self, handle):
            self.handle = handle

    def launch_configuration(kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster):
        captured_streams.append(stream)
        return lambda *args: None

    monkeypatch.setattr(descriptor_mod.numba_cuda_driver, "Stream", FakeStream)
    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    stream = FakeStream(123)
    same_handle_stream = FakeStream(123)

    marshaller = dispatcher.configure(2, 32, stream, 0)
    same_stream_marshaller = dispatcher.configure(2, 32, stream, 0)
    same_handle_marshaller = dispatcher.configure(2, 32, same_handle_stream, 0)

    assert captured_streams == [123, 123]
    assert same_stream_marshaller is marshaller
    assert same_handle_marshaller is not marshaller
    assert marshaller._stream_ref is stream
    assert same_handle_marshaller._stream_ref is same_handle_stream


def test_plain_configure_preserves_raw_sharedmem(monkeypatch):
    captured = []

    def launch_configuration(kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster):
        captured.append(sharedmem)
        return lambda *args: None

    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)

    marshaller = dispatcher.configure(2, 32, None, "dynamic")

    assert captured == ["dynamic"]
    assert marshaller._launch_config is None
    assert marshaller._available_launch_config["sharedmem"] == "dynamic"


def test_configure_cache_tracks_mutated_non_launch_extensions(monkeypatch):
    def launch_configuration(kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster):
        return lambda *args: None

    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    generic = dispatcher.configure(1, 32)

    extension = _NonLaunchConfigExtension()
    dispatcher.extensions.append(extension)
    updated = dispatcher.configure(1, 32)

    assert generic._extensions == []
    assert updated is not generic
    assert updated._extensions == [extension]
    assert updated._launch_config is None


def test_prelaunch_uses_retained_launch_extension_snapshot(monkeypatch):
    def launch_configuration(kernel_dispatcher, *args):
        return lambda *launch_args: None

    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel,
        targetoptions={"extensions": [_LaunchConfigExtension()]},
    )
    marshaller = dispatcher.configure(1, 32)
    retained_launch_config = marshaller._launch_config
    retained_kernel_dispatcher = marshaller._kernel_dispatcher
    retained_generation = marshaller._launch_config_generation
    observed_configs = []

    compile_result = _CompileResult((types.int32,))
    compile_result.metadata["required_dynamic_shared_memory"] = 2048

    def compile_result_for(argtypes, launch_config):
        observed_configs.append(launch_config)
        return compile_result

    monkeypatch.setattr(dispatcher, "_compile_result_for", compile_result_for)
    dispatcher.extensions.clear()

    prepared = dispatcher._prepare_for_launch(
        (),
        (types.int32,),
        marshaller._available_launch_config,
        retained_launch_config,
        retained_kernel_dispatcher,
        retained_generation,
    )

    assert prepared == (
        retained_kernel_dispatcher,
        retained_generation,
        retained_launch_config,
        2048,
    )
    assert observed_configs == [retained_launch_config]


def test_launch_config_configure_reports_invalid_sharedmem():
    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )

    with pytest.raises(TypeError, match="sharedmem.*integer-convertible"):
        dispatcher.configure(2, 32, None, object())


def test_configure_accepts_unhashable_extensions(monkeypatch):
    def launch_configuration(kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster):
        return lambda *args: None

    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_UnhashableLaunchConfigExtension()]}
    )

    marshaller = dispatcher.configure(2, 32)

    assert marshaller._launch_config == {
        "grid": (2, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }


def test_plain_configure_reports_unhashable_cache_values():
    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)

    with pytest.raises(TypeError, match="hashable"):
        dispatcher.configure(2, 32, None, [])


def test_shared_memory_carveout_helper_normalizes_strings():
    class Cufunc:
        def __init__(self):
            self.carveout = None

        def set_shared_memory_carveout(self, carveout):
            self.carveout = carveout

    class CodeLibrary:
        def __init__(self):
            self.cufunc = Cufunc()

        def get_cufunc(self):
            return self.cufunc

    class Wrapped:
        def __init__(self):
            self._codelibrary = CodeLibrary()

    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"shared_memory_carveout": "maxshared"}
    )
    wrapped = Wrapped()

    dispatcher._apply_shared_memory_carveout(wrapped)

    assert wrapped._codelibrary.cufunc.carveout == 100

    invalid = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"shared_memory_carveout": "invalid"}
    )
    with pytest.raises(KeyError):
        invalid._apply_shared_memory_carveout(wrapped)


def test_compile_impl_generic_applies_shared_memory_carveout(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"shared_memory_carveout": "maxshared"}
    )
    applied = []

    class CompilerResult:
        signature = cuda_typing.signature(types.none, types.int32)
        metadata = {"cubin": b"generic", "func_name": "kernel"}

    def mlir_compiler_entry(pyfunc, func_args, targetoptions, override_argtypes):
        return CompilerResult()

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)
    monkeypatch.setattr(
        dispatcher,
        "_apply_shared_memory_carveout",
        lambda wrapped: applied.append(wrapped),
    )

    descriptor_mod._compile_arg_types.types = (types.int32,)

    assert dispatcher._compile_impl([1]) == (b"generic", "kernel", False)
    assert len(applied) == 1
    assert dispatcher.overloads[(types.int32,)] is applied[0]


def test_launch_config_key_validation():
    launch_config = {
        "grid": (2, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": None,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)

    assert launch_key == (
        ("grid", (2, 1, 1)),
        ("block", (32, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    assert descriptor_mod._launch_config_dict_from_key(launch_key) == {
        "grid": (2, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    assert descriptor_mod._is_launch_config_dict({"grid": (1, 1, 1), "block": (32, 1, 1)})
    assert not descriptor_mod._is_launch_config_dict({"block": (32, 1, 1)})

    with pytest.raises(ValueError, match="block"):
        descriptor_mod._launch_config_key({"sharedmem": 0, "cluster": None})
    with pytest.raises(TypeError, match="block"):
        descriptor_mod._launch_config_key(
            {"grid": (1, 1, 1), "block": 32, "sharedmem": 0, "cluster": None}
        )
    with pytest.raises(ValueError, match="grid"):
        descriptor_mod._launch_config_key({"block": (32, 1, 1), "sharedmem": 0, "cluster": None})
    with pytest.raises(TypeError, match="grid"):
        descriptor_mod._launch_config_key(
            {
                "grid": 1,
                "block": (32, 1, 1),
                "sharedmem": 0,
                "cluster": None,
            }
        )
    with pytest.raises(TypeError, match="cluster"):
        descriptor_mod._launch_config_key(
            {
                "grid": (1, 1, 1),
                "block": (32, 1, 1),
                "sharedmem": 0,
                "cluster": 1,
            }
        )
    with pytest.raises(TypeError, match="sharedmem"):
        descriptor_mod._launch_config_key(
            {
                "grid": (1, 1, 1),
                "block": (32, 1, 1),
                "sharedmem": object(),
                "cluster": None,
            }
        )


def test_launch_config_uses_distinct_native_dispatchers():
    def kernel(out):
        current_target_options()["__launch_config__"]

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    config_32 = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    config_64 = {
        "grid": (1, 1, 1),
        "block": (64, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    config_grid_2 = {
        "grid": (2, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }

    dispatcher_32 = dispatcher._get_kernel_dispatcher(config_32)

    assert dispatcher_32 is dispatcher._get_kernel_dispatcher(config_32)
    assert dispatcher_32 is not dispatcher._get_kernel_dispatcher(config_64)
    assert dispatcher_32 is not dispatcher._get_kernel_dispatcher(config_grid_2)


def test_plain_kernel_uses_default_native_dispatcher():
    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }

    assert dispatcher._get_kernel_dispatcher(config) is dispatcher._c


def test_launch_config_dispatcher_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(descriptor_mod, "_OLD_DISPATCHER_RETAIN_LIMIT", 1024)

    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )

    first_dispatcher = dispatcher._get_kernel_dispatcher(
        {
            "grid": (1, 1, 1),
            "block": (1, 1, 1),
            "sharedmem": 0,
            "cluster": None,
        }
    )
    first_launch_key = descriptor_mod._launch_config_key(
        {
            "grid": (1, 1, 1),
            "block": (1, 1, 1),
            "sharedmem": 0,
            "cluster": None,
        }
    )
    first_overload_key = ((types.int32,), first_launch_key)
    dispatcher._launch_config_overloads[first_overload_key] = _CompileResult((types.int32,))

    for block_size in range(2, descriptor_mod._LAUNCH_CONFIG_CACHE_SIZE + 1):
        dispatcher._get_kernel_dispatcher(
            {
                "grid": (1, 1, 1),
                "block": (block_size, 1, 1),
                "sharedmem": 0,
                "cluster": None,
            }
        )

    assert len(dispatcher._launch_config_dispatchers) == descriptor_mod._LAUNCH_CONFIG_CACHE_SIZE
    assert first_dispatcher not in dispatcher._old_dispatchers

    dispatcher._get_kernel_dispatcher(
        {
            "grid": (1, 1, 1),
            "block": (descriptor_mod._LAUNCH_CONFIG_CACHE_SIZE + 1, 1, 1),
            "sharedmem": 0,
            "cluster": None,
        }
    )

    assert len(dispatcher._launch_config_dispatchers) == descriptor_mod._LAUNCH_CONFIG_CACHE_SIZE
    assert first_dispatcher in dispatcher._old_dispatchers
    assert first_overload_key not in dispatcher._launch_config_overloads


def test_launch_config_compatible_lookup_reuses_without_alias_growth():
    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    compile_result = _CompileResult((types.int32,))
    dispatcher._launch_config_overloads[((types.int32,), launch_key)] = compile_result

    with dispatcher._launch_config_lock:
        found = dispatcher._find_launch_config_overload_locked(
            (types.int64,),
            launch_key,
        )

    assert found is compile_result
    assert ((types.int64,), launch_key) not in dispatcher._launch_config_overloads


def test_disabled_compile_rejects_nonmatching_launch_specialization():
    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    dispatcher._launch_config_overloads[((types.float32,), launch_key)] = _CompileResult(
        (types.float32,)
    )
    dispatcher.disable_compile()
    descriptor_mod._compile_arg_types.types = (types.complex64,)
    descriptor_mod._compile_arg_types.launch_config = launch_config

    with pytest.raises(TypeError, match="No matching launch-config specialization"):
        dispatcher._compile_impl([1 + 0j])


def test_launch_config_enabled_tracks_mutated_extensions():
    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    assert (
        dispatcher._get_kernel_dispatcher(
            {
                "grid": (1, 1, 1),
                "block": (32, 1, 1),
                "sharedmem": 0,
                "cluster": None,
            }
        )
        is dispatcher._c
    )

    dispatcher.extensions.append(_LaunchConfigExtension())
    configured_dispatcher = dispatcher._get_kernel_dispatcher(
        {
            "grid": (1, 1, 1),
            "block": (32, 1, 1),
            "sharedmem": 0,
            "cluster": None,
        }
    )

    assert configured_dispatcher is not dispatcher._c


def test_configure_cache_tracks_mutated_extensions(monkeypatch):
    def launch_configuration(kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster):
        return lambda *args: None

    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    generic = dispatcher.configure(1, 32)
    assert hasattr(dispatcher.configure, "cache_clear")
    dispatcher.configure.cache_clear()
    generic_after_clear = dispatcher.configure(1, 32)

    dispatcher.extensions.append(_LaunchConfigExtension())
    launch_sensitive = dispatcher.configure(1, 32)

    assert generic._launch_config is None
    assert generic_after_clear is not generic
    assert launch_sensitive is not generic
    assert launch_sensitive._launch_config == {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    assert launch_sensitive._kernel_dispatcher is not dispatcher._c


def test_configure_uses_extension_snapshot_if_extensions_mutate_during_miss(monkeypatch):
    launch_extension = _LaunchConfigExtension()
    replacement_extension = _NonLaunchConfigExtension()

    def launch_configuration(kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster):
        dispatcher.extensions[:] = [replacement_extension]
        return lambda *args: None

    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [launch_extension]}
    )

    marshaller = dispatcher.configure(1, 32)

    assert dispatcher.extensions == [replacement_extension]
    assert marshaller._extensions == [launch_extension]
    assert marshaller._launch_config == {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }


def test_launch_config_dispatcher_cache_retains_boundary_before_eviction():
    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )

    for block_size in range(1, descriptor_mod._LAUNCH_CONFIG_CACHE_SIZE + 1):
        dispatcher._get_kernel_dispatcher(
            {
                "grid": (1, 1, 1),
                "block": (block_size, 1, 1),
                "sharedmem": 0,
                "cluster": None,
            }
        )

    assert len(dispatcher._launch_config_dispatchers) == descriptor_mod._LAUNCH_CONFIG_CACHE_SIZE
    assert not dispatcher._old_dispatchers


def test_retained_marshaller_reregisters_after_cache_eviction(monkeypatch):
    def launch_configuration(kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster):
        return lambda *args: None

    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    marshaller = dispatcher.configure(1, 1)
    launch_key = descriptor_mod._launch_config_key(marshaller._launch_config)
    retained_kernel_dispatcher = marshaller._kernel_dispatcher

    marshaller()
    assert dispatcher._launch_config_dispatchers[launch_key] is retained_kernel_dispatcher

    for block_size in range(2, descriptor_mod._LAUNCH_CONFIG_CACHE_SIZE + 2):
        dispatcher._get_kernel_dispatcher(
            {
                "grid": (1, 1, 1),
                "block": (block_size, 1, 1),
                "sharedmem": 0,
                "cluster": None,
            }
        )

    assert launch_key not in dispatcher._launch_config_dispatchers

    marshaller()

    assert dispatcher._launch_config_dispatchers[launch_key] is retained_kernel_dispatcher


def test_retained_marshaller_does_not_reregister_after_recompile(monkeypatch):
    def launch_configuration(kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster):
        return lambda *args: None

    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    marshaller = dispatcher.configure(1, 32)
    old_kernel_dispatcher = marshaller._kernel_dispatcher

    dispatcher.recompile()
    marshaller()
    fresh_marshaller = dispatcher.configure(1, 32)

    assert old_kernel_dispatcher not in dispatcher._launch_config_dispatchers.values()
    assert fresh_marshaller._kernel_dispatcher is not old_kernel_dispatcher
    assert fresh_marshaller._kernel_dispatcher in dispatcher._launch_config_dispatchers.values()


def test_configure_discards_marshaller_if_recompile_advances_generation(monkeypatch):
    captured_dispatchers = []

    def launch_configuration(kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster):
        captured_dispatchers.append(kernel_dispatcher)
        if len(captured_dispatchers) == 1:
            dispatcher.recompile()
        return lambda *args: None

    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )

    marshaller = dispatcher.configure(1, 32)

    assert len(captured_dispatchers) == 2
    assert captured_dispatchers[0] not in dispatcher._launch_config_dispatchers.values()
    assert marshaller._launch_config_generation == dispatcher._launch_config_generation
    assert marshaller._kernel_dispatcher in dispatcher._launch_config_dispatchers.values()


def test_concurrent_configure_same_key_builds_single_marshaller(monkeypatch):
    def launch_configuration(kernel_dispatcher, griddim, blockdim, stream, sharedmem, cluster):
        return lambda *args: None

    monkeypatch.setattr(descriptor_mod, "LaunchConfiguration", launch_configuration)

    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    original_configure_cached = dispatcher._configure_cached
    calls = []
    first_miss_started = threading.Event()
    release_first_miss = threading.Event()
    second_wait_started = threading.Event()
    thread_timeout = 10

    def configure_cached(*args):
        calls.append(args)
        first_miss_started.set()
        release_first_miss.wait(timeout=thread_timeout)
        return original_configure_cached(*args)

    monkeypatch.setattr(dispatcher, "_configure_cached", configure_cached)
    results = []

    def configure_kernel():
        results.append(dispatcher.configure(1, 32))

    first_thread = threading.Thread(target=configure_kernel)
    first_thread.start()
    assert first_miss_started.wait(timeout=thread_timeout)
    with dispatcher._launch_config_lock:
        [inflight] = dispatcher._configure_cache_inflight.values()
    original_wait = inflight.wait
    wait_timeouts = []

    def observe_wait(timeout=None):
        wait_timeouts.append(timeout)
        second_wait_started.set()
        return original_wait(timeout=timeout)

    monkeypatch.setattr(inflight, "wait", observe_wait)

    second_thread = threading.Thread(target=configure_kernel)
    second_thread.start()
    assert second_wait_started.wait(timeout=thread_timeout)
    release_first_miss.set()

    first_thread.join(timeout=thread_timeout)
    second_thread.join(timeout=thread_timeout)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert wait_timeouts == [None]
    assert len(calls) == 1
    assert len(results) == 2
    assert results[0] is results[1]


def test_stale_launch_generation_does_not_reregister_dispatcher():
    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    stale_dispatcher, stale_generation = dispatcher._get_kernel_dispatcher_and_generation(
        launch_config
    )

    dispatcher.recompile()
    dispatcher._remember_kernel_dispatcher(
        launch_config,
        stale_dispatcher,
        stale_generation,
    )

    assert stale_dispatcher not in dispatcher._launch_config_dispatchers.values()


def test_dispatcher_creation_retries_if_generation_advances(monkeypatch):
    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    created_dispatchers = []

    def new_kernel_dispatcher():
        kernel_dispatcher = object()
        created_dispatchers.append(kernel_dispatcher)
        if len(created_dispatchers) == 1:
            with dispatcher._launch_config_lock:
                dispatcher._launch_config_generation += 1
        return kernel_dispatcher

    monkeypatch.setattr(dispatcher, "_new_kernel_dispatcher", new_kernel_dispatcher)

    kernel_dispatcher, generation = dispatcher._get_kernel_dispatcher_and_generation(launch_config)

    assert len(created_dispatchers) == 2
    assert created_dispatchers[1] is kernel_dispatcher
    assert generation == dispatcher._launch_config_generation
    assert dispatcher._launch_config_dispatchers[launch_key] is kernel_dispatcher
    assert created_dispatchers[0] not in dispatcher._launch_config_dispatchers.values()


def test_retained_launch_dispatcher_replaces_duplicate_entry(monkeypatch):
    monkeypatch.setattr(descriptor_mod, "_OLD_DISPATCHER_RETAIN_LIMIT", 1024)

    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    retained = dispatcher._get_kernel_dispatcher(launch_config)
    duplicate = object()
    dispatcher._launch_config_dispatchers[launch_key] = duplicate

    dispatcher._remember_kernel_dispatcher(
        launch_config,
        retained,
        dispatcher._launch_config_generation,
    )

    assert dispatcher._launch_config_dispatchers[launch_key] is retained
    assert duplicate in dispatcher._old_dispatchers


def test_retained_marshaller_reregister_does_not_replace_newer_dispatcher():
    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    newer_dispatcher = object()
    retained_dispatcher = object()
    dispatcher._launch_config_dispatchers[launch_key] = newer_dispatcher

    registered = dispatcher._remember_kernel_dispatcher(
        launch_config,
        retained_dispatcher,
        dispatcher._launch_config_generation,
        replace_existing=False,
    )

    assert registered is False
    assert dispatcher._launch_config_dispatchers[launch_key] is newer_dispatcher
    assert retained_dispatcher not in dispatcher._old_dispatchers


def test_inspection_keys_keep_generic_overload_shape():
    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    generic_sig = (types.float32,)
    launch_sig = (types.int32,)
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    dispatcher.overloads[generic_sig] = _CompileResult(generic_sig, ptx="generic")
    launch_result = _CompileResult(launch_sig, ptx="launch")
    dispatcher._launch_config_overloads[(launch_sig, launch_key)] = launch_result

    asm = dispatcher.inspect_asm()
    launch_keys = [key for key in asm if isinstance(key, descriptor_mod.LaunchConfigInspectableKey)]

    assert asm[generic_sig] == "generic"
    assert len(launch_keys) == 1
    assert launch_keys[0].argtypes == launch_sig
    assert launch_keys[0].launch_config_key == launch_key
    assert asm[launch_keys[0]] == "launch"
    assert (generic_sig, None) not in asm
    assert launch_keys[0] in dispatcher.signatures

    compatible_key = descriptor_mod.LaunchConfigInspectableKey((types.int64,), launch_key)
    assert dispatcher.inspect_asm(compatible_key) == "launch"
    assert ((types.int64,), launch_key) not in dispatcher._launch_config_overloads
    assert dispatcher.inspect_asm((launch_sig, launch_config)) == "launch"
    assert (
        dispatcher.inspect_asm((launch_sig, {"grid": (1, 1, 1), "block": (32, 1, 1)})) == "launch"
    )
    assert dispatcher.get_metadata(generic_sig) == {"ptx": "generic"}
    assert dispatcher.get_metadata(launch_keys[0]) == {"ptx": "launch"}
    assert dispatcher.get_metadata()[launch_keys[0]] == {"ptx": "launch"}

    missing_key = descriptor_mod.LaunchConfigInspectableKey(
        launch_sig,
        descriptor_mod._launch_config_key(
            {
                "grid": (1, 1, 1),
                "block": (64, 1, 1),
                "sharedmem": 0,
                "cluster": None,
            }
        ),
    )
    with pytest.raises(KeyError, match="No launch-config overload"):
        dispatcher.inspect_asm(missing_key)
    with pytest.raises(KeyError, match="No launch-config overload"):
        dispatcher.inspect_asm((launch_sig, missing_key.launch_config_key))
    with pytest.raises(KeyError, match="No launch-config overload"):
        dispatcher.inspect_asm(
            (
                launch_sig,
                {
                    "grid": (1, 1, 1),
                    "block": (64, 1, 1),
                    "sharedmem": 0,
                    "cluster": None,
                },
            )
        )
    with pytest.raises(KeyError, match="No overload found"):
        dispatcher.inspect_asm((launch_sig, {"block": (32, 1, 1)}))


def test_disabled_launch_config_overload_coerces_to_active_specialization():
    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    dispatcher._launch_config_overloads[((types.float32,), launch_key)] = _CompileResult(
        (types.float32,)
    )
    dispatcher.disable_compile()
    marshaller = _ArgMarshaller(
        lambda: None,
        dispatcher=dispatcher,
        launch_config=launch_config,
    )

    coerced_args, coerced_types = marshaller._coerce_to_overload([7], [types.int64])

    assert coerced_args == [np.float32(7)]
    assert coerced_types == [types.float32]


def test_compile_ignores_stale_launch_config_after_extension_removed():
    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    sig_args = (types.float32,)
    compile_result = _CompileResult(sig_args)
    compile_result.metadata.update({"cubin": b"generic", "func_name": "kernel"})
    dispatcher.overloads[sig_args] = compile_result
    dispatcher.disable_compile()
    dispatcher.extensions.clear()

    descriptor_mod._compile_arg_types.types = sig_args
    descriptor_mod._compile_arg_types.launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }

    assert dispatcher._compile_impl([np.float32(1)]) == (b"generic", "kernel", False)


def test_retained_marshaller_compiles_with_extension_snapshot_after_mutation(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(x):
        pass

    launch_extension = _LaunchConfigExtension()
    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [launch_extension]}
    )
    compile_calls = []

    class CompilerResult:
        signature = cuda_typing.signature(types.none, types.int32)
        metadata = {"cubin": b"launch", "func_name": "kernel"}

    def mlir_compiler_entry(pyfunc, func_args, targetoptions, override_argtypes):
        compile_calls.append((tuple(override_argtypes), dict(targetoptions)))
        return CompilerResult()

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)

    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    marshaller = _ArgMarshaller(
        lambda *args: dispatcher._compile_impl(list(args)),
        extensions=dispatcher.extensions,
        dispatcher=dispatcher,
        launch_config=launch_config,
    )
    dispatcher.extensions.clear()

    with pytest.warns(
        descriptor_mod.NumbaPerformanceWarning,
        match="Persistent disk cache is disabled for launch-config-specialized compiles",
    ):
        assert marshaller._launch((types.int32,), [1]) == (b"launch", "kernel", False)

    assert len(compile_calls) == 1
    assert compile_calls[0][0] == (types.int32,)
    compile_targetoptions = compile_calls[0][1]
    assert compile_targetoptions["extensions"] is marshaller._extensions
    assert compile_targetoptions["extensions"] == [launch_extension]
    assert compile_targetoptions["extensions"] is not dispatcher.extensions
    assert compile_targetoptions["__launch_config__"] == launch_config


def test_disabled_coercion_ignores_stale_launch_config_after_extension_removed():
    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    dispatcher.overloads[(types.float32,)] = _CompileResult((types.float32,))
    dispatcher.disable_compile()
    dispatcher.extensions.clear()

    marshaller = _ArgMarshaller(
        lambda: None,
        dispatcher=dispatcher,
        launch_config={
            "grid": (1, 1, 1),
            "block": (32, 1, 1),
            "sharedmem": 0,
            "cluster": None,
        },
    )

    coerced_args, coerced_types = marshaller._coerce_to_overload([7], [types.int64])

    assert coerced_args == [np.float32(7)]
    assert coerced_types == [types.float32]


def test_forall_uses_launch_config_overload_for_occupancy(monkeypatch):
    from cuda.bindings import driver

    def kernel(x):
        pass

    class Cufunc:
        _handle = object()

    class CodeLibrary:
        def __init__(self):
            self.cufunc = Cufunc()

        def get_cufunc(self):
            return self.cufunc

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    compile_result = _CompileResult((types.int32,))
    compile_result._codelibrary = CodeLibrary()
    dispatcher._launch_config_overloads[((types.int32,), launch_key)] = compile_result
    occupancy_calls = []

    def occupancy(handle, callback, sharedmem, block_limit):
        occupancy_calls.append((handle, callback, sharedmem, block_limit))
        return 1, 256

    monkeypatch.setattr(driver, "cuOccupancyMaxPotentialBlockSize", occupancy)

    launcher = descriptor_mod._ForAll(dispatcher, ntasks=1000, tpb=0, stream=0, sharedmem=48)

    assert launcher._compute_thread_per_block() == 256
    assert occupancy_calls == [(compile_result._codelibrary.cufunc._handle, None, 48, 1024)]


def test_stats_preserves_legacy_shape_and_exposes_launch_config_stats():
    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)

    assert dispatcher.stats._fields == ("cache_path", "cache_hits", "cache_misses")
    assert dispatcher.launch_config_stats.cache_hits is dispatcher._launch_config_cache_hits
    assert dispatcher.launch_config_stats.cache_misses is dispatcher._launch_config_cache_misses


def test_launch_config_accessors_initialize_restored_dispatcher_state():
    dispatcher = descriptor_mod.MLIRDispatcher.__new__(descriptor_mod.MLIRDispatcher)
    dispatcher.overloads = {}

    assert not dispatcher.launch_config_stats.cache_hits
    assert not dispatcher.launch_config_stats.cache_misses
    assert dispatcher.signatures == []


def test_recompile_resets_launch_config_cache_notice():
    def kernel():
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    dispatcher._launch_config_cache_notice_emitted = True

    dispatcher.recompile()

    assert dispatcher._launch_config_cache_notice_emitted is False


def test_compile_impl_launch_config_publishes_aliases_and_skips_disk_cache(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    compile_calls = []
    saved_overloads = []
    trace_messages = []

    class CompilerResult:
        def __init__(self, sig_args, cubin):
            self.signature = cuda_typing.signature(types.none, *sig_args)
            self.metadata = {"cubin": cubin, "func_name": "kernel"}

    def mlir_compiler_entry(pyfunc, func_args, targetoptions, override_argtypes):
        compile_calls.append((tuple(override_argtypes), dict(targetoptions)))
        return CompilerResult(tuple(override_argtypes), f"cubin-{len(compile_calls)}".encode())

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)
    monkeypatch.setattr(
        dispatcher._cache,
        "save_overload",
        lambda *args: saved_overloads.append(args),
    )
    monkeypatch.setattr(descriptor_mod, "trace", lambda message: trace_messages.append(message))

    first_launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    first_launch_key = descriptor_mod._launch_config_key(first_launch_config)
    descriptor_mod._compile_arg_types.types = (types.int32,)
    descriptor_mod._compile_arg_types.launch_config = first_launch_config

    with pytest.warns(
        descriptor_mod.NumbaPerformanceWarning,
        match="Persistent disk cache is disabled for launch-config-specialized compiles",
    ) as warning_records:
        assert dispatcher._compile_impl([1]) == (b"cubin-1", "kernel", False)

        second_launch_config = {
            "grid": (1, 1, 1),
            "block": (64, 1, 1),
            "sharedmem": 0,
            "cluster": None,
        }
        second_launch_key = descriptor_mod._launch_config_key(second_launch_config)
        descriptor_mod._compile_arg_types.types = (types.int64,)
        descriptor_mod._compile_arg_types.launch_config = second_launch_config

        assert dispatcher._compile_impl([2]) == (b"cubin-2", "kernel", False)

    assert len(warning_records) == 1
    assert [call[0] for call in compile_calls] == [(types.int32,), (types.int64,)]
    assert compile_calls[0][1]["extensions"] is dispatcher.extensions
    assert compile_calls[0][1]["__launch_config__"] == first_launch_config
    assert compile_calls[1][1]["extensions"] is dispatcher.extensions
    assert compile_calls[1][1]["__launch_config__"] == second_launch_config
    assert (
        dispatcher._launch_config_overloads[((types.int32,), first_launch_key)].metadata["cubin"]
        == b"cubin-1"
    )
    assert (
        dispatcher._launch_config_overloads[((types.int64,), second_launch_key)].metadata["cubin"]
        == b"cubin-2"
    )
    assert not dispatcher.overloads
    assert saved_overloads == []
    assert trace_messages == [
        "Persistent disk cache is disabled for launch-config-specialized "
        "compiles because the disk cache key does not include launch metadata."
    ]


def test_compile_impl_launch_config_separates_same_signature_by_grid(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    compile_calls = []

    class CompilerResult:
        def __init__(self, cubin):
            self.signature = cuda_typing.signature(types.none, types.int32)
            self.metadata = {"cubin": cubin, "func_name": "kernel"}

    def mlir_compiler_entry(pyfunc, func_args, targetoptions, override_argtypes):
        compile_calls.append((tuple(override_argtypes), dict(targetoptions)))
        return CompilerResult(f"cubin-{len(compile_calls)}".encode())

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)

    first_launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    second_launch_config = {
        "grid": (2, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    first_launch_key = descriptor_mod._launch_config_key(first_launch_config)
    second_launch_key = descriptor_mod._launch_config_key(second_launch_config)

    with pytest.warns(
        descriptor_mod.NumbaPerformanceWarning,
        match="Persistent disk cache is disabled for launch-config-specialized compiles",
    ) as warning_records:
        descriptor_mod._compile_arg_types.types = (types.int32,)
        descriptor_mod._compile_arg_types.launch_config = first_launch_config
        assert dispatcher._compile_impl([1]) == (b"cubin-1", "kernel", False)

        descriptor_mod._compile_arg_types.types = (types.int32,)
        descriptor_mod._compile_arg_types.launch_config = second_launch_config
        assert dispatcher._compile_impl([1]) == (b"cubin-2", "kernel", False)

    assert len(warning_records) == 1
    assert first_launch_key != second_launch_key
    assert [call[0] for call in compile_calls] == [(types.int32,), (types.int32,)]
    assert compile_calls[0][1]["__launch_config__"] == first_launch_config
    assert compile_calls[1][1]["__launch_config__"] == second_launch_config
    assert (
        dispatcher._launch_config_overloads[((types.int32,), first_launch_key)].metadata["cubin"]
        == b"cubin-1"
    )
    assert (
        dispatcher._launch_config_overloads[((types.int32,), second_launch_key)].metadata["cubin"]
        == b"cubin-2"
    )


def test_compile_impl_discards_callbacks_after_generation_retry(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    stale_setup_callback = lambda obj: None
    accepted_setup_callback = lambda obj: None
    compile_calls = []

    class CompilerResult:
        def __init__(self, cubin, setup_callback):
            self.signature = cuda_typing.signature(types.none, types.int32)
            self.metadata = {
                "cubin": cubin,
                "func_name": "kernel",
                "setup_callbacks": [setup_callback],
            }

    def mlir_compiler_entry(pyfunc, func_args, targetoptions, override_argtypes):
        compile_calls.append(dict(targetoptions))
        if len(compile_calls) == 1:
            with dispatcher._launch_config_lock:
                dispatcher._launch_config_generation += 1
            return CompilerResult(b"stale", stale_setup_callback)
        return CompilerResult(b"accepted", accepted_setup_callback)

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)
    descriptor_mod._compile_arg_types.types = (types.int32,)
    descriptor_mod._compile_arg_types.launch_config = launch_config

    with pytest.warns(
        descriptor_mod.NumbaPerformanceWarning,
        match="Persistent disk cache is disabled for launch-config-specialized compiles",
    ):
        assert dispatcher._compile_impl([1]) == (b"accepted", "kernel", False)

    assert len(compile_calls) == 2
    assert stale_setup_callback not in dispatcher._module_setup_callbacks
    assert accepted_setup_callback in dispatcher._module_setup_callbacks


def test_compile_impl_discards_callbacks_from_duplicate_launch_compile(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    losing_setup_callback = lambda obj: None

    class CompilerResult:
        signature = cuda_typing.signature(types.none, types.int32)
        metadata = {
            "cubin": b"loser",
            "func_name": "kernel",
            "setup_callbacks": [losing_setup_callback],
        }

    def mlir_compiler_entry(pyfunc, func_args, targetoptions, override_argtypes):
        winner = _CompileResult((types.int32,))
        winner.metadata.update({"cubin": b"winner", "func_name": "kernel"})
        dispatcher._launch_config_overloads[((types.int32,), launch_key)] = winner
        return CompilerResult()

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)
    descriptor_mod._compile_arg_types.types = (types.int32,)
    descriptor_mod._compile_arg_types.launch_config = launch_config

    assert dispatcher._compile_impl([1]) == (b"winner", "kernel", False)
    assert losing_setup_callback not in dispatcher._module_setup_callbacks


def test_compile_impl_retries_when_planner_requests_launch_config(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    available_launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": None,
        "cluster": None,
    }
    compile_calls = []

    class CompilerResult:
        signature = cuda_typing.signature(types.none, types.int32)
        metadata = {"cubin": b"qualified", "func_name": "kernel"}

    def mlir_compiler_entry(pyfunc, func_args, targetoptions, override_argtypes):
        compile_calls.append(dict(targetoptions))
        if "__launch_config__" not in targetoptions:
            raise _RequireLaunchConfig("launch config required")
        return CompilerResult()

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)
    descriptor_mod._compile_arg_types.types = (types.int32,)
    descriptor_mod._compile_arg_types.available_launch_config = available_launch_config

    with pytest.warns(
        descriptor_mod.NumbaPerformanceWarning,
        match="Persistent disk cache is disabled for launch-config-specialized compiles",
    ):
        result = dispatcher._compile_impl([1])

    normalized_launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(normalized_launch_config)
    assert result == (b"qualified", "kernel", False)
    assert len(compile_calls) == 2
    assert "__launch_config__" not in compile_calls[0]
    assert compile_calls[1]["__launch_config__"] == normalized_launch_config
    assert dispatcher._requires_launch_config is True
    assert not dispatcher.overloads
    assert ((types.int32,), launch_key) in dispatcher._launch_config_overloads


def test_compile_impl_reports_unavailable_or_repeated_launch_request(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(x):
        pass

    def request_launch_config(*args, **kwargs):
        raise _RequireLaunchConfig("launch config required")

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", request_launch_config)
    descriptor_mod._compile_arg_types.types = (types.int32,)
    dispatcher = descriptor_mod.MLIRDispatcher(kernel)

    with pytest.raises(
        RuntimeError, match="not initiated by a configured kernel launch"
    ) as unavailable:
        dispatcher._compile_impl([1])
    assert unavailable.value.__cause__ is None
    assert unavailable.value.__suppress_context__ is True

    descriptor_mod._compile_arg_types.available_launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    with pytest.raises(RuntimeError, match="after a launch-qualified retry") as repeated:
        dispatcher._compile_impl([1])
    assert repeated.value.__cause__ is None
    assert repeated.value.__suppress_context__ is True


def test_compile_impl_preserves_invalid_launch_config_type_error(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(x):
        pass

    def request_launch_config(*args, **kwargs):
        raise _RequireLaunchConfig("launch config required")

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", request_launch_config)
    descriptor_mod._compile_arg_types.types = (types.int32,)
    descriptor_mod._compile_arg_types.available_launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": "dynamic",
        "cluster": None,
    }
    dispatcher = descriptor_mod.MLIRDispatcher(kernel)

    with pytest.raises(TypeError, match="sharedmem.*integer-convertible"):
        dispatcher._compile_impl([1])


def test_disabled_launch_config_reduce_rebuild_restores_launch_sigs(monkeypatch):
    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    sig_args = (types.float32,)
    compile_result = _CompileResult(sig_args)
    dispatcher._launch_config_overloads[(sig_args, launch_key)] = compile_result
    dispatcher.disable_compile()

    compiled = []

    def compile_launch_config_signature(self, sig, launch_config_key):
        compiled.append((tuple(sig.args), launch_config_key, self._can_compile))
        self._launch_config_overloads[(tuple(sig.args), launch_config_key)] = _CompileResult(
            tuple(sig.args)
        )

    monkeypatch.setattr(
        descriptor_mod.MLIRDispatcher,
        "_compile_launch_config_signature",
        compile_launch_config_signature,
    )

    states = dispatcher._reduce_states()
    states["uuid"] = states["uuid"] + "-rebuilt"
    rebuilt = descriptor_mod.MLIRDispatcher._rebuild(**states)

    assert states["sigs"] == []
    assert states["launch_config_sigs"] == [(compile_result.signature, launch_key)]
    assert compiled == [(sig_args, launch_key, True)]
    assert rebuilt._can_compile is False
    assert (sig_args, launch_key) in rebuilt.launch_config_overloads


def test_reduce_rebuild_and_recompile_preserve_learned_launch_requirement():
    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    dispatcher._mark_requires_launch_config()

    states = dispatcher._reduce_states()
    states["uuid"] = str(uuid4())
    rebuilt = descriptor_mod.MLIRDispatcher._rebuild(**states)

    assert states["requires_launch_config"] is True
    assert rebuilt._requires_launch_config is True
    assert rebuilt._launch_config_enabled is True

    rebuilt.recompile()
    assert rebuilt._requires_launch_config is True
    assert rebuilt._launch_config_enabled is True


def test_reduce_omits_stale_generic_sigs_after_learned_launch_requirement(monkeypatch):
    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    generic_result = _CompileResult((types.int32,))
    launch_result = _CompileResult((types.float32,))
    dispatcher.overloads[(types.int32,)] = generic_result
    dispatcher._launch_config_overloads[((types.float32,), launch_key)] = launch_result
    dispatcher._mark_requires_launch_config()
    dispatcher.disable_compile()

    compiled = []

    def compile_launch_config_signature(self, sig, launch_config_key):
        compiled.append((tuple(sig.args), launch_config_key))
        self._launch_config_overloads[(tuple(sig.args), launch_config_key)] = _CompileResult(
            tuple(sig.args)
        )

    monkeypatch.setattr(
        descriptor_mod.MLIRDispatcher,
        "_compile_launch_config_signature",
        compile_launch_config_signature,
    )

    states = dispatcher._reduce_states()
    states["uuid"] = str(uuid4())
    rebuilt = descriptor_mod.MLIRDispatcher._rebuild(**states)

    assert states["sigs"] == []
    assert states["launch_config_sigs"] == [(launch_result.signature, launch_key)]
    assert compiled == [((types.float32,), launch_key)]
    assert rebuilt._requires_launch_config is True
    assert rebuilt._can_compile is False
    assert not rebuilt.overloads
    assert ((types.float32,), launch_key) in rebuilt.launch_config_overloads


def test_compile_launch_config_signature_forces_launch_rebuild_without_extensions(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    compile_calls = []

    class CompilerResult:
        signature = cuda_typing.signature(types.none, types.int32)
        metadata = {"cubin": b"rebuilt", "func_name": "kernel"}

    def mlir_compiler_entry(pyfunc, func_args, targetoptions, override_argtypes):
        compile_calls.append(dict(targetoptions))
        return CompilerResult()

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)

    with pytest.warns(
        descriptor_mod.NumbaPerformanceWarning,
        match="Persistent disk cache is disabled for launch-config-specialized compiles",
    ):
        rebuilt = dispatcher._compile_launch_config_signature(
            cuda_typing.signature(types.none, types.int32),
            launch_key,
        )

    assert rebuilt.metadata["cubin"] == b"rebuilt"
    assert dispatcher.launch_config_overloads[((types.int32,), launch_key)] is rebuilt
    assert len(compile_calls) == 1
    assert compile_calls[0]["extensions"] == []
    assert compile_calls[0]["__launch_config__"] == launch_config
    assert not hasattr(descriptor_mod._compile_arg_types, "force_launch_config")


def test_disabled_launch_config_reduce_skips_launch_sigs_after_extension_removed(
    monkeypatch,
):
    def kernel(x):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )
    launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(launch_config)
    sig_args = (types.float32,)
    compile_result = _CompileResult(sig_args)
    dispatcher._launch_config_overloads[(sig_args, launch_key)] = compile_result
    dispatcher.disable_compile()
    dispatcher.extensions.clear()

    def compile_launch_config_signature(*args):
        raise AssertionError("stale launch-config signatures should not be rebuilt")

    monkeypatch.setattr(
        descriptor_mod.MLIRDispatcher,
        "_compile_launch_config_signature",
        compile_launch_config_signature,
    )

    states = dispatcher._reduce_states()
    states["uuid"] = states["uuid"] + "-rebuilt"
    rebuilt = descriptor_mod.MLIRDispatcher._rebuild(**states)

    assert states["launch_config_sigs"] == []
    assert rebuilt._can_compile is False
    assert not rebuilt.launch_config_overloads


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_launch_config_specializes_same_signature_launches():
    @cuda.jit(extensions=[_LaunchConfigExtension()])
    def kernel(out):
        out[0] = consteval(current_target_options()["__launch_config__"]["block"][0])

    out = np.zeros(1, dtype=np.int32)

    kernel[1, 32](out)
    assert out[0] == 32
    misses_after_first_launch = sum(kernel._launch_config_cache_misses.values())

    kernel[1, 32](out)
    assert out[0] == 32
    assert sum(kernel._launch_config_cache_misses.values()) == misses_after_first_launch

    kernel[2, 32](out)
    assert out[0] == 32
    assert sum(kernel._launch_config_cache_misses.values()) == misses_after_first_launch + 1

    kernel[1, 64](out)
    assert out[0] == 64
    assert sum(kernel._launch_config_cache_misses.values()) == misses_after_first_launch + 2
    assert kernel.signatures
    assert kernel.nopython_signatures
    assert kernel.launch_config_overloads
