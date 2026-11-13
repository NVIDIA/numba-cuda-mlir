# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dispatcher launch metadata plumbing."""

from concurrent.futures import ThreadPoolExecutor
import threading
from uuid import uuid4

import pytest
import numpy as np

from numba_cuda_mlir import descriptor as descriptor_mod
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda.experimental import consteval, current_target_options
from numba_cuda_mlir.errors import ForceLiteralArg
from numba_cuda_mlir._launch_config import _LAUNCH_CONFIG_TRACKER_OPTION
from numba_cuda_mlir._whole_function_planners import (
    _planner_registry,
    _RequireLaunchConfig,
)
from numba_cuda_mlir.numba_cuda import types, typing as cuda_typing
from numba_cuda_mlir.numba_cuda.core import errors as cuda_errors


class _LaunchConfigExtension:
    uses_launch_config = True

    def prepare_args(self, ty, val, stream=None, retr=None):
        return ty, val


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


def test_target_initialization_waits_for_concurrent_initialization():
    class Context:
        def __init__(self, started=None, release=None):
            self.started = started
            self.release = release
            self.refresh_count = 0
            self.registry_count = 0

        def refresh(self):
            self.refresh_count += 1
            if self.started is not None:
                self.started.set()
                assert self.release.wait(timeout=10)

        def install_registry(self, registry):
            self.registry_count += 1

    initialized = threading.Event()
    release_initialization = threading.Event()
    target = descriptor_mod.MLIRTarget("test_target_initialization")
    typing_context = Context(initialized, release_initialization)
    target_context = Context()
    target._typingctx = typing_context
    target._targetctx = target_context
    errors = []
    second_finished = threading.Event()

    def ensure_initialized(finished=None):
        try:
            target.ensure_initialized()
        except BaseException as exc:
            errors.append(exc)
        finally:
            if finished is not None:
                finished.set()

    first = threading.Thread(target=ensure_initialized)
    first.start()
    assert initialized.wait(timeout=10)

    second = threading.Thread(target=ensure_initialized, args=(second_finished,))
    second.start()
    assert not second_finished.wait(timeout=0.1)

    release_initialization.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert typing_context.refresh_count == 1
    assert target_context.refresh_count == 1
    assert typing_context.registry_count == 1


def test_launch_config_configure_reports_invalid_sharedmem():
    def kernel(out):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(
        kernel, targetoptions={"extensions": [_LaunchConfigExtension()]}
    )

    with pytest.raises(TypeError, match="sharedmem.*integer-convertible"):
        dispatcher.configure(2, 32, None, object())


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

    dispatcher.overloads[generic_sig].metadata["llvmir"] = "generic llvm"
    launch_result.metadata["llvmir"] = "launch llvm"
    llvm_ir = dispatcher.inspect_llvm()
    assert llvm_ir[generic_sig] == "generic llvm"
    assert llvm_ir[launch_keys[0]] == "launch llvm"

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


def test_literalize_argtypes_matches_numba_scalar_literal_semantics():
    def kernel(count, enabled):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    dispatcher._literal_arg_positions = frozenset({0, 1})

    literalized = dispatcher._literalize_argtypes(
        (types.int64, types.boolean),
        (7, True),
        abi_arg_count=2,
    )

    assert literalized == (types.literal(7), types.literal(True))

    # Learned positions are conditional hints, not permanent restrictions on
    # the signatures this dispatcher may compile.
    assert dispatcher._literalize_argtypes(
        (types.float64, types.boolean),
        (1.5, True),
        abi_arg_count=2,
    ) == (types.float64, types.literal(True))

    class IntSubclass(int):
        pass

    assert dispatcher._literalize_argtypes(
        (types.int64, types.boolean),
        (IntSubclass(7), True),
        abi_arg_count=2,
    ) == (types.int64, types.literal(True))

    with pytest.raises(TypeError, match="top-level Python int and bool"):
        dispatcher._record_literal_arg_positions({0}, (1.5, True), 2)


def test_literalize_argtypes_rejects_flattened_extension_abi():
    def kernel(value):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    dispatcher._literal_arg_positions = frozenset({0})

    with pytest.raises(TypeError, match=r"flattened launch arguments.*issue #60"):
        dispatcher._literalize_argtypes(
            (types.UniTuple(types.int32, 2),),
            ((1, 2),),
            abi_arg_count=2,
        )


def test_concurrent_literal_requests_rebuild_native_dispatcher_once():
    def kernel(selector):
        pass

    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    generation = dispatcher._launch_config_generation

    def record(_):
        return dispatcher._record_literal_arg_positions({0}, (7,), 1)

    with ThreadPoolExecutor(max_workers=8) as executor:
        changed = list(executor.map(record, range(32)))

    assert changed.count(True) == 1
    assert changed.count(False) == 31
    assert dispatcher._literal_arg_positions == frozenset({0})
    assert dispatcher._launch_config_generation == generation + 1


def test_literal_retry_counts_concurrent_discovery_as_attempt_progress(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(selector):
        pass

    compile_calls = []

    class CompilerResult:
        def __init__(self, argtype):
            self.signature = cuda_typing.signature(types.none, argtype)
            self.metadata = {"cubin": b"literal-7", "func_name": "kernel"}

    def mlir_compiler_entry(pyfunc, func_args, targetoptions, override_argtypes):
        [argtype] = override_argtypes
        compile_calls.append(argtype)
        if len(compile_calls) == 1:
            # Model another compile publishing the requested position after
            # this attempt selected its generic argument types.
            assert dispatcher._record_literal_arg_positions({0}, (7,), 1) is True
            raise ForceLiteralArg({0})
        assert isinstance(argtype, types.Literal)
        return CompilerResult(argtype)

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)
    monkeypatch.setattr(_planner_registry, "_planners", [object()])
    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    descriptor_mod._compile_arg_types.types = (types.int32,)
    descriptor_mod._compile_arg_types.values = (7,)

    assert dispatcher._compile_impl([7]) == (b"literal-7", "kernel", False)
    assert compile_calls == [types.int32, types.literal(7)]
    assert dispatcher._literal_arg_positions == frozenset({0})


def test_literal_retry_allows_later_generic_signature(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(selector):
        pass

    compile_calls = []

    class CompilerResult:
        def __init__(self, argtype):
            self.signature = cuda_typing.signature(types.none, argtype)
            self.metadata = {
                "cubin": str(argtype).encode(),
                "func_name": "kernel",
            }

    def mlir_compiler_entry(pyfunc, func_args, targetoptions, override_argtypes):
        [argtype] = override_argtypes
        compile_calls.append(argtype)
        if argtype == types.int32:
            raise ForceLiteralArg({0})
        return CompilerResult(argtype)

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)
    monkeypatch.setattr(_planner_registry, "_planners", [object()])
    dispatcher = descriptor_mod.MLIRDispatcher(kernel)

    descriptor_mod._compile_arg_types.types = (types.int32,)
    descriptor_mod._compile_arg_types.values = (7,)
    dispatcher._compile_impl([7])

    descriptor_mod._compile_arg_types.types = (types.float64,)
    descriptor_mod._compile_arg_types.values = (1.5,)
    dispatcher._compile_impl([1.5])
    descriptor_mod._compile_arg_types.values = (2.5,)
    dispatcher._compile_impl([2.5])

    assert compile_calls == [types.int32, types.literal(7), types.float64]
    assert set(dispatcher.overloads) == {(types.literal(7),), (types.float64,)}


def test_literal_retry_preserves_same_attempt_launch_promotion(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(selector):
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
        def __init__(self, argtype):
            self.signature = cuda_typing.signature(types.none, argtype)
            self.metadata = {
                "cubin": f"literal-{argtype.literal_value}".encode(),
                "func_name": "kernel",
                "required_dynamic_shared_memory": 4096,
            }

    def mlir_compiler_entry(pyfunc, func_args, targetoptions, override_argtypes):
        call = (tuple(func_args), tuple(override_argtypes), dict(targetoptions))
        compile_calls.append(call)
        if len(compile_calls) == 1:
            tracker = targetoptions.pop(_LAUNCH_CONFIG_TRACKER_OPTION)
            targetoptions["__launch_config__"] = tracker.require()
            raise ForceLiteralArg({0})
        [argtype] = override_argtypes
        assert isinstance(argtype, types.Literal)
        return CompilerResult(argtype)

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)
    descriptor_mod._compile_arg_types.types = (types.int32,)
    descriptor_mod._compile_arg_types.values = (7,)
    descriptor_mod._compile_arg_types.available_launch_config = available_launch_config

    with pytest.warns(
        descriptor_mod.NumbaPerformanceWarning,
        match="Persistent disk cache is disabled for launch-config-specialized compiles",
    ):
        first = dispatcher._compile_impl([7])

    descriptor_mod._compile_arg_types.values = (9,)
    second = dispatcher._compile_impl([9])

    normalized_launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(normalized_launch_config)
    literal7 = types.literal(7)
    literal9 = types.literal(9)

    assert first == (b"literal-7", "kernel", False)
    assert second == (b"literal-9", "kernel", False)
    assert [call[0] for call in compile_calls] == [(7,), (7,), (9,)]
    assert [call[1] for call in compile_calls] == [
        (types.int32,),
        (literal7,),
        (literal9,),
    ]
    assert "__launch_config__" not in compile_calls[0][2]
    assert compile_calls[0][2][_LAUNCH_CONFIG_TRACKER_OPTION].required is True
    assert all(
        call[2].get("__launch_config__") == normalized_launch_config for call in compile_calls[1:]
    )
    assert dispatcher._literal_arg_positions == frozenset({0})
    assert ((literal7,), launch_key) in dispatcher._launch_config_overloads
    assert ((literal9,), launch_key) in dispatcher._launch_config_overloads


def test_literal_retry_accumulates_staged_requests_with_launch_promotion(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(first, second):
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
        def __init__(self, argtypes):
            self.signature = cuda_typing.signature(types.none, *argtypes)
            self.metadata = {
                "cubin": b"literal-7-11",
                "func_name": "kernel",
                "required_dynamic_shared_memory": 4096,
            }

    def mlir_compiler_entry(pyfunc, func_args, targetoptions, override_argtypes):
        call = (tuple(func_args), tuple(override_argtypes), dict(targetoptions))
        compile_calls.append(call)
        if len(compile_calls) == 1:
            # Planner A promotes launch metadata, then asks for its scalar.
            targetoptions[_LAUNCH_CONFIG_TRACKER_OPTION].require()
            raise ForceLiteralArg({0})
        if len(compile_calls) == 2:
            # Planner B is reached only after Planner A's request is satisfied.
            raise ForceLiteralArg({1})
        assert all(isinstance(argtype, types.Literal) for argtype in override_argtypes)
        return CompilerResult(override_argtypes)

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)
    descriptor_mod._compile_arg_types.types = (types.int32, types.int32)
    descriptor_mod._compile_arg_types.values = (7, 11)
    descriptor_mod._compile_arg_types.available_launch_config = available_launch_config

    with pytest.warns(
        descriptor_mod.NumbaPerformanceWarning,
        match="Persistent disk cache is disabled for launch-config-specialized compiles",
    ):
        result = dispatcher._compile_impl([7, 11])

    normalized_launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": 0,
        "cluster": None,
    }
    launch_key = descriptor_mod._launch_config_key(normalized_launch_config)
    literal7 = types.literal(7)
    literal11 = types.literal(11)

    assert result == (b"literal-7-11", "kernel", False)
    assert [call[1] for call in compile_calls] == [
        (types.int32, types.int32),
        (literal7, types.int32),
        (literal7, literal11),
    ]
    assert "__launch_config__" not in compile_calls[0][2]
    assert compile_calls[0][2][_LAUNCH_CONFIG_TRACKER_OPTION].required is True
    assert all(
        call[2].get("__launch_config__") == normalized_launch_config for call in compile_calls[1:]
    )
    assert dispatcher._literal_arg_positions == frozenset({0, 1})
    assert dispatcher._launch_config_generation == 2
    compile_result = dispatcher._launch_config_overloads[((literal7, literal11), launch_key)]
    assert compile_result.metadata["required_dynamic_shared_memory"] == 4096


def test_literal_retry_rejects_flattened_values_and_no_progress(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(value):
        pass

    compile_calls = []

    def request_literal(*args, **kwargs):
        compile_calls.append(tuple(kwargs["override_argtypes"]))
        raise ForceLiteralArg({0})

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", request_literal)
    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    descriptor_mod._compile_arg_types.types = (types.UniTuple(types.int32, 2),)
    descriptor_mod._compile_arg_types.values = ((1, 2),)

    with pytest.raises(TypeError, match=r"flattened launch arguments.*issue #60"):
        dispatcher._compile_impl([1, 2])

    compile_calls.clear()
    dispatcher = descriptor_mod.MLIRDispatcher(kernel)
    descriptor_mod._compile_arg_types.types = (types.int32,)
    descriptor_mod._compile_arg_types.values = (7,)

    with pytest.raises(cuda_errors.CompilerError, match="Repeated literal typing request"):
        dispatcher._compile_impl([7])

    assert compile_calls == [(types.int32,), (types.literal(7),)]
    assert dispatcher._literal_arg_positions == frozenset({0})


def test_compile_impl_promotes_available_launch_config_in_current_attempt(monkeypatch):
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
        tracker = targetoptions.pop(_LAUNCH_CONFIG_TRACKER_OPTION)
        targetoptions["__launch_config__"] = tracker.require()
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
    assert len(compile_calls) == 1
    assert "__launch_config__" not in compile_calls[0]
    tracker = compile_calls[0][_LAUNCH_CONFIG_TRACKER_OPTION]
    assert tracker.required is True
    assert tracker.launch_config == normalized_launch_config
    assert dispatcher._requires_launch_config is True
    assert not dispatcher.overloads
    assert ((types.int32,), launch_key) in dispatcher._launch_config_overloads


def test_compile_impl_retries_only_after_promoted_config_generation_changes(monkeypatch):
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
            tracker = targetoptions.pop(_LAUNCH_CONFIG_TRACKER_OPTION)
            targetoptions["__launch_config__"] = tracker.require()
            with dispatcher._launch_config_lock:
                dispatcher._launch_config_generation += 1
            return CompilerResult(b"stale", stale_setup_callback)
        assert targetoptions["__launch_config__"] == launch_config
        return CompilerResult(b"accepted", accepted_setup_callback)

    monkeypatch.setattr(mlir_compiler, "mlir_compiler_entry", mlir_compiler_entry)
    descriptor_mod._compile_arg_types.types = (types.int32,)
    descriptor_mod._compile_arg_types.available_launch_config = launch_config

    with pytest.warns(
        descriptor_mod.NumbaPerformanceWarning,
        match="Persistent disk cache is disabled for launch-config-specialized compiles",
    ):
        assert dispatcher._compile_impl([1]) == (b"accepted", "kernel", False)

    assert len(compile_calls) == 2
    assert "__launch_config__" not in compile_calls[0]
    assert compile_calls[1]["__launch_config__"] == launch_config
    assert dispatcher._requires_launch_config is True
    assert stale_setup_callback not in dispatcher._module_setup_callbacks
    assert accepted_setup_callback in dispatcher._module_setup_callbacks


def test_compile_impl_reports_unavailable_launch_request(monkeypatch):
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


def test_compile_impl_preserves_invalid_launch_config_type_error(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(x):
        pass

    def request_launch_config(pyfunc, func_args, targetoptions, override_argtypes):
        targetoptions[_LAUNCH_CONFIG_TRACKER_OPTION].require()

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


def test_compile_impl_ignores_opaque_available_launch_config_without_demand(monkeypatch):
    from numba_cuda_mlir import mlir_compiler

    def kernel(x):
        pass

    compile_calls = []

    class CompilerResult:
        signature = cuda_typing.signature(types.none, types.int32)
        metadata = {"cubin": b"generic", "func_name": "kernel"}

    def compile_without_launch_demand(pyfunc, func_args, targetoptions, override_argtypes):
        compile_calls.append(dict(targetoptions))
        return CompilerResult()

    monkeypatch.setattr(
        mlir_compiler,
        "mlir_compiler_entry",
        compile_without_launch_demand,
    )
    descriptor_mod._compile_arg_types.types = (types.int32,)
    descriptor_mod._compile_arg_types.available_launch_config = {
        "grid": (1, 1, 1),
        "block": (32, 1, 1),
        "sharedmem": "dynamic",
        "cluster": None,
    }
    dispatcher = descriptor_mod.MLIRDispatcher(kernel)

    assert dispatcher._compile_impl([1]) == (b"generic", "kernel", False)
    assert len(compile_calls) == 1
    tracker = compile_calls[0][_LAUNCH_CONFIG_TRACKER_OPTION]
    assert tracker.required is False
    assert tracker._available_launch_config["sharedmem"] == "dynamic"
    assert "__launch_config__" not in compile_calls[0]
    assert dispatcher._requires_launch_config is False
    assert dispatcher.overloads
    assert not dispatcher._launch_config_overloads


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
