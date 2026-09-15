# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

from numba_cuda_mlir._threading import _LockedCounter
from numba_cuda_mlir.numba_cuda.core import event
from numba_cuda_mlir.numba_cuda.core.base import BaseContext as LoweringContext
from numba_cuda_mlir.numba_cuda.core.imputils import Registry as LoweringRegistry
from numba_cuda_mlir.numba_cuda.descriptor import CUDATarget
from numba_cuda_mlir.numba_cuda.typing.context import BaseContext as TypingContext
from numba_cuda_mlir.numba_cuda.typing.context import CallStack
from numba_cuda_mlir.numba_cuda.typing.templates import Registry as TypingRegistry
from numba_cuda_mlir.numba_cuda.typing.templates import RegistryLoader
from numba_cuda_mlir.numba_cuda.types import Type


def test_locked_counter_is_unique_across_threads():
    counter = _LockedCounter(3)
    workers = 8
    values_per_worker = 100
    start = threading.Barrier(workers)

    def count():
        start.wait()
        return [next(counter) for _ in range(values_per_worker)]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = [
            value for result in executor.map(lambda _: count(), range(workers)) for value in result
        ]

    assert sorted(values) == list(range(3, 3 + workers * values_per_worker))


def test_registry_loader_does_not_commit_a_partial_window():
    registry = TypingRegistry()

    class Template:
        pass

    registry.register(Template)
    loader = RegistryLoader(registry)
    registrations = loader.new_registrations("functions")

    assert next(registrations) is Template
    registrations.close()
    assert list(loader.new_registrations("functions")) == [Template]


def test_typing_registry_installation_is_atomic():
    initialized = threading.Event()
    release_initialization = threading.Event()
    second_finished = threading.Event()
    key = object()

    class Template:
        def __init__(self, context):
            initialized.set()
            assert release_initialization.wait(timeout=10)
            self.key = key

    registry = TypingRegistry()
    registry.register(Template)
    context = TypingContext()
    errors = []

    def install(finished=None):
        try:
            context.install_registry(registry)
        except BaseException as exc:
            errors.append(exc)
        finally:
            if finished is not None:
                finished.set()

    first = threading.Thread(target=install)
    first.start()
    assert initialized.wait(timeout=10)

    second = threading.Thread(target=install, args=(second_finished,))
    second.start()
    assert not second_finished.wait(timeout=0.1)

    release_initialization.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert len(context._functions[key]) == 1


def test_lowering_registry_installation_is_atomic():
    registry = LoweringRegistry()

    @registry.lower("test_op")
    def lower_test_op(context, builder, signature, args):
        pass

    context = object.__new__(LoweringContext)
    context._registry_lock = threading.RLock()
    context._registries = {}
    context._registry_versions = {}
    initialized = threading.Event()
    release_initialization = threading.Event()
    second_finished = threading.Event()
    installed = []
    errors = []

    def insert_func_defn(definitions):
        definitions = list(definitions)
        initialized.set()
        assert release_initialization.wait(timeout=10)
        installed.extend(definitions)

    context.insert_func_defn = insert_func_defn
    context._insert_getattr_defn = lambda definitions: list(definitions)
    context._insert_setattr_defn = lambda definitions: list(definitions)
    context._insert_cast_defn = lambda definitions: list(definitions)
    context._insert_get_constant_defn = lambda definitions: list(definitions)

    def install(finished=None):
        try:
            context.install_registry(registry)
        except BaseException as exc:
            errors.append(exc)
        finally:
            if finished is not None:
                finished.set()

    first = threading.Thread(target=install)
    first.start()
    assert initialized.wait(timeout=10)

    second = threading.Thread(target=install, args=(second_finished,))
    second.start()
    assert not second_finished.wait(timeout=0.1)

    release_initialization.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert installed == [(lower_test_op, "test_op", ())]


def test_cuda_target_initialization_waits_for_concurrent_initialization(monkeypatch):
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

    from numba_cuda_mlir import device_declarations

    initialized = threading.Event()
    release_initialization = threading.Event()
    second_finished = threading.Event()
    target = CUDATarget("test_cuda_target_initialization")
    typing_context = Context(initialized, release_initialization)
    target_context = Context()
    target._typingctx = typing_context
    target._targetctx = target_context
    errors = []

    monkeypatch.setattr(target, "_seed_target_registry", lambda: None)
    monkeypatch.setattr(device_declarations, "apply_device_declarations", lambda *args: None)

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


def test_cuda_target_initialization_waits_for_declaration_application(monkeypatch):
    class Context:
        def refresh(self):
            pass

        def install_registry(self, registry):
            pass

    from numba_cuda_mlir import device_declarations

    declarations_started = threading.Event()
    release_declarations = threading.Event()
    second_finished = threading.Event()
    target = CUDATarget("test_cuda_target_declaration_application")
    target._typingctx = Context()
    target._targetctx = Context()
    errors = []

    monkeypatch.setattr(target, "_seed_target_registry", lambda: None)

    def apply_declarations(*args):
        declarations_started.set()
        assert release_declarations.wait(timeout=10)

    monkeypatch.setattr(device_declarations, "apply_device_declarations", apply_declarations)

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
    assert declarations_started.wait(timeout=10)

    second = threading.Thread(target=ensure_initialized, args=(second_finished,))
    second.start()
    assert not second_finished.wait(timeout=0.1)

    release_declarations.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []


def test_call_stack_is_thread_local():
    callstack = CallStack()
    functions = [lambda: None, lambda: None]
    start = threading.Barrier(2)

    def inspect_stack(worker):
        func = functions[worker]
        other_func = functions[1 - worker]
        func_id = SimpleNamespace(func=func)
        with callstack.register("target", None, func_id, (worker,)):
            start.wait()
            frame = callstack.findfirst(func)
            return len(callstack), frame.args, callstack.findfirst(other_func)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(inspect_stack, range(2)))

    assert results == [(1, (0,), None), (1, (1,), None)]
    assert len(callstack) == 0


def test_event_broadcast_uses_listener_snapshot():
    kind = "test-shared-iterator-broadcast"
    notified = threading.Event()
    release_notification = threading.Event()

    class Listener(event.Listener):
        def on_start(self, evt):
            notified.set()
            assert release_notification.wait(timeout=10)

        def on_end(self, evt):
            pass

    listener = Listener()
    event.register(kind, listener)
    broadcaster = threading.Thread(
        target=event.broadcast,
        args=(event.Event(kind, event.EventStatus.START),),
    )
    broadcaster.start()
    assert notified.wait(timeout=10)

    event.unregister(kind, listener)
    release_notification.set()
    broadcaster.join(timeout=10)

    assert not broadcaster.is_alive()


def test_type_interning_is_atomic():
    class SharedType(Type):
        def __init__(self):
            super().__init__(name="shared-iterator-test-type")

    workers = 8
    start = threading.Barrier(workers)

    def make_type():
        start.wait()
        return SharedType()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        instances = list(executor.map(lambda _: make_type(), range(workers)))

    assert all(instance is instances[0] for instance in instances)
    assert all(instance._code == instances[0]._code for instance in instances)
