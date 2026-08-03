# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy, process-wide initialization for compiler extensions."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, cast


_ENTRY_POINT_GROUP = "numba_cuda_mlir_extensions"
_ENTRY_POINT_NAME = "init"
_UNINITIALIZED = object()

logger = logging.getLogger(__name__)
_initialization_lock = threading.RLock()
_initializing_thread: int | None = None
_initialization: object = _UNINITIALIZED


class ExtensionInitializationError(RuntimeError):
    """Report a compiler extension that could not be initialized."""


@dataclass(frozen=True)
class _FailedInitialization:
    error: ExtensionInitializationError


@dataclass(frozen=True)
class _RegistrationSnapshot:
    planners: tuple[type, ...]
    rewrites: dict[str, tuple[type, ...]]


def _snapshot_registrations() -> _RegistrationSnapshot:
    """Snapshot compiler-owned registries populated by extension imports."""

    from numba_cuda_mlir._whole_function_planners import _planner_registry
    from numba_cuda_mlir.numba_cuda.core.rewrites import rewrite_registry

    with _planner_registry._lock:
        planners = tuple(_planner_registry._planners)
    rewrites = {
        kind: tuple(rewrite_classes) for kind, rewrite_classes in rewrite_registry.rewrites.items()
    }
    return _RegistrationSnapshot(planners=planners, rewrites=rewrites)


def _restore_registrations(snapshot: _RegistrationSnapshot) -> None:
    """Roll compiler-owned registries back to *snapshot*."""

    from numba_cuda_mlir._whole_function_planners import _planner_registry
    from numba_cuda_mlir.numba_cuda.core.rewrites import rewrite_registry

    with _planner_registry._lock:
        _planner_registry._planners[:] = snapshot.planners
    rewrite_registry.rewrites.clear()
    for kind, rewrite_classes in snapshot.rewrites.items():
        rewrite_registry.rewrites[kind].extend(rewrite_classes)


def _entry_point_identity(entry_point: Any) -> tuple[str, str, str]:
    distribution = getattr(entry_point, "dist", None)
    distribution_name = getattr(distribution, "name", "") or ""
    module = getattr(entry_point, "module", "") or ""
    value = getattr(entry_point, "value", "") or module
    return str(distribution_name), str(module), str(value)


def _entry_point_sort_key(entry_point: Any) -> tuple[str, str, str]:
    return tuple(part.casefold() for part in _entry_point_identity(entry_point))


def _discover_entry_points() -> tuple[Any, ...]:
    entry_points = importlib_metadata.entry_points()
    if hasattr(entry_points, "select"):
        selected = entry_points.select(group=_ENTRY_POINT_GROUP, name=_ENTRY_POINT_NAME)
    else:
        selected = (
            entry_point
            for entry_point in entry_points.get(_ENTRY_POINT_GROUP, ())
            if entry_point.name == _ENTRY_POINT_NAME
        )
    return tuple(sorted(selected, key=_entry_point_sort_key))


def _format_entry_point(entry_point: Any) -> str:
    distribution, module, value = _entry_point_identity(entry_point)
    source = distribution or module or "unknown distribution"
    target = value or "unknown initializer"
    return f"{source!r} ({target})"


def _initialization_error(
    entry_point: Any | None,
    exc: BaseException,
    stage: str,
) -> ExtensionInitializationError:
    if entry_point is None:
        location = stage
    else:
        location = f"extension {_format_entry_point(entry_point)}"
    return ExtensionInitializationError(
        f"Numba-CUDA-MLIR {location} failed with {type(exc).__name__}: {exc}"
    )


def initialize_extensions() -> tuple[str, ...]:
    """Initialize installed compiler extensions exactly once.

    Discovery is deferred until the first compilation route calls this
    function. Initialization is serialized across threads. A recursive compile
    from an initializer is rejected, and an initialization failure rolls back
    compiler-owned planner and rewrite registrations before the same failure is
    cached for later callers.
    """

    global _initialization
    global _initializing_thread

    initialization = _initialization
    if isinstance(initialization, _FailedInitialization):
        raise initialization.error
    if initialization is not _UNINITIALIZED:
        return cast(tuple[str, ...], initialization)

    with _initialization_lock:
        if isinstance(_initialization, _FailedInitialization):
            raise _initialization.error
        if _initialization is not _UNINITIALIZED:
            return cast(tuple[str, ...], _initialization)

        current_thread = threading.get_ident()
        if _initializing_thread == current_thread:
            raise RuntimeError(
                "recursive Numba-CUDA-MLIR extension initialization attempted "
                "compilation before extension registration completed"
            )

        _initializing_thread = current_thread
        snapshot = None
        entry_point = None
        stage = "registration snapshot"
        try:
            try:
                snapshot = _snapshot_registrations()
                stage = "entry-point discovery"
                entry_points = _discover_entry_points()
                initialized = []
                for entry_point in entry_points:
                    logger.debug("Loading extension: %s", entry_point)
                    initializer = entry_point.load()
                    if not callable(initializer):
                        raise TypeError(
                            f"extension initializer {_format_entry_point(entry_point)} "
                            "is not callable"
                        )
                    initializer()
                    initialized.append(_format_entry_point(entry_point))
            except BaseException as exc:
                if snapshot is not None:
                    _restore_registrations(snapshot)
                if not isinstance(exc, Exception):
                    _initialization = _UNINITIALIZED
                    raise
                error = _initialization_error(entry_point, exc, stage)
                _initialization = _FailedInitialization(error)
                raise error from exc

            result = tuple(initialized)
            _initialization = result
            return result
        finally:
            _initializing_thread = None


__all__: list[str] = []
