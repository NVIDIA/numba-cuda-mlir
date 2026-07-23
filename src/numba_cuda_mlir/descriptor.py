# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import collections
from contextlib import contextmanager
from dataclasses import dataclass
import sys
import sysconfig
import os
import threading
from functools import cached_property

from numba_cuda_mlir.typing import unicode

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override
from numba_cuda_mlir.errors import UserFacingInternalCompilerError
from numba_cuda_mlir.logging import trace
import inspect

from numba_cuda_mlir import codegen
from numba_cuda_mlir.decorators import mlir_jit
from numba_cuda_mlir.numba_cuda.errors import normalize_kernel_dimensions
from numba_cuda_mlir.numba_cuda.core.errors import NumbaPerformanceWarning
from numba_cuda_mlir.numba_cuda.core import config as cuda_config
from numba_cuda_mlir.numba_cuda.cudadrv import driver as numba_cuda_driver
from importlib.util import find_spec
from numba_cuda_mlir.numba_cuda.core import errors, sigutils
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof
from numba_cuda_mlir.numba_cuda.cudadecl import registry as cuda_registry
from numba_cuda_mlir.numba_cuda import serialize, typing
from numba_cuda_mlir.numba_cuda.core.base import BaseContext
from numba_cuda_mlir.numba_cuda.core.callconv import MinimalCallConv
from numba_cuda_mlir.numba_cuda.cudadrv.devicearray import DeviceNDArrayBase
from numba_cuda_mlir.numba_cuda.np import numpy_support
from numba_cuda_mlir.numba_cuda.core.descriptors import TargetDescriptor
from numba_cuda_mlir.numba_cuda.core.compiler_lock import global_compiler_lock
from numba_cuda_mlir.numba_cuda.dispatcher import Dispatcher
from numba_cuda_mlir.numba_cuda.core.options import TargetOptions
from numba_cuda_mlir._whole_function_planners import (
    _REQUIRED_DYNAMIC_SHARED_MEMORY_KEY,
    _RequireLaunchConfig,
    _planner_registry,
)
from numba_cuda_mlir.numba_cuda.core.target_extension import (
    CPU,
    target_registry,
    dispatcher_registry,
    jit_registry,
)
from numba_cuda_mlir.type_defs.builtin_types import Namespace
from numba_cuda_mlir.caching import MLIRCache, NullCache
import logging
import warnings
import numpy as np
from numba_cuda_mlir.errors import ExtensionError
from numba_cuda_mlir import _cext
from numba_cuda_mlir._cext import LaunchConfiguration
import cProfile
import pstats

# Thread-local storage for passing original tuple arg types to _compile
_compile_arg_types = threading.local()
_PY_GIL_DISABLED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
_DISPATCHER_STATE_BOOTSTRAP_LOCK = threading.Lock()
_DISPATCHER_STATE_ATTRS = (
    "_launch_config_lock",
    "_launch_lock",
    "_launch_config_overloads",
    "_launch_config_dispatchers",
    "_launch_config_generation",
    "_requires_launch_config",
    "_launch_config_cache_misses",
    "_launch_config_cache_hits",
    "_launch_config_cache_notice_emitted",
    "_old_dispatchers",
    "_configure_cache",
    "_configure_cache_inflight",
)


def _new_dispatcher_rlock():
    return threading.RLock()


def _is_strided_memory_view(arg):
    """Check if arg is a cuda.core StridedMemoryView."""
    try:
        from cuda.core._memoryview import StridedMemoryView

        return isinstance(arg, StridedMemoryView)
    except ImportError:
        return False


def _strided_memory_view_to_device_array(smv):
    """Convert a StridedMemoryView to a DeviceNDArray for kernel dispatch."""
    from numba_cuda_mlir.numba_cuda.cudadrv import driver, devices
    from numba_cuda_mlir.numba_cuda.cudadrv.devicearray import DeviceNDArray

    shape = smv.shape
    dtype = np.dtype(smv.dtype)
    strides = smv.strides
    if strides is None:
        strides = tuple(dtype.itemsize * int(np.prod(shape[i + 1 :])) for i in range(len(shape)))

    size = driver.memory_size_from_info(shape, strides, dtype.itemsize)
    devptr = driver.get_devptr_for_active_ctx(smv.ptr)
    gpu_data = driver.MemoryPointer(devices.get_context(), devptr, size=size, owner=smv)

    return DeviceNDArray(
        shape=shape,
        strides=strides,
        dtype=dtype,
        gpu_data=gpu_data,
    )


class _LoadedModule:
    """Wrapper around a CUlibrary handle passed to module callbacks."""

    __slots__ = ("handle",)

    def __init__(self, handle):
        self.handle = handle


class _ModuleCallbackRef:
    """Prevent GC of ObjectCode and invoke teardown callbacks when this ref is collected."""

    __slots__ = ("_teardown_cbs", "_object_code")

    def __init__(self, teardown_cbs, object_code):
        self._teardown_cbs = list(teardown_cbs)
        self._object_code = object_code

    def __del__(self):
        for cb in self._teardown_cbs:
            _run_teardown_callback(cb, self._object_code)
        self._teardown_cbs = []
        self._object_code = None


def _run_teardown_callback(cb, object_code):
    try:
        cb(object_code)
    except Exception:
        logging.exception("teardown callback %r failed for %r", cb, object_code)


def _flatten_arg(arg, out):
    """Recursively flatten a tuple/list argument into *out*."""
    if isinstance(arg, (tuple, list)):
        for elem in arg:
            _flatten_arg(elem, out)
    else:
        out.append(arg)


def _cuda_array_interface(arg):
    return getattr(arg, "__cuda_array_interface__", None)


def _validate_cuda_array_interface(cai):
    version = cai.get("version")
    if version is not None and 1 <= version and cai.get("mask") is not None:
        raise NotImplementedError("Masked arrays are not supported")


def _sync_cuda_array_interface_stream(arg):
    if isinstance(arg, DeviceNDArrayBase):
        return

    cai = _cuda_array_interface(arg)
    if cai is None:
        return

    _validate_cuda_array_interface(cai)
    stream_ptr = cai.get("stream")
    if stream_ptr is None or not cuda_config.CUDA_ARRAY_INTERFACE_SYNC:
        return

    from numba_cuda_mlir.numba_cuda.cudadrv import devices

    devices.get_context().create_external_stream(stream_ptr).synchronize()


def _array_shape(arg):
    shape = getattr(arg, "shape", None)
    if shape is not None:
        return tuple(shape)

    cai = _cuda_array_interface(arg)
    if cai is not None:
        return tuple(cai["shape"])

    return None


def _array_dtype(arg, ty=None):
    dtype = getattr(arg, "dtype", None)
    if dtype is not None:
        return np.dtype(dtype)

    cai = _cuda_array_interface(arg)
    if cai is not None:
        return np.dtype(cai["typestr"])

    if isinstance(ty, types.Array):
        return numpy_support.as_dtype(ty.dtype)

    raise AttributeError(f"{type(arg).__name__!r} object has no dtype metadata")


def _array_strides(arg):
    strides = getattr(arg, "strides", None)
    if strides is not None:
        return tuple(strides)

    cai = _cuda_array_interface(arg)
    if cai is not None:
        strides = cai.get("strides")
        if strides is not None:
            return tuple(strides)

    return None


def _array_layout(arg, ty=None):
    """Return the Numba array layout key for array-like runtime arguments."""
    ndim = getattr(arg, "ndim", None)
    if ndim is None:
        shape = _array_shape(arg)
        if shape is None:
            if isinstance(ty, types.Array):
                return ty.layout
            shape = ()
        ndim = len(shape)
    if ndim == 0:
        return "C"

    strides = _array_strides(arg)
    if strides is None:
        return "C"

    shape = _array_shape(arg)
    itemsize = _array_dtype(arg, ty).itemsize

    c_stride = itemsize
    c_strides = []
    for extent in reversed(shape):
        c_strides.append(c_stride)
        c_stride *= extent
    c_strides = tuple(reversed(c_strides))
    if tuple(strides) == c_strides:
        return "C"

    f_stride = itemsize
    f_strides = []
    for extent in shape:
        f_strides.append(f_stride)
        f_stride *= extent
    if tuple(strides) == tuple(f_strides):
        return "F"

    return "A"


def _array_readonly(arg, ty=None):
    cai = _cuda_array_interface(arg)
    if cai is not None:
        _, readonly = cai["data"]
        return bool(readonly)

    flags = getattr(arg, "flags", None)
    if flags is not None:
        writeable = getattr(flags, "writeable", None)
        if writeable is None:
            writeable = flags["WRITEABLE"]
        return not writeable

    if isinstance(ty, types.Array):
        return not ty.mutable

    return False


def _array_cache_key(arg, ty):
    """Return metadata that determines whether a cached array type is reusable."""
    dtype = _array_dtype(arg, ty)
    ndim = getattr(arg, "ndim", None)
    if ndim is None:
        shape = _array_shape(arg)
        if shape is None:
            ndim = ty.ndim
        else:
            ndim = len(shape)
    layout = _array_layout(arg, ty)
    readonly = _array_readonly(arg, ty)
    return (dtype, ndim, layout, readonly)


def _raise_as_cuda_error(e):
    """Convert RuntimeError from C++ launcher to cuda.core CUDAError."""
    try:
        from cuda.core._utils.cuda_utils import CUDAError
    except ImportError:
        from cuda.core.experimental._utils.cuda_utils import CUDAError
    msg = str(e)
    if "Failed to launch CUDA kernel:" in msg:
        raise CUDAError(msg) from None
    raise


def _ensure_numba_cuda_context():
    from numba_cuda_mlir.numba_cuda.cudadrv import devices

    return devices.get_context()


_MISSING = object()
_CONFIGURE_CACHE_SIZE = 128
_CONFIGURE_CACHE_STALE_RETRY_LIMIT = 8
# Keep this larger than configure()'s cache so any marshaller retained by
# configure() also has room to keep its native dispatcher entry live.
_LAUNCH_CONFIG_CACHE_SIZE = _CONFIGURE_CACHE_SIZE * 2


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        warnings.warn(
            f"Ignoring invalid {name}={value!r}; expected an integer.",
            RuntimeWarning,
            stacklevel=2,
        )
        return default


_OLD_DISPATCHER_RETAIN_LIMIT = _env_int("NUMBA_CUDA_MLIR_OLD_DISPATCHER_RETAIN_LIMIT", 1024)


@dataclass(frozen=True)
class LaunchConfigInspectableKey:
    argtypes: tuple
    launch_config_key: tuple


def _is_launch_config_key_tuple(value):
    return (
        isinstance(value, tuple)
        and len(value) == 4
        and all(isinstance(item, tuple) and len(item) == 2 for item in value)
        and value[0][0] == "grid"
        and value[1][0] == "block"
        and value[2][0] == "sharedmem"
        and value[3][0] == "cluster"
    )


def _is_launch_config_dict(value):
    return isinstance(value, dict) and "grid" in value and "block" in value


def _launch_config_key(launch_config):
    """Return the specialization key for a configure-produced launch config."""
    if launch_config is None:
        return None
    block = launch_config.get("block")
    if block is None:
        raise ValueError("launch_config must contain a 'block' entry")
    if not isinstance(block, tuple):
        raise TypeError("launch_config 'block' must be a normalized tuple")
    grid = launch_config.get("grid")
    if grid is None:
        raise ValueError("launch_config must contain a 'grid' entry")
    if not isinstance(grid, tuple):
        raise TypeError("launch_config 'grid' must be a normalized tuple")
    cluster = launch_config.get("cluster")
    if cluster is not None and not isinstance(cluster, tuple):
        raise TypeError("launch_config 'cluster' must be a normalized tuple or None")
    sharedmem = launch_config.get("sharedmem", 0)
    if sharedmem is None:
        sharedmem = 0
    try:
        sharedmem = int(sharedmem)
    except (TypeError, ValueError):
        raise TypeError("launch_config 'sharedmem' must be integer-convertible") from None
    return (
        ("grid", grid),
        ("block", block),
        ("sharedmem", sharedmem),
        ("cluster", cluster),
    )


def _launch_config_dict_from_key(launch_config_key):
    if not _is_launch_config_key_tuple(launch_config_key):
        raise TypeError("launch_config_key must be a normalized launch-config key")
    return {name: value for name, value in launch_config_key}


def _normalize_available_launch_config(launch_config):
    launch_config_key = _launch_config_key(launch_config)
    if launch_config_key is None:
        return None
    return _launch_config_dict_from_key(launch_config_key)


def _normalize_launch_sharedmem(sharedmem):
    if sharedmem is None:
        return 0
    try:
        return int(sharedmem)
    except (TypeError, ValueError):
        raise TypeError("launch_config 'sharedmem' must be integer-convertible") from None


def _validate_configure_cache_key(cache_key):
    try:
        hash(cache_key)
    except TypeError:
        raise TypeError("kernel launch configuration values must be hashable") from None


def _extensions_use_launch_config(extensions):
    return any(getattr(ext, "uses_launch_config", False) for ext in extensions)


class _ObjectIdentityKey:
    __slots__ = ("obj",)

    def __init__(self, obj):
        # Keep a strong reference so cached identity keys cannot collide with a
        # newly allocated extension after the original extension is removed.
        self.obj = obj

    def __hash__(self):
        return id(self.obj)

    def __eq__(self, other):
        return isinstance(other, _ObjectIdentityKey) and self.obj is other.obj


class _ArgMarshaller:
    """
    Handles copying numpy arrays to the device and back to the host
    before and after the kernel is launched.
    NOTE: This adds overhead to the kernel launch!! This is not good!
    But numba-cuda supports it, so we must, for now.
    """

    def __init__(
        self,
        launcher,
        extensions=None,
        dispatcher=None,
        launch_config=None,
        available_launch_config=None,
        kernel_dispatcher=None,
        launch_config_generation=None,
        stream_ref=None,
        launch_stream=None,
    ):
        self._launcher = launcher
        self._callbacks = []
        # Snapshot extensions for this configured launcher; configure() cache keys
        # include extension identities so replacing the extension list builds a
        # distinct marshaller.
        self._has_extension_snapshot = extensions is not None
        self._extensions = list(extensions) if extensions is not None else []
        self._dispatcher = dispatcher
        self._launch_config = launch_config
        self._available_launch_config = (
            launch_config if available_launch_config is None else available_launch_config
        )
        self._kernel_dispatcher = kernel_dispatcher
        self._launch_config_generation = launch_config_generation
        self._stream_ref = stream_ref
        self._launch_stream = launch_stream
        self._kernel_dispatcher_registered = False
        self._kernel_dispatcher_registration_lock = threading.Lock()
        self._adjusted_launcher_key = None
        self._adjusted_launcher = None
        self._sig_cache = {}  # {type_key: (argtypes, fast_ok)}
        self._array_sig_cache = {}  # {type_key: [(argtypes, ((idx, array_key), ...))]}

    def _maybe_copy_to_device_item(self, arg):
        if isinstance(arg, DeviceNDArrayBase):
            return arg

        from numba_cuda_mlir import cuda

        match arg:
            case tuple() | list():
                # Recursively process tuple contents but preserve structure
                # For namedtuples, we need to unpack the processed elements
                processed = [self._maybe_copy_to_device_item(elem) for elem in arg]
                if hasattr(arg, "_fields"):  # namedtuple
                    return type(arg)(*processed)
                return type(arg)(processed)
            case np.datetime64() | np.timedelta64():
                return arg
            case np.integer():
                # Keep as-is; C++ launcher handles np.integer directly.
                return arg
            case np.float16():
                # Keep as-is; C++ launcher handles f16 conversion
                return arg
            case np.float32():
                # Keep as-is; C++ launcher handles f32 directly
                return arg
            case np.floating():
                # float64 and others -> Python float (64-bit)
                return float(arg)
            case np.complexfloating():
                # Keep numpy complex types as-is to preserve precision
                # The C++ launcher will extract with the correct bit width
                return arg
            case np.bool_():
                return bool(arg)
            case _ if hasattr(arg, "value") and hasattr(
                arg, "to_device"
            ):  # ArgHint type (cuda.In / cuda.Out / cuda.InOut)
                from numba_cuda_mlir.numba_cuda.args import In

                host_val = arg.value
                device_arg = cuda.to_device(host_val)
                if not isinstance(arg, In):

                    def copy_back(host_arg=host_val, dev_arg=device_arg):
                        host_array = dev_arg.copy_to_host()
                        np.copyto(host_arg, host_array)

                    self._callbacks.append(copy_back)
                return device_arg
            case _ if hasattr(arg, "__cuda_array_interface__") and not isinstance(arg, np.ndarray):
                # The C++ launcher can consume CUDA Array Interface objects
                # directly. Keep the original object so repeated CuPy/foreign
                # array launches do not rebuild DeviceNDArray wrappers.
                _sync_cuda_array_interface_stream(arg)
                return arg
            case np.ndarray() if hasattr(arg, "__cuda_array_interface__"):
                return arg
            case np.ndarray():
                trace("Copying numpy array: arg=%s", arg)
                if (
                    cuda_config.CUDA_WARN_ON_IMPLICIT_COPY
                    and not cuda_config.DISABLE_PERFORMANCE_WARNINGS
                ):
                    warnings.warn(
                        "Host array used in CUDA kernel will incur copy overhead to/from device",
                        NumbaPerformanceWarning,
                        stacklevel=4,
                    )
                device_arg = cuda.to_device(arg)

                def copy_back(host_arg=arg, dev_arg=device_arg):
                    host_array = dev_arg.copy_to_host()
                    np.copyto(host_arg, host_array)

                self._callbacks.append(copy_back)
                return device_arg
            case _ if _is_strided_memory_view(arg):
                return _strided_memory_view_to_device_array(arg)
            case _:
                return arg

    def _coerce_to_overload(self, device_args, argtypes):
        """Coerce scalar arguments to match a pre-compiled overload signature."""
        disp = self._dispatcher
        if disp is None or disp._can_compile:
            return device_args, argtypes

        extensions = (
            self._extensions if self._has_extension_snapshot else getattr(disp, "extensions", ())
        )
        uses_launch_config = _extensions_use_launch_config(extensions) or getattr(
            disp, "_requires_launch_config", False
        )
        active_launch_config = self._available_launch_config if uses_launch_config else None
        active_launch_config_key = _launch_config_key(active_launch_config)
        if active_launch_config_key is None:
            overload_argtypes = tuple(disp.overloads)
            if not overload_argtypes:
                return device_args, argtypes
            for sig_args in overload_argtypes:
                if len(sig_args) == len(argtypes) and all(
                    disp._can_reuse_overload(s, r) for s, r in zip(sig_args, argtypes)
                ):
                    return device_args, argtypes
            best, has_match = disp._resolve_overload(argtypes, allow_unsafe=True)
        else:
            overload_argtypes = disp._launch_config_overload_argtypes(active_launch_config_key)
            if not overload_argtypes:
                return device_args, argtypes
            for sig_args in overload_argtypes:
                if len(sig_args) == len(argtypes) and all(
                    disp._can_reuse_overload(s, r) for s, r in zip(sig_args, argtypes)
                ):
                    return device_args, argtypes
            best, has_match = disp._resolve_overload(
                argtypes,
                allow_unsafe=True,
                candidate_sig_args=overload_argtypes,
            )

        if not has_match or len(best) != 1:
            return device_args, argtypes

        sig_args = best[0]
        coerced_args = []
        coerced_types = []
        for arg, sig_type, runtime_type in zip(device_args, sig_args, argtypes):
            if sig_type == runtime_type:
                coerced_args.append(arg)
                coerced_types.append(runtime_type)
            elif isinstance(sig_type, types.Float) and isinstance(
                runtime_type, (types.Float, types.Integer)
            ):
                if sig_type == types.float32:
                    coerced_args.append(np.float32(arg))
                    coerced_types.append(types.float32)
                elif sig_type == types.float64:
                    coerced_args.append(float(arg))
                    coerced_types.append(types.float64)
                else:
                    coerced_args.append(arg)
                    coerced_types.append(runtime_type)
            else:
                coerced_args.append(arg)
                coerced_types.append(runtime_type)
        return coerced_args, coerced_types

    def _launch(self, argtypes, launch_args):
        """Set compile arg types and invoke the C++ launcher with error handling."""
        # Preserve existing thread-local state for nested launches triggered by
        # compile-time helpers.
        previous_types = getattr(_compile_arg_types, "types", _MISSING)
        previous_launch_config = getattr(_compile_arg_types, "launch_config", _MISSING)
        previous_available_launch_config = getattr(
            _compile_arg_types, "available_launch_config", _MISSING
        )
        previous_extensions = getattr(_compile_arg_types, "extensions", _MISSING)
        try:
            _compile_arg_types.types = argtypes
            if self._available_launch_config is not None:
                _compile_arg_types.available_launch_config = self._available_launch_config
            elif hasattr(_compile_arg_types, "available_launch_config"):
                delattr(_compile_arg_types, "available_launch_config")
            if self._has_extension_snapshot:
                _compile_arg_types.extensions = self._extensions
            elif self._dispatcher is not None and hasattr(self._dispatcher, "extensions"):
                _compile_arg_types.extensions = self._dispatcher.extensions
            elif hasattr(_compile_arg_types, "extensions"):
                delattr(_compile_arg_types, "extensions")

            active_launch_config = self._launch_config
            active_kernel_dispatcher = self._kernel_dispatcher
            active_launch_config_generation = self._launch_config_generation
            launcher = self._launcher
            dispatcher = self._dispatcher
            snapshot_launch_config_is_stale = _extensions_use_launch_config(
                self._extensions
                if self._has_extension_snapshot
                else getattr(dispatcher, "extensions", ())
            ) and self._launch_config_generation != getattr(
                dispatcher,
                "_launch_config_generation",
                self._launch_config_generation,
            )
            if dispatcher is not None and (
                _planner_registry.has_planners
                or getattr(dispatcher, "_requires_launch_config", False)
                or snapshot_launch_config_is_stale
            ):
                (
                    active_kernel_dispatcher,
                    active_launch_config_generation,
                    active_launch_config,
                    required_dynamic_shared_memory,
                ) = dispatcher._prepare_for_launch(
                    tuple(launch_args),
                    tuple(argtypes),
                    self._available_launch_config,
                    self._launch_config,
                    self._kernel_dispatcher,
                    self._launch_config_generation,
                )
                if (
                    active_kernel_dispatcher is not self._kernel_dispatcher
                    or active_launch_config != self._launch_config
                ):
                    # A generic compile after recompile() replaces the native
                    # dispatcher without activating launch metadata. Rebuild
                    # the launcher from the configure-time geometry while
                    # keeping launch_config absent from compiler TLS.
                    launcher_config = (
                        active_launch_config
                        if active_launch_config is not None
                        else self._available_launch_config
                    )
                    launcher = LaunchConfiguration(
                        active_kernel_dispatcher,
                        launcher_config["grid"],
                        launcher_config["block"],
                        self._launch_stream,
                        launcher_config["sharedmem"],
                        launcher_config.get("cluster"),
                    )
                    with self._kernel_dispatcher_registration_lock:
                        self._launcher = launcher
                        self._launch_config = active_launch_config
                        self._kernel_dispatcher = active_kernel_dispatcher
                        self._launch_config_generation = active_launch_config_generation
                        self._kernel_dispatcher_registered = False

                launcher_config = (
                    active_launch_config
                    if active_launch_config is not None
                    else self._available_launch_config
                )
                configured_sharedmem = _normalize_launch_sharedmem(launcher_config["sharedmem"])
                effective_sharedmem = max(configured_sharedmem, required_dynamic_shared_memory)
                if effective_sharedmem > configured_sharedmem:
                    adjusted_launcher_key = (
                        id(active_kernel_dispatcher),
                        active_launch_config_generation,
                        effective_sharedmem,
                    )
                    if adjusted_launcher_key != self._adjusted_launcher_key:
                        self._adjusted_launcher = LaunchConfiguration(
                            active_kernel_dispatcher,
                            launcher_config["grid"],
                            launcher_config["block"],
                            self._launch_stream,
                            effective_sharedmem,
                            launcher_config.get("cluster"),
                        )
                        self._adjusted_launcher_key = adjusted_launcher_key
                    launcher = self._adjusted_launcher

            if active_launch_config is not None:
                _compile_arg_types.launch_config = active_launch_config
            elif hasattr(_compile_arg_types, "launch_config"):
                delattr(_compile_arg_types, "launch_config")
            if (
                dispatcher is not None
                and active_launch_config is not None
                and active_kernel_dispatcher is not None
            ):
                with self._kernel_dispatcher_registration_lock:
                    if self._kernel_dispatcher_registered and hasattr(
                        dispatcher, "_is_kernel_dispatcher_registered"
                    ):
                        self._kernel_dispatcher_registered = (
                            dispatcher._is_kernel_dispatcher_registered(
                                active_launch_config,
                                active_kernel_dispatcher,
                                active_launch_config_generation,
                            )
                        )
                    if not self._kernel_dispatcher_registered:
                        self._kernel_dispatcher_registered = dispatcher._remember_kernel_dispatcher(
                            active_launch_config,
                            active_kernel_dispatcher,
                            active_launch_config_generation,
                            replace_existing=False,
                        )
            return launcher(*launch_args)
        except UserFacingInternalCompilerError as e:
            if os.environ.get("NUMBA_CUDA_MLIR_ICE_FULL_TB", "0").strip() == "1":
                raise
            raise e.with_traceback(None) from None
        except RuntimeError as e:
            _raise_as_cuda_error(e)
        finally:
            if previous_types is _MISSING:
                if hasattr(_compile_arg_types, "types"):
                    delattr(_compile_arg_types, "types")
            else:
                _compile_arg_types.types = previous_types
            if previous_launch_config is _MISSING:
                if hasattr(_compile_arg_types, "launch_config"):
                    delattr(_compile_arg_types, "launch_config")
            else:
                _compile_arg_types.launch_config = previous_launch_config
            if previous_available_launch_config is _MISSING:
                if hasattr(_compile_arg_types, "available_launch_config"):
                    delattr(_compile_arg_types, "available_launch_config")
            else:
                _compile_arg_types.available_launch_config = previous_available_launch_config
            if previous_extensions is _MISSING:
                if hasattr(_compile_arg_types, "extensions"):
                    delattr(_compile_arg_types, "extensions")
            else:
                _compile_arg_types.extensions = previous_extensions

    def __call__(self, *args):
        launch_lock = None
        if self._dispatcher is not None:
            launch_lock = getattr(self._dispatcher, "_launch_lock", _MISSING)
            if launch_lock is _MISSING:
                ensure_dispatcher_state = getattr(
                    self._dispatcher, "_ensure_dispatcher_state", None
                )
                if ensure_dispatcher_state is not None:
                    ensure_dispatcher_state()
                    launch_lock = getattr(self._dispatcher, "_launch_lock", None)
                else:
                    launch_lock = None
        if launch_lock is None:
            return self._call_impl(*args)
        # Free-threaded launches share caches, callbacks, and native dispatcher
        # state across this marshaller path.
        with launch_lock:
            return self._call_impl(*args)

    def _call_impl(self, *args):
        nargs = len(args)
        has_ext = bool(self._extensions)

        # Fast caches are only valid when no extension can transform values or
        # register per-launch callbacks through prepare_args().
        type_key = None
        if not has_ext:
            type_key = tuple(type(a) for a in args)
            cached = self._sig_cache.get(type_key)
            if cached is not None:
                cached_argtypes, fast_ok = cached
                if fast_ok:
                    return self._launch(cached_argtypes, args)

                # Array values keep the same Python class across many dtypes.
                # Reuse cached argtypes only when the Numba array type is stable.
                array_entries = self._array_sig_cache.get(type_key)
                if array_entries is not None:
                    for cached_argtypes, array_checks in array_entries:
                        ok = True
                        for idx, array_key in array_checks:
                            a = args[idx]
                            if _array_cache_key(a, cached_argtypes[idx]) != array_key:
                                ok = False
                                break
                        if ok:
                            for idx, _ in array_checks:
                                _sync_cuda_array_interface_stream(args[idx])
                            return self._launch(cached_argtypes, args)

        reversed_ext = self._extensions[::-1] if has_ext else None
        copy_item = self._maybe_copy_to_device_item
        callbacks = self._callbacks

        device_args = [None] * nargs
        argtypes = [None] * nargs
        all_pass_through = True

        for i in range(nargs):
            val = args[i]

            try:
                ty = typeof(val)
            except (ValueError, TypeError):
                ty = None

            if has_ext:
                if ty is None:
                    raise ExtensionError(
                        f"Could not get type of argument: {val}. "
                        "Please register a typeof_impl for this type."
                    )
                for ext in reversed_ext:
                    ty, val = ext.prepare_args(ty, val, stream=None, retr=callbacks)
                if val is not args[i]:
                    all_pass_through = False

            dev = copy_item(val)
            if dev is not val:
                all_pass_through = False
                try:
                    ty = typeof(dev)
                except (ValueError, TypeError):
                    ty = None

            device_args[i] = dev
            argtypes[i] = ty

        coerced_args, coerced_types = self._coerce_to_overload(device_args, argtypes)
        if coerced_args is not device_args:
            all_pass_through = False

        # _compile_arg_types carries top-level Numba types for compilation;
        # the C++ launcher receives flattened tuple/list leaves for the ABI.
        flat_args = []
        for arg in coerced_args:
            _flatten_arg(arg, flat_args)
        result = self._launch(coerced_types, flat_args)

        for callback in callbacks:
            callback()
        callbacks.clear()  # Clear callbacks to prevent accumulation.

        if not has_ext:
            has_arrays = any(isinstance(ct, types.Array) for ct in coerced_types)
            if all_pass_through and len(flat_args) == nargs and not has_arrays:
                self._sig_cache[type_key] = (list(coerced_types), True)
            else:
                if type_key not in self._sig_cache:
                    self._sig_cache[type_key] = (list(coerced_types), False)

                # Only cache already-device array launches. Host arrays and
                # wrappers need per-launch copy/copyback setup.
                if has_arrays and all_pass_through:
                    array_checks = []
                    for i, ct in enumerate(coerced_types):
                        if isinstance(ct, types.Array):
                            array_checks.append((i, _array_cache_key(args[i], ct)))
                    array_checks = tuple(array_checks)
                    entries = self._array_sig_cache.setdefault(type_key, [])
                    if not any(checks == array_checks for _, checks in entries):
                        entries.append((list(coerced_types), array_checks))
                        del entries[:-8]

        return result


class _ConfigureProxy:
    __slots__ = ("_dispatcher",)

    def __init__(self, dispatcher):
        self._dispatcher = dispatcher

    def __call__(self, *args, **kwargs):
        return type(self._dispatcher).configure(self._dispatcher, *args, **kwargs)

    def cache_clear(self):
        self._dispatcher._clear_configure_cache()


class _ForAll:
    """Deferred forall launcher to compute optimal block size at call time."""

    def __init__(self, dispatcher, ntasks, tpb, stream, sharedmem):
        self.dispatcher = dispatcher
        self.ntasks = ntasks
        self.thread_per_block = tpb
        self.stream = stream
        self.sharedmem = sharedmem

    def __call__(self, *args):
        if self.ntasks == 0:
            return

        blockdim = self._compute_thread_per_block()
        griddim = (self.ntasks + blockdim - 1) // blockdim
        return self.dispatcher[griddim, blockdim, self.stream, self.sharedmem](*args)

    def _compute_thread_per_block(self):
        if self.thread_per_block != 0:
            return self.thread_per_block
        try:
            from cuda.bindings import driver

            compile_results = list(self.dispatcher.overloads.values())
            if not compile_results and hasattr(self.dispatcher, "_launch_config_compile_results"):
                compile_results = self.dispatcher._launch_config_compile_results()
            cres = compile_results[0]
            cufunc = cres._codelibrary.get_cufunc()
            grid_size, block_size = driver.cuOccupancyMaxPotentialBlockSize(
                cufunc._handle, None, self.sharedmem, 1024
            )
            return int(block_size)
        except Exception:
            return 128  # default block size


def get_constant_args(py_func):
    # TODO: get actual get_constant_args from py_func,
    # omit args not present in gpu module
    num_args = len(inspect.signature(py_func).parameters)
    return tuple([False] * num_args)


class MLIRTypingContext(typing.BaseContext):
    def get_getattr(self, typ, attr):
        return super().get_getattr(typ, attr)

    def resolve_getattr(self, typ, attr):
        trace("typ=%s, attr=%s", typ, attr)
        return super().resolve_getattr(typ, attr)

    def load_additional_registries(self):
        from numba_cuda_mlir.typing.numpy import registry as npydecl_registry
        from numba_cuda_mlir.typing.libdevice import registry as libdevice_registry
        from numba_cuda_mlir.typing.externals import registry as externals_registry
        from numba_cuda_mlir.typing.ctypes import registry as ctypes_registry
        from numba_cuda_mlir.typing.cuda import (
            registry as numba_cuda_mlir_cuda_registry,
        )
        from numba_cuda_mlir.typing.builtin import (
            registry as numba_cuda_mlir_builtins_registry,
        )
        from numba_cuda_mlir.typing.struct import registry as struct_registry
        from numba_cuda_mlir.typing.cmath import registry as cmath_registry
        from numba_cuda_mlir.typing.math import registry as math_registry
        from numba_cuda_mlir.typing.half_precision import (
            registry as half_precision_registry,
            register_bf16_globals,
        )
        from numba_cuda_mlir.typing.exotic_float import register_fp8_globals
        from numba_cuda_mlir.typing.exotic_float import (
            registry as exotic_float_typing_registry,
        )
        from numba_cuda_mlir.typing.vector import registry as vector_registry
        from numba_cuda_mlir.typing.cuda_vector_types import (
            registry as cuda_vector_types_registry,
        )
        from numba_cuda_mlir.extending import (
            typing_registry as extending_typing_registry,
        )
        import numba_cuda_mlir.lowering.unicode  # noqa: F401 - registers string overloads
        from numba_cuda_mlir.numba_cuda.typing import enumdecl, cffi_utils
        from numba_cuda_mlir.numba_cuda.typing.templates import builtin_registry
        from numba_cuda_mlir.numba_cuda.target import load_cuda_target_registration_modules

        from numba_cuda_mlir.typing.unicode import registry as unicode_registry

        load_cuda_target_registration_modules()
        register_bf16_globals()
        register_fp8_globals()

        # Install numba_cuda_mlir registries first to give them priority
        self.install_registry(ctypes_registry)
        self.install_registry(libdevice_registry)
        self.install_registry(npydecl_registry)
        self.install_registry(numba_cuda_mlir_cuda_registry)
        self.install_registry(externals_registry)
        self.install_registry(numba_cuda_mlir_builtins_registry)
        self.install_registry(struct_registry)
        self.install_registry(cmath_registry)
        self.install_registry(math_registry)
        self.install_registry(half_precision_registry)
        self.install_registry(exotic_float_typing_registry)
        self.install_registry(vector_registry)
        self.install_registry(cuda_vector_types_registry)
        self.install_registry(unicode_registry)

        # Install numba-cuda registries after numba_cuda_mlir ones
        self.install_registry(cuda_registry)

        # Install numba-cuda's bf16 typing registry (includes operators)
        from numba_cuda_mlir.numba_cuda._internal.cuda_bf16 import (
            typing_registry as bf16_typing_registry,
        )

        self.install_registry(bf16_typing_registry)

        self.install_registry(enumdecl.registry)
        self.install_registry(cffi_utils.registry)
        self.install_registry(extending_typing_registry)
        if find_spec("torch") is not None:
            from numba_cuda_mlir.type_defs.torch_types import registry as torch_registry

            self.install_registry(torch_registry)

        # Install builtin_registry last: imports above (bf16, extending, etc.)
        # register @overload templates into it at import time.
        self.install_registry(builtin_registry)

        if find_spec("cupy") is not None:
            import numba_cuda_mlir.type_defs.cupy_types  # noqa: F401

    _conflicts_filtered = False

    def refresh(self):
        super().refresh()
        if not self._conflicts_filtered:
            self._filter_conflicting_overload_methods()
            self._conflicts_filtered = True

    def _filter_conflicting_overload_methods(self):
        """Remove upstream overload_method templates that conflict with
        numba_cuda_mlir's own Array method typing (min, max, etc.).  Upstream
        numba-cuda registers @overload_method(types.Array, "min") / "max"
        whose bodies use numpy_take which cannot handle 0-d arrays under
        the numba_cuda_mlir pipeline."""
        from numba_cuda_mlir.numba_cuda import types as nb_types

        numba_cuda_mlir_methods = {"min", "max", "sum", "prod", "mean", "std", "var"}
        arr_templates = self._attributes.get(nb_types.Array, [])
        self._attributes[nb_types.Array] = [
            t
            for t in arr_templates
            if not (
                hasattr(type(t), "_attr")
                and getattr(type(t), "_attr", None) in numba_cuda_mlir_methods
                and getattr(type(t), "__module__", "").startswith("numba_cuda_mlir.numba_cuda")
            )
        ]

    def resolve_value_type(self, val):
        # treat other dispatcher object as another njit_mlir function
        if isinstance(val, Dispatcher) and not isinstance(val, MLIRDispatcher):
            try:
                # use cached njit_mlir function
                val = val.__njit_mlir_dispatcher
            except AttributeError:
                if not val._can_compile:
                    raise ValueError(
                        "using cpu function in njit_mlir code but its compilation is disabled"
                    )
                targetoptions = val.targetoptions.copy()
                disp = MLIRDispatcher(val.py_func, targetoptions)
                # cache the device function for future use and to avoid
                # duplicated copy of the same function.
                val.__njit_mlir_dispatcher = disp
                val = disp

        # Use numba-cuda's typeof first, fall back to parent logic for the rest.
        from numba_cuda_mlir.numba_cuda.typing.typeof import typeof as cuda_typeof

        try:
            return cuda_typeof(val)
        except ValueError:
            return super().resolve_value_type(val)


class MLIRCallConv(MinimalCallConv):
    """Use simple default call convention for now"""


# The MLIRTargetContext allows us to use the data models that we registered.
class MLIRTargetContext(BaseContext):
    strict_alignment = True

    def __init__(self, typingctx, target="numba_cuda_mlir"):
        super().__init__(typingctx, target)
        from numba_cuda_mlir.models import mlir_data_manager

        self.data_model_manager = mlir_data_manager

    def _is_nonconst_module_attr(self, typ, attr):
        """
        Check if a module attribute requires runtime lowering instead of constant folding.

        This checks by module name patterns rather than module object identity,
        since the redirector system can create different module objects for
        the same logical module (e.g., numba.cuda -> numba_cuda.numba.cuda).
        """
        if not isinstance(typ, types.Module):
            return False

        # Attributes that need NVVM intrinsics at runtime
        nonconst_attrs = ("warpsize", "laneid")
        if attr not in nonconst_attrs:
            return False

        # Module name patterns that correspond to cuda modules
        # The pymod might be redirected, so check various patterns
        pymod = typ.pymod
        mod_name = getattr(pymod, "__name__", "")

        cuda_module_patterns = (
            "numba_cuda_mlir.cuda",
            "numba_cuda_mlir.numba_cuda",
            "numba.cuda",
            "numba_cuda.numba.cuda",
        )

        return any(
            mod_name == pattern or mod_name.startswith(pattern + ".")
            for pattern in cuda_module_patterns
        )

    def init(self):
        self._internal_codegen = codegen.JITMLIRCodegen("numba.mlir.jit")
        self._target_data = None

    @cached_property
    def call_conv(self):
        # "Placeholder", required for Numba to function correctly
        return MLIRCallConv(self)

    def codegen(self):
        # "Placeholder", required for Numba to function correctly
        return self._internal_codegen

    def load_additional_registries(self):
        from numba_cuda_mlir.install_registry import setup_lowering_patches

        setup_lowering_patches()

        from numba_cuda_mlir.models import register_fp8_models

        register_fp8_models()

        # Import individual lowering registries
        from numba_cuda_mlir.lowering.builtins import (
            registry as builtins_lowering_registry,
        )
        from numba_cuda_mlir.lowering.math import registry as math_lowering_registry
        from numba_cuda_mlir.lowering.cmath import registry as cmath_lowering_registry
        from numba_cuda_mlir.lowering.numpy import registry as numpy_lowering_registry
        from numba_cuda_mlir.lowering.libdevice import (
            registry as libdevice_lowering_registry,
        )
        from numba_cuda_mlir.lowering.cuda import registry as cuda_lowering_registry
        from numba_cuda_mlir.lowering.print import registry as print_lowering_registry
        from numba_cuda_mlir.lowering.ctypes import registry as ctypes_lowering_registry
        from numba_cuda_mlir.lowering.struct import registry as struct_lowering_registry
        from numba_cuda_mlir.lowering.union import registry as union_lowering_registry
        from numba_cuda_mlir.lowering.half_precision import (
            registry as half_precision_lowering_registry,
        )
        from numba_cuda_mlir.lowering.exotic_float import (
            registry as exotic_float_lowering_registry,
        )
        from numba_cuda_mlir.lowering.vector import registry as vector_lowering_registry
        from numba_cuda_mlir.lowering.cuda_vector_types import (
            registry as cuda_vector_types_lowering_registry,
        )
        from numba_cuda_mlir.lowering.record import registry as record_lowering_registry
        from numba_cuda_mlir.lowering.unicode import (
            registry as unicode_lowering_registry,
        )
        from numba_cuda_mlir.lowering.nrt import registry as nrt_lowering_registry
        from numba_cuda_mlir.extending import (
            lowering_registry as extending_lowering_registry,
        )
        from numba_cuda_mlir.lowering.enum import registry as enum_lowering_registry
        from numba_cuda_mlir.lowering.datetime import (
            registry as datetime_lowering_registry,
        )

        import numba_cuda_mlir.lowering.cpython  # noqa: F401

        # Install registries in order (foundational first, specialized last)
        self.install_registry(builtins_lowering_registry)
        self.install_registry(math_lowering_registry)
        self.install_registry(cmath_lowering_registry)
        self.install_registry(numpy_lowering_registry)
        self.install_registry(libdevice_lowering_registry)
        self.install_registry(cuda_lowering_registry)
        self.install_registry(print_lowering_registry)
        self.install_registry(ctypes_lowering_registry)
        self.install_registry(struct_lowering_registry)
        self.install_registry(union_lowering_registry)
        self.install_registry(record_lowering_registry)
        self.install_registry(half_precision_lowering_registry)
        self.install_registry(exotic_float_lowering_registry)
        self.install_registry(vector_lowering_registry)
        self.install_registry(cuda_vector_types_lowering_registry)
        self.install_registry(enum_lowering_registry)
        self.install_registry(unicode_lowering_registry)
        self.install_registry(nrt_lowering_registry)
        self.install_registry(datetime_lowering_registry)
        self.install_registry(extending_lowering_registry)

    @override
    def refresh(self):
        if self._registries_unchanged():
            return
        self.load_additional_registries()

    def get_overload_builder(self, fn, sig):
        """Return an MLIR builder for an overloaded function, or None.

        Searches the typing templates for an overload Dispatcher that
        can be compiled through numba_cuda_mlir's MLIR pipeline. For BoundFunction
        types, resolves the underlying overload function's templates.
        """
        if not isinstance(fn, types.Callable):
            return None

        templates = list(getattr(fn, "templates", []))

        if isinstance(fn, types.BoundFunction):
            overload_func = getattr(fn.template, "_overload_func", None)
            if overload_func is not None:
                inner_fnty = self.typing_context.resolve_value_type(overload_func)
                templates.extend(getattr(inner_fnty, "templates", []))

        match_args = (sig.recvr, *sig.args) if sig.recvr else sig.args

        for temp_cls in templates:
            if not hasattr(temp_cls, "_impl_cache"):
                continue
            for cache_key, cache_value in temp_cls._impl_cache.items():
                if cache_value is None or len(cache_key) != 4:
                    continue
                _, args, _, _ = cache_key
                cache_args = tuple(args)
                non_omitted_cache_args = tuple(
                    arg
                    for arg in cache_args
                    if not isinstance(arg, (types.Omitted, types.NoneType))
                )
                non_omitted_match_args = tuple(
                    arg
                    for arg in match_args
                    if not isinstance(arg, (types.Omitted, types.NoneType))
                )
                if cache_args == match_args or non_omitted_cache_args == non_omitted_match_args:
                    disp, _ = cache_value
                    if hasattr(disp, "py_func"):

                        def builder(mlir_lower, target, args, kws, _disp=disp):
                            mlir_lower.lower_overload_call(target, _disp, args, kws)

                        return builder
        return None

    def get_value_type(self, *args):
        return super().get_value_type(*args)

    def get_setattr(self, attr, sig):
        """
        Get the setattr() implementation for the given attribute name
        and signature, filtering out upstream numba lowerings.
        """
        assert len(sig.args) == 2
        typ = sig.args[0]
        valty = sig.args[1]

        def wrap_setattr(impl):
            def wrapped(builder, args):
                return impl(self, builder, sig, args, attr)

            return wrapped

        # Lookup specific setattr implementation for this type and attribute
        overloads = self._setattrs[attr]
        self._filter_numba_lowerings(overloads)
        try:
            return wrap_setattr(overloads.find((typ, valty)))
        except errors.NumbaNotImplementedError:
            pass

        # Lookup generic setattr implementation for this type
        overloads = self._setattrs[None]
        self._filter_numba_lowerings(overloads)
        try:
            return wrap_setattr(overloads.find((typ, valty)))
        except errors.NumbaNotImplementedError:
            pass

        raise NotImplementedError("No definition for lowering %s.%s = %s" % (typ, attr, valty))

    def _filter_numba_lowerings(self, overloads):
        filtered_versions = list(
            filter(lambda x: x[1].__module__.split(".")[0] != "numba", overloads.versions)
        )
        overloads.versions = type(overloads.versions)(filtered_versions)

    def _find_module_getattr_by_name(self, typ, attr):
        """
        Find a getattr implementation for a Module type by matching module name patterns.

        This handles cases where different module objects represent the same logical
        module. For example, `numba.cuda` might be redirected to `numba_cuda.numba.cuda`.
        """
        if not isinstance(typ, types.Module):
            return None

        overloads = self._getattrs.get(attr)
        if overloads is None:
            return None

        # Get the module name of the type we're looking for
        target_mod_name = getattr(typ.pymod, "__name__", "")

        # Define module name equivalences - these all refer to "cuda" modules
        cuda_module_names = {
            "numba_cuda_mlir.cuda",
            "numba_cuda_mlir.numba_cuda",
            "numba.cuda",
            "numba_cuda.numba.cuda",
        }

        # Check if target is a cuda-like module
        is_cuda_module = target_mod_name in cuda_module_names

        if not is_cuda_module:
            return None

        # Find a registered implementation for any equivalent cuda module
        for sig, impl in overloads.versions:
            if len(sig) != 1:
                continue
            registered_typ = sig[0]
            if not isinstance(registered_typ, types.Module):
                continue
            registered_mod_name = getattr(registered_typ.pymod, "__name__", "")
            if registered_mod_name in cuda_module_names:
                # Found a match - return the implementation
                return impl

        return None

    @override
    def get_getattr(self, typ, attr):
        """
        Get the getattr() implementation for the given type and attribute name.
        The return value is a callable with the signature
        (context, builder, typ, val, attr).
        """
        const_attr = not self._is_nonconst_module_attr(typ, attr)
        is_module = isinstance(typ, types.Module)

        if is_module and const_attr:
            # Implement getattr for module-level globals that we treat as
            # constants.
            # XXX We shouldn't have to retype this
            attrty = self.typing_context.resolve_module_constants(typ, attr)
            if attrty is None or isinstance(attrty, types.Dummy):
                # No implementation required for dummies (functions, modules...),
                # which are dealt with later
                return None
            else:
                pyval = getattr(typ.pymod, attr)

                # TODO(ajm): I'm not sure what purpose this serves.
                # In other similar situations where we grab a constant getattr, we just
                # return it. I don't see why we need this borrowed return.
                # We may need to re-enable this when we better understand its purpose.

                # def imp(context, builder, typ, val, attr):
                #     llval = self.get_constant_generic(builder, attrty, pyval)
                #     return impl_ret_borrowed(context, builder, attrty, llval)
                # return imp

                def imp(context, builder, target, value, attr):
                    from numba_cuda_mlir.lowering_utilities import convert

                    target_type = builder.get_mlir_type(target)
                    mod = builder.load_var(value)
                    pyval = getattr(mod, attr)
                    pyval = convert(pyval, target_type)
                    builder.store_var(target, pyval)

                return imp

        if isinstance(typ, Namespace):

            def imp(context, builder, target, val, attr):
                """
                We already know what attributes the library has, just use getattr
                to get the function and we'll link it in later.
                """
                library = builder.load_var(val)
                assert hasattr(library, attr), f"Library {library} has no attribute {attr!r}"
                func = getattr(library, attr)
                builder.store_var(target, func)

            return imp

        # Lookup specific getattr implementation for this type and attribute
        overloads = self._getattrs[attr]
        # Remove lowerings from upstream numba
        self._filter_numba_lowerings(overloads)
        try:
            return overloads.find((typ,))
        except errors.NumbaNotImplementedError:
            pass

        # For Module types, try matching by module name pattern
        # This handles cases where the redirector creates different module objects
        # for the same logical module (e.g., numba.cuda -> numba_cuda.numba.cuda)
        if is_module and attr in self._getattrs:
            impl = self._find_module_getattr_by_name(typ, attr)
            if impl is not None:
                return impl

        # Lookup generic getattr implementation for this type
        overloads = self._getattrs[None]
        # Remove lowerings from upstream numba
        self._filter_numba_lowerings(overloads)
        try:
            return overloads.find((typ,))
        except errors.NumbaNotImplementedError:
            pass

        raise NotImplementedError("No definition for lowering %s.%s" % (typ, attr))


class MLIRTargetOptions(TargetOptions):
    pass


class MLIRTarget(TargetDescriptor):
    def __init__(self, name):
        self.options = MLIRTargetOptions
        # The typing and target contexts are initialized only when needed -
        # this prevents an attempt to load CPU libraries at import time on
        # systems that might not have them present.
        self._typingctx = None
        self._targetctx = None
        self._typingctx_initialized = False
        self._targetctx_initialized = False
        self._initializing = False
        super().__init__(name)

    @property
    def typing_context(self):
        if self._typingctx is None:
            self._typingctx = MLIRTypingContext()
        return self._typingctx

    @property
    def target_context(self):
        if self._targetctx is None:
            # Ensure typing context is initialized before target context.
            self._targetctx = MLIRTargetContext(self.typing_context)
        return self._targetctx

    def ensure_initialized(self):
        if self._typingctx_initialized and self._targetctx_initialized:
            return
        if self._initializing:
            return

        self._initializing = True
        try:
            if not self._typingctx_initialized:
                try:
                    self.typing_context.refresh()
                except Exception:
                    self._typingctx_initialized = False
                    raise
                else:
                    self._typingctx_initialized = True

            if not self._targetctx_initialized:
                try:
                    self.target_context.refresh()
                except Exception:
                    self._targetctx_initialized = False
                    raise
                else:
                    self._targetctx_initialized = True

            from numba_cuda_mlir.numba_cuda.typing.templates import builtin_registry

            self.typing_context.install_registry(builtin_registry)
        finally:
            self._initializing = False

    def refresh_registries(self, *, typing=True, target=True):
        if self._initializing:
            return
        self._initializing = True
        try:
            if typing:
                self.typing_context.refresh()
                self._typingctx_initialized = True
            if target:
                self.target_context.refresh()
                self._targetctx_initialized = True
        finally:
            self._initializing = False


mlir_target = MLIRTarget("numba_cuda_mlir")


def _get_cuda_base():
    """Return the CUDA target base class if present, else CPU.

    This allows CUDA-targeted typing templates (e.g., for cuda.grid) to be
    considered usable under the MLIR target by making MLIR inherit from the
    CUDA target class. Falls back to CPU if CUDA is unavailable.
    """
    try:
        return target_registry["cuda"]
    except KeyError:
        return CPU


class MLIR(_get_cuda_base()):
    """Mark the target as mlir."""


_CompileStats = collections.namedtuple(
    "_CompileStats",
    (
        "cache_path",
        "cache_hits",
        "cache_misses",
    ),
)
_LaunchConfigCompileStats = collections.namedtuple(
    "_LaunchConfigCompileStats",
    (
        "cache_hits",
        "cache_misses",
    ),
)


class MLIRDispatcherType(types.Dispatcher):
    """The type of MLIR dispatchers"""

    @property
    def templates(self):
        """
        The type system checks for templates when type checking, but dispatchers
        can be generic without templates since we can just recompile them for a new
        signature if the arguments at the callsite don't match previously compiled
        signatures. So, just return an empty list here.
        """
        return []


class MLIRDispatcher(Dispatcher, serialize.ReduceMixin):
    _fold_args = False
    targetdescr = mlir_target

    def __init__(self, py_func, targetoptions=None):
        from numba_cuda_mlir.mlir_compiler import get_compiler_class

        if targetoptions is None:
            targetoptions = {}
        # AST transforms now happen at compile time (in compile_mlir) when we
        # have the signature (argtypes) available. This allows consteval to
        # access argument types and target options.
        super().__init__(py_func, targetoptions=targetoptions)

        # ``Dispatcher.__init__`` constructs ``self._compiler`` with
        # ``pipeline_class=None``. We need to set the pipeline class for
        # _OverloadFunctionTemplate.generic() to use it when an overload is
        # created with inline="always".
        self._compiler.pipeline_class = get_compiler_class(targetoptions)

        # Free-threaded launches take _launch_lock before configure/cache work
        # that may take _launch_config_lock. Keep that order one-way.
        self._launch_config_lock = _new_dispatcher_rlock()
        self._launch_lock = _new_dispatcher_rlock() if _PY_GIL_DISABLED else None
        # Launch-specialized overloads are retained until either recompile() or
        # cache eviction, whichever happens first. Retained marshallers can
        # reinstall their native dispatcher after dispatcher cache eviction, but
        # recompile() invalidates old launch-config generations.
        self._launch_config_overloads = {}
        self._launch_config_dispatchers = collections.OrderedDict()
        self._launch_config_generation = 0
        self._requires_launch_config = False
        self._c = self._new_kernel_dispatcher()
        self.extensions = targetoptions.get("extensions") or []
        self._specialized = False
        self.specializations = {}
        self._module_setup_callbacks = []
        self._module_teardown_callbacks = []
        self._old_dispatchers = []
        self._configure_cache = collections.OrderedDict()
        self._configure_cache_inflight = {}
        self.configure = _ConfigureProxy(self)
        for link_item in targetoptions.get("link", []):
            if hasattr(link_item, "setup_callback") and link_item.setup_callback:
                self._module_setup_callbacks.append(link_item.setup_callback)
            if hasattr(link_item, "teardown_callback") and link_item.teardown_callback:
                self._module_teardown_callbacks.append(link_item.teardown_callback)

        # Caching support
        self._cache = NullCache()
        self._cache_hits = collections.Counter()
        self._cache_misses = collections.Counter()
        self._launch_config_cache_hits = collections.Counter()
        self._launch_config_cache_misses = collections.Counter()
        self._launch_config_cache_notice_emitted = False

        # Checked by type inferer (numba-cuda's typeinfer.py) to detect self-recursive calls
        self._is_compiling = False

    @property
    def is_compiling(self):
        return self._is_compiling

    @property
    def _numba_type_(self):
        return MLIRDispatcherType(self)

    def _new_kernel_dispatcher(self):
        # Only debug kernels pay the per-launch device-side error readback
        # (cuCtxSynchronize + status D2H). This mirrors numba-cuda, where kernel
        # exceptions (raise/assert/bounds checks) surface only with debug=True;
        # debug=False launches stay asynchronous.
        return _cext.KernelDispatcher(
            self._compile,
            get_constant_args(self.py_func),
            _ensure_numba_cuda_context,
            debug=bool(self.targetoptions.get("debug", False)),
        )

    @property
    def _launch_config_enabled(self):
        return self._requires_launch_config or _extensions_use_launch_config(self.extensions)

    def _mark_requires_launch_config(self):
        with self._launch_config_lock:
            self._requires_launch_config = True

    def _get_kernel_dispatcher(self, launch_config):
        dispatcher, _ = self._get_kernel_dispatcher_and_generation(launch_config)
        return dispatcher

    def _get_kernel_dispatcher_and_generation(self, launch_config):
        self._ensure_dispatcher_state()
        if not self._launch_config_enabled:
            return self._c, None
        return self._get_launch_config_dispatcher_and_generation(launch_config)

    def _get_launch_config_dispatcher_and_generation(self, launch_config):
        """Get a launch-keyed dispatcher independent of current extensions."""

        self._ensure_dispatcher_state()
        launch_config_key = _launch_config_key(launch_config)
        if launch_config_key is None:
            return self._c, None

        while True:
            with self._launch_config_lock:
                dispatcher = self._launch_config_dispatchers.get(launch_config_key)
                generation = self._launch_config_generation
                if dispatcher is not None:
                    self._launch_config_dispatchers.move_to_end(launch_config_key)
                    return dispatcher, generation

            # Construct outside the lock. If another thread wins the insertion
            # race or recompile() advances the generation, this passive native
            # dispatcher is discarded without ever launching.
            dispatcher = self._new_kernel_dispatcher()
            with self._launch_config_lock:
                if generation != self._launch_config_generation:
                    continue
                existing = self._launch_config_dispatchers.get(launch_config_key)
                if existing is not None:
                    self._launch_config_dispatchers.move_to_end(launch_config_key)
                    return existing, generation
                self._remember_kernel_dispatcher_locked(launch_config_key, dispatcher)
                return dispatcher, generation

    def _remember_kernel_dispatcher(
        self,
        launch_config,
        kernel_dispatcher,
        launch_config_generation=None,
        replace_existing=True,
    ):
        self._ensure_dispatcher_state()
        launch_config_key = _launch_config_key(launch_config)
        if launch_config_key is None:
            return False
        with self._launch_config_lock:
            if (
                launch_config_generation is not None
                and launch_config_generation != self._launch_config_generation
            ):
                return False
            return self._remember_kernel_dispatcher_locked(
                launch_config_key,
                kernel_dispatcher,
                replace_existing=replace_existing,
            )

    def _is_kernel_dispatcher_registered(
        self, launch_config, kernel_dispatcher, launch_config_generation=None
    ):
        self._ensure_dispatcher_state()
        launch_config_key = _launch_config_key(launch_config)
        if launch_config_key is None:
            return False
        with self._launch_config_lock:
            if (
                launch_config_generation is not None
                and launch_config_generation != self._launch_config_generation
            ):
                return False
            return self._launch_config_dispatchers.get(launch_config_key) is kernel_dispatcher

    def _ensure_dispatcher_state(self):
        # ReduceMixin restore paths may bypass __init__, so recreate the new
        # mutable caches lazily before paths that append or clear them.
        if all(hasattr(self, name) for name in _DISPATCHER_STATE_ATTRS) and (
            "configure" in self.__dict__
        ):
            return

        with _DISPATCHER_STATE_BOOTSTRAP_LOCK:
            if not hasattr(self, "_launch_config_lock"):
                self._launch_config_lock = _new_dispatcher_rlock()
            if not hasattr(self, "_launch_lock"):
                self._launch_lock = _new_dispatcher_rlock() if _PY_GIL_DISABLED else None
            if not hasattr(self, "_launch_config_overloads"):
                self._launch_config_overloads = {}
            if not hasattr(self, "_launch_config_dispatchers"):
                self._launch_config_dispatchers = collections.OrderedDict()
            if not hasattr(self, "_launch_config_generation"):
                self._launch_config_generation = 0
            if not hasattr(self, "_requires_launch_config"):
                self._requires_launch_config = False
            if not hasattr(self, "_launch_config_cache_misses"):
                self._launch_config_cache_misses = collections.Counter()
            if not hasattr(self, "_launch_config_cache_hits"):
                self._launch_config_cache_hits = collections.Counter()
            if not hasattr(self, "_launch_config_cache_notice_emitted"):
                self._launch_config_cache_notice_emitted = False
            if not hasattr(self, "_old_dispatchers"):
                self._old_dispatchers = []
            if not hasattr(self, "_configure_cache"):
                self._configure_cache = collections.OrderedDict()
            if not hasattr(self, "_configure_cache_inflight"):
                self._configure_cache_inflight = {}
            if "configure" not in self.__dict__:
                self.configure = _ConfigureProxy(self)

    def _clear_configure_cache(self):
        self._ensure_dispatcher_state()
        with self._launch_config_lock:
            self._configure_cache.clear()
            inflight = tuple(self._configure_cache_inflight.values())
            self._configure_cache_inflight.clear()
            for event in inflight:
                event.set()

    def _retain_old_dispatcher(self, kernel_dispatcher):
        self._ensure_dispatcher_state()
        self._old_dispatchers.append(kernel_dispatcher)
        overflow = len(self._old_dispatchers) - _OLD_DISPATCHER_RETAIN_LIMIT
        if overflow > 0:
            # The cap trades crash-avoidance retention against unbounded growth;
            # users with very long-lived dispatchers can raise it via the env var.
            del self._old_dispatchers[:overflow]

    def _remember_kernel_dispatcher_locked(
        self, launch_config_key, kernel_dispatcher, replace_existing=True
    ):
        existing = self._launch_config_dispatchers.get(launch_config_key)
        if existing is not None:
            if existing is not kernel_dispatcher:
                if not replace_existing:
                    return False
                self._retain_old_dispatcher(existing)
                self._launch_config_dispatchers[launch_config_key] = kernel_dispatcher
            self._launch_config_dispatchers.move_to_end(launch_config_key)
            return True
        self._launch_config_dispatchers[launch_config_key] = kernel_dispatcher
        self._launch_config_dispatchers.move_to_end(launch_config_key)
        while len(self._launch_config_dispatchers) > _LAUNCH_CONFIG_CACHE_SIZE:
            evicted_key, evicted_dispatcher = self._launch_config_dispatchers.popitem(last=False)
            self._retain_old_dispatcher(evicted_dispatcher)
            for key in list(self._launch_config_overloads):
                if key[1] == evicted_key:
                    del self._launch_config_overloads[key]
        return True

    def configure(self, griddim, blockdim, stream=None, sharedmem=None, cluster=None):
        griddim, blockdim = normalize_kernel_dimensions(griddim, blockdim)
        if cluster is not None:
            cluster = normalize_kernel_dimensions(cluster, (1, 1, 1))[0]
        launch_config_enabled = self._launch_config_enabled
        extension_key = tuple(_ObjectIdentityKey(ext) for ext in self.extensions)
        configured_sharedmem = sharedmem
        if launch_config_enabled:
            configured_sharedmem = _normalize_launch_sharedmem(sharedmem)
        cache_key = (
            launch_config_enabled,
            extension_key,
            griddim,
            blockdim,
            stream,
            configured_sharedmem,
            cluster,
        )
        _validate_configure_cache_key(cache_key)
        self._ensure_dispatcher_state()

        stale_retries = 0
        while True:
            with self._launch_config_lock:
                cached = self._configure_cache.get(cache_key)
                if cached is not None:
                    self._configure_cache.move_to_end(cache_key)
                    return cached
                inflight = self._configure_cache_inflight.get(cache_key)
                if inflight is None:
                    inflight = threading.Event()
                    self._configure_cache_inflight[cache_key] = inflight
                    owns_inflight = True
                else:
                    owns_inflight = False

            if not owns_inflight:
                inflight.wait()
                continue

            try:
                marshaller = self._configure_cached(*cache_key)
            except BaseException:
                with self._launch_config_lock:
                    if self._configure_cache_inflight.get(cache_key) is inflight:
                        del self._configure_cache_inflight[cache_key]
                    inflight.set()
                raise

            retry_stale_marshaller = False
            with self._launch_config_lock:
                try:
                    cached = self._configure_cache.get(cache_key)
                    if cached is not None:
                        self._configure_cache.move_to_end(cache_key)
                        return cached
                    if self._configure_marshaller_stale_locked(launch_config_enabled, marshaller):
                        stale_retries += 1
                        if stale_retries >= _CONFIGURE_CACHE_STALE_RETRY_LIMIT:
                            raise RuntimeError(
                                "Kernel launch configuration was invalidated repeatedly "
                                "during configure(); retry after concurrent recompile() calls finish."
                            )
                        retry_stale_marshaller = True
                    else:
                        self._configure_cache[cache_key] = marshaller
                        self._configure_cache.move_to_end(cache_key)
                        while len(self._configure_cache) > _CONFIGURE_CACHE_SIZE:
                            self._configure_cache.popitem(last=False)
                        return marshaller
                finally:
                    if self._configure_cache_inflight.get(cache_key) is inflight:
                        del self._configure_cache_inflight[cache_key]
                    inflight.set()
            if retry_stale_marshaller:
                continue

    def _configure_marshaller_stale_locked(self, launch_config_enabled, marshaller):
        if launch_config_enabled:
            return marshaller._launch_config_generation != self._launch_config_generation
        return marshaller._kernel_dispatcher is not self._c

    def _configure_cached(
        self,
        launch_config_enabled,
        extension_key,
        griddim,
        blockdim,
        stream,
        configured_sharedmem,
        cluster,
    ):
        # Warn when the grid has fewer than 128 blocks (low occupancy)
        if cuda_config.CUDA_LOW_OCCUPANCY_WARNINGS and not cuda_config.DISABLE_PERFORMANCE_WARNINGS:
            min_grid_size = 128
            grid_size = griddim[0] * griddim[1] * griddim[2]
            if grid_size < min_grid_size:
                msg = (
                    f"Grid size {grid_size} will likely result in GPU "
                    "under-utilization due to low occupancy."
                )
                warnings.warn(NumbaPerformanceWarning(msg))

        launch_config = None
        available_launch_config = {
            "grid": griddim,
            "block": blockdim,
            "sharedmem": configured_sharedmem,
            "cluster": cluster,
        }
        kernel_dispatcher = self._c
        launch_config_generation = None
        stream_ref = None
        launch_stream = stream
        if isinstance(stream, numba_cuda_driver.Stream):
            stream_ref = stream
            launch_stream = int(stream.handle)
        if launch_config_enabled:
            launch_config = available_launch_config
            (
                kernel_dispatcher,
                launch_config_generation,
            ) = self._get_kernel_dispatcher_and_generation(launch_config)
        extensions = [key.obj for key in extension_key]

        return _ArgMarshaller(
            LaunchConfiguration(
                kernel_dispatcher, griddim, blockdim, launch_stream, configured_sharedmem, cluster
            ),
            extensions=extensions,
            dispatcher=self,
            launch_config=launch_config,
            available_launch_config=available_launch_config,
            kernel_dispatcher=kernel_dispatcher,
            launch_config_generation=launch_config_generation,
            stream_ref=stream_ref,
            launch_stream=launch_stream,
        )

    def _reduce_states(self):
        """Serialize compiled signatures, including launch-qualified entries."""
        self._ensure_dispatcher_state()
        if self._can_compile:
            sigs = []
            launch_config_sigs = []
        else:
            # Once a planner requests launch metadata, generic overloads are
            # no longer reachable through dispatch. Rebuilding them would also
            # rerun the active planner without a configured launch.
            sigs = (
                []
                if self._requires_launch_config
                else [cr.signature for cr in self.overloads.values()]
            )
            launch_config_sigs = []
            if self._launch_config_enabled:
                seen = set()
                with self._launch_config_lock:
                    launch_items = list(self._launch_config_overloads.items())
                for (_sig_args, launch_config_key), cres in launch_items:
                    cres_id = id(cres)
                    if cres_id in seen:
                        continue
                    seen.add(cres_id)
                    launch_config_sigs.append((cres.signature, launch_config_key))

        return dict(
            uuid=str(self._uuid),
            py_func=self.py_func,
            locals=self.locals,
            targetoptions=self.targetoptions,
            can_compile=self._can_compile,
            sigs=sigs,
            launch_config_sigs=launch_config_sigs,
            requires_launch_config=self._requires_launch_config,
        )

    @classmethod
    def _rebuild(
        cls,
        uuid,
        py_func,
        locals,
        targetoptions,
        can_compile,
        sigs,
        launch_config_sigs=(),
        requires_launch_config=False,
    ):
        """Rebuild an MLIRDispatcher after serialization."""
        try:
            return cls._memo[uuid]
        except KeyError:
            pass
        self = cls(py_func, targetoptions=targetoptions)
        self.locals = locals
        self._set_uuid(uuid)
        self._requires_launch_config = bool(requires_launch_config)
        for sig in sigs:
            self.compile(sig)
        for sig, launch_config_key in launch_config_sigs:
            self._compile_launch_config_signature(sig, launch_config_key)
        self._can_compile = can_compile
        return self

    def _compile_launch_config_signature(self, sig, launch_config_key):
        """Rebuild a serialized launch-specialized signature via the dispatch path.

        Reduce state stores signatures, not runtime argument values, so override_argtypes
        drives compilation while placeholder values satisfy the dispatch entry point.
        """
        argtypes, _return_type = sigutils.normalize_signature(sig)
        launch_config = _launch_config_dict_from_key(launch_config_key)
        launch_config_key = _launch_config_key(launch_config)

        with self._launch_config_lock:
            existing = self._launch_config_overloads.get((argtypes, launch_config_key))
            if existing is not None:
                return existing

        previous_types = getattr(_compile_arg_types, "types", _MISSING)
        previous_launch_config = getattr(_compile_arg_types, "launch_config", _MISSING)
        previous_extensions = getattr(_compile_arg_types, "extensions", _MISSING)
        previous_force_launch_config = getattr(_compile_arg_types, "force_launch_config", _MISSING)
        try:
            _compile_arg_types.types = argtypes
            _compile_arg_types.launch_config = launch_config
            _compile_arg_types.extensions = self.extensions
            _compile_arg_types.force_launch_config = True
            self._compile_impl([None] * len(argtypes))
        finally:
            if previous_types is _MISSING:
                if hasattr(_compile_arg_types, "types"):
                    delattr(_compile_arg_types, "types")
            else:
                _compile_arg_types.types = previous_types
            if previous_launch_config is _MISSING:
                if hasattr(_compile_arg_types, "launch_config"):
                    delattr(_compile_arg_types, "launch_config")
            else:
                _compile_arg_types.launch_config = previous_launch_config
            if previous_extensions is _MISSING:
                if hasattr(_compile_arg_types, "extensions"):
                    delattr(_compile_arg_types, "extensions")
            else:
                _compile_arg_types.extensions = previous_extensions
            if previous_force_launch_config is _MISSING:
                if hasattr(_compile_arg_types, "force_launch_config"):
                    delattr(_compile_arg_types, "force_launch_config")
            else:
                _compile_arg_types.force_launch_config = previous_force_launch_config

        with self._launch_config_lock:
            rebuilt = self._find_launch_config_overload_locked(argtypes, launch_config_key)
        if rebuilt is None:
            raise RuntimeError(f"Failed to rebuild launch-config specialization for {argtypes}")
        return rebuilt

    def _apply_shared_memory_carveout(self, wrapped):
        carveout = self.targetoptions.get("shared_memory_carveout")
        if carveout is None:
            return
        if isinstance(carveout, str):
            carveout_map = {"default": -1, "maxl1": 0, "maxshared": 100}
            carveout = carveout_map[carveout.lower()]
        wrapped._codelibrary.get_cufunc().set_shared_memory_carveout(carveout)

    def __getitem__(self, args):
        assert isinstance(args, tuple)
        assert len(args) in (2, 3, 4, 5)
        return self.configure(*args)

    def enable_caching(self):
        """Enable on-disk caching for this dispatcher."""
        self._cache = MLIRCache(self.py_func, self.targetoptions)

    def _resolve_target_options(self):
        from numba_cuda_mlir.tools import resolve_target_options

        resolve_target_options(self.targetoptions)

    @property
    def stats(self):
        """Return cache statistics."""
        return _CompileStats(
            cache_path=self._cache.cache_path,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
        )

    @property
    def launch_config_stats(self):
        """Return cache counters for launch-config-specialized compiles."""
        self._ensure_dispatcher_state()
        return _LaunchConfigCompileStats(
            cache_hits=self._launch_config_cache_hits,
            cache_misses=self._launch_config_cache_misses,
        )

    def _launch_config_compile_results(self):
        self._ensure_dispatcher_state()
        results = []
        seen = set()
        with self._launch_config_lock:
            for cres in self._launch_config_overloads.values():
                cres_id = id(cres)
                if cres_id in seen:
                    continue
                seen.add(cres_id)
                results.append(cres)
        return results

    def _launch_config_overload_argtypes(self, launch_config_key):
        self._ensure_dispatcher_state()
        # The launch overload table is bounded by the dispatcher cache and is
        # small in normal use; keep a flat table instead of a secondary index.
        signatures = []
        seen = set()
        with self._launch_config_lock:
            for sig_args, key in self._launch_config_overloads:
                if key != launch_config_key or sig_args in seen:
                    continue
                seen.add(sig_args)
                signatures.append(sig_args)
        return tuple(signatures)

    @property
    def signatures(self):
        """Return compiled overload keys.

        Generic entries use Numba-compatible argtypes tuples. Launch entries use
        LaunchConfigInspectableKey so same-argtype launch specializations remain
        distinguishable.
        """
        self._ensure_dispatcher_state()
        signatures = list(self.overloads)
        seen = set(signatures)
        seen_cres = set()
        with self._launch_config_lock:
            launch_items = list(self._launch_config_overloads.items())
        for (sig_args, launch_config_key), cres in launch_items:
            cres_id = id(cres)
            if cres_id in seen_cres:
                continue
            seen_cres.add(cres_id)
            key = LaunchConfigInspectableKey(sig_args, launch_config_key)
            if key not in seen:
                seen.add(key)
                signatures.append(key)
        return signatures

    @property
    def nopython_signatures(self):
        """Return unique nopython signatures across generic and launch-specific overloads."""
        signatures = [cres.signature for cres in self.overloads.values() if not cres.objectmode]
        seen = {tuple(sig.args) for sig in signatures}
        for cres in self._launch_config_compile_results():
            if cres.objectmode:
                continue
            sig_args = tuple(cres.signature.args)
            if sig_args not in seen:
                seen.add(sig_args)
                signatures.append(cres.signature)
        return signatures

    def __call__(self, *args, **kwargs):
        """
        Compile if necessary and invoke this kernel with *args*.
        """
        raise ValueError("launch configuration was not specified")

    def _find_overload(self, sig):
        """Find an overload matching the given signature.

        The signature can be:
        - A tuple of argument types (e.g., from inspect_asm(signature=...))
        - A typing.Signature object
        - Already in self.overloads
        - A raw (argtypes, launch_config_key) key from launch_config_overloads
        - A LaunchConfigInspectableKey from no-argument inspect_*() results

        Returns the compile result if found, or compiles and returns it.
        """
        if isinstance(sig, tuple) and len(sig) == 2 and isinstance(sig[1], dict):
            if _is_launch_config_dict(sig[1]):
                sig = (sig[0], _launch_config_key(sig[1]))
            else:
                raise KeyError(f"No overload found for signature {sig}")
        launch_key = None
        if isinstance(sig, LaunchConfigInspectableKey):
            launch_key = (sig.argtypes, sig.launch_config_key)
        elif isinstance(sig, tuple) and len(sig) == 2 and _is_launch_config_key_tuple(sig[1]):
            launch_key = sig
        if launch_key is not None:
            argtypes, launch_config_key = launch_key
            with self._launch_config_lock:
                found = self._find_launch_config_overload_locked(argtypes, launch_config_key)
                if found is not None:
                    return found
            raise KeyError(f"No launch-config overload found for signature {sig}")

        # Direct lookup
        if sig in self.overloads:
            return self.overloads[sig]
        with self._launch_config_lock:
            if sig in self._launch_config_overloads:
                return self._launch_config_overloads[sig]

        # Extract args to search for matching overload
        search_args = sig.args if hasattr(sig, "args") else sig if isinstance(sig, tuple) else None
        if search_args is not None:
            for key, cres in self.overloads.items():
                key_args = (
                    key.args if hasattr(key, "args") else key if isinstance(key, tuple) else None
                )
                if key_args == search_args:
                    return cres
            launch_matches = []
            seen_launch_matches = set()
            with self._launch_config_lock:
                for (sig_args, _lc_key), cres in self._launch_config_overloads.items():
                    cres_id = id(cres)
                    if sig_args == search_args and cres_id not in seen_launch_matches:
                        seen_launch_matches.add(cres_id)
                        launch_matches.append(cres)
            if len(launch_matches) == 1:
                return launch_matches[0]
            if len(launch_matches) > 1:
                raise KeyError(
                    f"Multiple launch-config overloads found for signature {sig}; "
                    "use a launch-config-qualified key from inspect_*()"
                )

        # Not found - compile it
        self.compile(sig)
        # After compile, try lookup again
        if sig in self.overloads:
            return self.overloads[sig]
        # Search again for matching args
        if search_args is not None:
            for key, cres in self.overloads.items():
                key_args = (
                    key.args if hasattr(key, "args") else key if isinstance(key, tuple) else None
                )
                if key_args == search_args:
                    return cres

        raise KeyError(f"No overload found for signature {sig}")

    @property
    def launch_config_overloads(self):
        self._ensure_dispatcher_state()
        with self._launch_config_lock:
            return dict(self._launch_config_overloads)

    def get_metadata(self, signature=None):
        """Return compile metadata for generic and launch-specialized overloads."""
        if signature is not None:
            return self._find_overload(signature).metadata
        return {sig: cres.metadata for sig, cres in self._inspectable_overloads().items()}

    def _inspectable_overloads(self):
        self._ensure_dispatcher_state()
        with self._launch_config_lock:
            generic_overloads = dict(self.overloads)
            launch_overloads = dict(self._launch_config_overloads)
        if not launch_overloads:
            return generic_overloads
        overloads = generic_overloads
        seen = {id(cres) for cres in overloads.values()}
        for (sig_args, launch_config_key), cres in launch_overloads.items():
            if id(cres) in seen:
                continue
            seen.add(id(cres))
            overloads[LaunchConfigInspectableKey(sig_args, launch_config_key)] = cres
        return overloads

    def inspect_llvm(self, sig=None):
        raise NotImplementedError(
            "inspect_llvm is not supported. "
            "Use inspect_mlir() to inspect the MLIR module or inspect_asm() to inspect the PTX."
        )

    def inspect_asm(self, signature=None):
        """Get generated PTX.

        With no signature, launch-specialized entries are keyed by
        LaunchConfigInspectableKey and generic entries keep their argtypes tuple.
        """
        if signature is None:
            return {sig: self.inspect_asm(sig) for sig in self._inspectable_overloads()}
        cres = self._find_overload(signature)
        ptx = cres.metadata.get("ptx")
        if ptx:
            return ptx
        if not cres.metadata.get("ltoir"):
            return cres.metadata["ptx"]

        from numba_cuda_mlir.mlir_optimization import get_ptx

        ptx = get_ptx(cres)
        cres.metadata["ptx"] = ptx
        return ptx

    def inspect_mlir(self, sig=None):
        """Get generated MLIR.

        With no signature, launch-specialized entries are keyed by
        LaunchConfigInspectableKey and generic entries keep their argtypes tuple.
        """
        if sig is None:
            return {sig: self.inspect_mlir(sig) for sig in self._inspectable_overloads()}
        from numba_cuda_mlir.mlir_lowering import get_mlir_module_str

        cres = self._find_overload(sig)
        return get_mlir_module_str(cres.metadata)

    def inspect_mlir_optimized(self, sig=None):
        """Get optimized MLIR.

        With no signature, launch-specialized entries are keyed by
        LaunchConfigInspectableKey and generic entries keep their argtypes tuple.
        """
        if sig is None:
            return {sig: self.inspect_mlir_optimized(sig) for sig in self._inspectable_overloads()}
        cres = self._find_overload(sig)
        return cres.metadata["mlir_module_optimized"]

    def inspect_transformed_source(self, sig=None):
        """Return the AST-transformed source code for the given signature.

        Args:
            sig: The signature to get the transformed source for. If None, returns
                a dict mapping all compiled signatures to their transformed sources.

        Returns:
            The transformed source string, or None if no transforms were applied.
            If sig is None, returns a dict of {signature: transformed_source};
            launch-specialized entries use LaunchConfigInspectableKey keys.
        """
        if sig is None:
            return {
                sig: self.inspect_transformed_source(sig) for sig in self._inspectable_overloads()
            }
        cres = self._find_overload(sig)
        return cres.metadata.get("transformed_source")

    def inspect_ptx(self, sig=None):
        """Get generated PTX.

        With no signature, launch-specialized entries are keyed by
        LaunchConfigInspectableKey and generic entries keep their argtypes tuple.
        """
        if sig is None:
            return {sig: self.inspect_ptx(sig) for sig in self._inspectable_overloads()}
        return self.inspect_asm(sig)

    @staticmethod
    def _can_reuse_overload(sig_arg, runtime_type):
        """Check if a pre-compiled overload for sig_arg can be reused for
        runtime_type, allowing subtype compatibility."""
        if sig_arg == runtime_type:
            return True
        # If signature has CPointer and runtime is int (pointer address), accept
        if isinstance(sig_arg, types.CPointer) and isinstance(runtime_type, types.Integer):
            return True
        # Allow integer type coercion (e.g., int64 runtime -> int32 sig)
        if isinstance(sig_arg, types.Integer) and isinstance(runtime_type, types.Integer):
            return True
        # Allow array type compatibility (matching dtype and ndim)
        if isinstance(sig_arg, types.Array) and isinstance(runtime_type, types.Array):
            return sig_arg.dtype == runtime_type.dtype and sig_arg.ndim == runtime_type.ndim
        # Recursively check Tuple element compatibility
        if isinstance(sig_arg, types.BaseTuple) and isinstance(runtime_type, types.BaseTuple):
            if len(sig_arg) == len(runtime_type):
                reuse = MLIRDispatcher._can_reuse_overload
                return all(reuse(s, r) for s, r in zip(sig_arg, runtime_type))
        return False

    @staticmethod
    def _conversion_cost(sig_arg, runtime_type):
        """Return (unsafe, safe, promote) conversion cost, or None if impossible.
        Mirrors numba's Rating.astuple() ordering so lower tuples are better."""
        if sig_arg == runtime_type:
            return (0, 0, 0)
        if isinstance(sig_arg, types.Integer) and isinstance(runtime_type, types.Integer):
            if runtime_type.bitwidth <= sig_arg.bitwidth:
                return (0, 0, 1)
            return (1, 0, 0)
        if isinstance(sig_arg, types.Float) and isinstance(runtime_type, types.Float):
            if runtime_type.bitwidth <= sig_arg.bitwidth:
                return (0, 0, 1)
            return (1, 0, 0)
        if isinstance(sig_arg, types.Float) and isinstance(runtime_type, types.Integer):
            return (0, 1, 0)
        if isinstance(sig_arg, types.Integer) and isinstance(runtime_type, types.Float):
            return (1, 0, 0)
        if isinstance(sig_arg, types.Array) and isinstance(runtime_type, types.Array):
            if sig_arg.ndim != runtime_type.ndim:
                return None
            if sig_arg.dtype != runtime_type.dtype:
                return None
            return (0, 0, 0)
        if isinstance(sig_arg, types.CPointer) and isinstance(runtime_type, types.Integer):
            return (0, 1, 0)
        if isinstance(sig_arg, types.BaseTuple) and isinstance(runtime_type, types.BaseTuple):
            if len(sig_arg) != len(runtime_type):
                return None
            total = [0, 0, 0]
            for s, r in zip(sig_arg, runtime_type):
                cost = MLIRDispatcher._conversion_cost(s, r)
                if cost is None:
                    return None
                for i in range(3):
                    total[i] += cost[i]
            return tuple(total)
        return None

    @classmethod
    def _rate_overload(cls, sig_args, argtypes):
        """Sum per-argument conversion costs. Returns total tuple or None."""
        if len(sig_args) != len(argtypes):
            return None
        total = [0, 0, 0]
        for s, r in zip(sig_args, argtypes):
            cost = cls._conversion_cost(s, r)
            if cost is None:
                return None
            for i in range(3):
                total[i] += cost[i]
        return tuple(total)

    def _resolve_overload(self, argtypes, allow_unsafe=True, candidate_sig_args=None):
        """Find the best-matching overload(s) for argtypes.
        Returns (tied_best_sig_args_list, has_any_match)."""
        candidates = []
        if candidate_sig_args is None:
            candidate_sig_args = self.overloads
        for sig_args in candidate_sig_args:
            rating = self._rate_overload(sig_args, argtypes)
            if rating is None:
                continue
            if not allow_unsafe and rating[0] > 0:
                continue
            candidates.append((rating, sig_args))
        if not candidates:
            return [], False
        candidates.sort(key=lambda x: x[0])
        best_rate = candidates[0][0]
        tied = [sig_args for rate, sig_args in candidates if rate == best_rate]
        return tied, True

    def _compile_result_for(self, argtypes, launch_config):
        if launch_config is not None:
            launch_config_key = _launch_config_key(launch_config)
            with self._launch_config_lock:
                return self._find_launch_config_overload_locked(argtypes, launch_config_key)
        exact = self.overloads.get(argtypes)
        if exact is not None:
            return exact
        for sig_args, cres in self.overloads.items():
            if len(sig_args) == len(argtypes) and all(
                self._can_reuse_overload(s, r) for s, r in zip(sig_args, argtypes)
            ):
                return cres
        return None

    @global_compiler_lock
    def _prepare_for_launch(
        self,
        args,
        argtypes,
        available_launch_config,
        configured_launch_config,
        configured_kernel_dispatcher,
        configured_launch_config_generation,
    ):
        def active_launch_config():
            if self._requires_launch_config:
                return _normalize_available_launch_config(available_launch_config)
            return configured_launch_config

        launch_config = active_launch_config()
        compile_result = self._compile_result_for(argtypes, launch_config)
        if compile_result is None:
            self._compile_impl(list(args))
            launch_config = active_launch_config()
            compile_result = self._compile_result_for(argtypes, launch_config)

        if compile_result is None:
            raise RuntimeError(
                "kernel precompilation completed without publishing a matching overload"
            )

        required_dynamic_shared_memory = compile_result.metadata.get(
            _REQUIRED_DYNAMIC_SHARED_MEMORY_KEY, 0
        )
        if launch_config is None:
            return self._c, None, None, required_dynamic_shared_memory
        if not self._requires_launch_config and configured_launch_config is not None:
            if self._is_kernel_dispatcher_registered(
                configured_launch_config,
                configured_kernel_dispatcher,
                configured_launch_config_generation,
            ):
                return (
                    configured_kernel_dispatcher,
                    configured_launch_config_generation,
                    configured_launch_config,
                    required_dynamic_shared_memory,
                )
            # A retained marshaller may outlive recompile() or removal of the
            # extension that originally opted it into launch specialization.
            # Its extension snapshot still requires a launch-keyed native
            # dispatcher for the freshly compiled overload.
            kernel_dispatcher, generation = self._get_launch_config_dispatcher_and_generation(
                launch_config
            )
            return (
                kernel_dispatcher,
                generation,
                launch_config,
                required_dynamic_shared_memory,
            )
        kernel_dispatcher, generation = self._get_kernel_dispatcher_and_generation(launch_config)
        return (
            kernel_dispatcher,
            generation,
            launch_config,
            required_dynamic_shared_memory,
        )

    def _raise_ambiguous(self, argtypes, tied):
        sigs_str = "\n".join(f"{sig_args} -> none" for sig_args in tied)
        raise TypeError(f"Ambiguous overloading for {self.py_func!r} {argtypes}:\n{sigs_str}")

    def _propagate_compile_callbacks(self, metadata):
        self._ensure_dispatcher_state()
        # Reuse the dispatcher state lock so callback dedup is atomic with
        # launch-config compile publication.
        with self._launch_config_lock:
            for cb in metadata.get("setup_callbacks", []):
                if cb not in self._module_setup_callbacks:
                    self._module_setup_callbacks.append(cb)
            for cb in metadata.get("teardown_callbacks", []):
                if cb not in self._module_teardown_callbacks:
                    self._module_teardown_callbacks.append(cb)

    def _find_launch_config_overload_locked(self, argtypes, launch_config_key):
        # Exact hits are O(1). Compatible fallback scans the bounded launch
        # overload table to preserve existing overload-reuse semantics.
        exact = self._launch_config_overloads.get((argtypes, launch_config_key))
        if exact is not None:
            return exact
        reusable = None
        for (sig_args, key), cres in self._launch_config_overloads.items():
            if (
                key == launch_config_key
                and len(sig_args) == len(argtypes)
                and all(self._can_reuse_overload(s, r) for s, r in zip(sig_args, argtypes))
            ):
                reusable = cres
                break
        return reusable

    def _make_post_load_hook(self):
        """Create a callback for C++ to invoke after loading the CUlibrary."""
        setup_cbs = self._module_setup_callbacks
        teardown_cbs = self._module_teardown_callbacks

        def post_load(lib_handle_int):
            from cuda.bindings.driver import CUlibrary
            from numba_cuda_mlir.numba_cuda.cudadrv import devices

            # Wrap the raw C++ handle so callbacks can access it via obj.handle
            obj = _LoadedModule(CUlibrary(lib_handle_int))
            if teardown_cbs:
                ref = _ModuleCallbackRef(teardown_cbs, obj)
            else:
                ref = obj
            for cb in setup_cbs:
                cb(obj)
            ctx = devices.get_context()
            ctx.modules[id(ref)] = ref

        return post_load

    @contextmanager
    def _compile_profiler(self):
        profile_opt = self.targetoptions.get("profile_jit", False)
        if not profile_opt:
            yield
            return

        prof = cProfile.Profile()
        prof.enable()
        try:
            yield
        finally:
            prof.disable()
            print(
                f"\n--- cProfile for compilation of {self.py_func.__qualname__} ---",
                file=sys.stderr,
            )
            pstats.Stats(prof, stream=sys.stderr).sort_stats("cumulative").print_stats(50)
            if isinstance(profile_opt, str):
                prof.dump_stats(profile_opt)
                print(f"Profile saved to: {profile_opt}", file=sys.stderr)

    def _compile(self, args):
        with global_compiler_lock:
            cubin, func_name, cooperative = self._compile_impl(args)
            if self._module_setup_callbacks or self._module_teardown_callbacks:
                return (cubin, func_name, cooperative, self._make_post_load_hook())
            return (cubin, func_name, cooperative)

    def _compile_impl(self, args, launch_config_retry_budget=_CONFIGURE_CACHE_STALE_RETRY_LIMIT):
        from numba_cuda_mlir import mlir_compiler
        from numba_cuda_mlir.compiler import CompileResult

        self._ensure_dispatcher_state()

        # Get original arg types - either from thread-local storage (for tuple args)
        # or None to let mlir_compiler_entry infer from args
        override_argtypes = None
        if hasattr(_compile_arg_types, "types") and _compile_arg_types.types is not None:
            override_argtypes = tuple(_compile_arg_types.types)

        # For overload lookup, we need the effective argtypes
        if override_argtypes is not None:
            argtypes = override_argtypes
        else:
            argtypes = tuple(typeof(arg) for arg in args)
        active_extensions = getattr(_compile_arg_types, "extensions", _MISSING)
        has_extension_snapshot = active_extensions is not _MISSING
        if not has_extension_snapshot:
            active_extensions = self.extensions
        force_launch_config = getattr(_compile_arg_types, "force_launch_config", False)
        active_launch_config = None
        if (
            force_launch_config
            or self._requires_launch_config
            or _extensions_use_launch_config(active_extensions)
        ):
            active_launch_config = getattr(_compile_arg_types, "launch_config", None)
            if active_launch_config is None:
                active_launch_config = _normalize_available_launch_config(
                    getattr(_compile_arg_types, "available_launch_config", None)
                )
        if self._requires_launch_config and active_launch_config is None:
            raise RuntimeError(
                "whole-function planner requires launch metadata; compile through a "
                "configured kernel launch"
            )
        active_launch_config_key = _launch_config_key(active_launch_config)
        active_launch_config_generation = None
        if active_launch_config_key is not None:
            with self._launch_config_lock:
                active_launch_config_generation = self._launch_config_generation

        def _result(cres):
            return (
                cres.metadata["cubin"],
                cres.metadata["func_name"],
                cres.metadata.get("use_cooperative", False),
            )

        if active_launch_config_key is not None:
            # Launch metadata is visible to compilation. Keep those
            # specializations separate so consteval/rewrite users never observe
            # stale launch configuration values from a previous same-signature
            # launch.
            with self._launch_config_lock:
                reusable = self._find_launch_config_overload_locked(
                    argtypes, active_launch_config_key
                )
                if reusable is not None:
                    self._launch_config_cache_hits[(argtypes, active_launch_config_key)] += 1
                    return _result(reusable)
        elif self.overloads:
            for sig_args, cres in tuple(self.overloads.items()):
                if len(sig_args) == len(argtypes) and all(
                    self._can_reuse_overload(s, r) for s, r in zip(sig_args, argtypes)
                ):
                    return _result(cres)

        if not self._can_compile:
            if active_launch_config_key is not None:
                with self._launch_config_lock:
                    candidate_overloads = {}
                    for (sig_args, key), cres in self._launch_config_overloads.items():
                        if key == active_launch_config_key:
                            candidate_overloads[sig_args] = cres
                generic_note = (
                    f"; {len(self.overloads)} generic overload(s) ignored because "
                    "active extensions use launch metadata"
                    if self.overloads
                    else ""
                )
                best, has_any = self._resolve_overload(
                    argtypes,
                    allow_unsafe=True,
                    candidate_sig_args=tuple(candidate_overloads),
                )
                if not has_any:
                    # Generic overloads are not a safe fallback here: opt-in
                    # extensions may lower code from __launch_config__ metadata.
                    raise TypeError(
                        "No matching launch-config specialization for argument "
                        f"type(s) {argtypes} and launch config {active_launch_config}"
                        f"{generic_note}"
                    )
                if len(best) > 1:
                    # _resolve_overload returns tied best matches; it does not
                    # raise ambiguity on its own.
                    best_results = {id(candidate_overloads[sig_args]) for sig_args in best}
                    if len(best_results) > 1:
                        self._raise_ambiguous(argtypes, best)
                cres = candidate_overloads.get(best[0])
                if cres is None:
                    raise TypeError(
                        "No matching launch-config specialization for argument "
                        f"type(s) {argtypes} and launch config {active_launch_config}"
                        f"{generic_note}"
                    )
                return _result(cres)
            best, has_any = self._resolve_overload(argtypes, allow_unsafe=True)
            if not has_any:
                raise TypeError(f"No matching definition for argument type(s) {argtypes}")
            if len(best) > 1:
                self._raise_ambiguous(argtypes, best)
            cres = self.overloads[best[0]]
            return _result(cres)

        self._resolve_target_options()
        targetoptions = self.targetoptions
        if active_launch_config is not None or has_extension_snapshot:
            targetoptions = self.targetoptions.copy()
            if has_extension_snapshot:
                targetoptions["extensions"] = active_extensions
            if active_launch_config is not None:
                targetoptions["__launch_config__"] = dict(active_launch_config)

        # A cached result bypasses compiler passes. Until planner identity and
        # implementation are part of the persistent cache key, active planners
        # must compile in-process so their transformations cannot be skipped.
        disk_cache_eligible = active_launch_config_key is None
        disk_cache_miss = False

        # Serialize cache-hit publication with planner registration. If an
        # import during cache loading registers a planner reentrantly, discard
        # that cached result and compile through the planner pass instead.
        mlir_target.ensure_initialized()
        sig = typing.signature(types.none, *argtypes)
        # Launch-specialized compiles use an in-memory cache because the
        # generic in-memory and on-disk cache keys do not include launch metadata.
        if disk_cache_eligible:
            with _planner_registry._lock:
                if not _planner_registry._planners:
                    cres = self._cache.load_overload(sig, mlir_target.target_context)
                    if not _planner_registry._planners:
                        if cres is not None:
                            self._cache_hits[argtypes] += 1
                            wrapped = CompileResult(cres)
                            self.overloads[argtypes] = wrapped
                            return _result(wrapped)
                        disk_cache_miss = True

        # Cache miss - need to compile
        if disk_cache_miss:
            self._cache_misses[argtypes] += 1

        # Compile using mlir_compiler_entry which handles annotations and AST transforms
        try:
            with self._compile_profiler():
                result = mlir_compiler.mlir_compiler_entry(
                    pyfunc=self.py_func,
                    func_args=list(args),
                    targetoptions=targetoptions,
                    override_argtypes=override_argtypes,
                )
        except _RequireLaunchConfig:
            if active_launch_config is not None:
                raise RuntimeError(
                    "whole-function planner requested launch metadata after a "
                    "launch-qualified retry"
                ) from None
            available_launch_config = getattr(_compile_arg_types, "available_launch_config", None)
            if available_launch_config is None:
                raise RuntimeError(
                    "whole-function planner requires launch metadata, but compilation "
                    "was not initiated by a configured kernel launch"
                ) from None
            _normalize_available_launch_config(available_launch_config)
            self._mark_requires_launch_config()
            return self._compile_impl(args, launch_config_retry_budget)
        wrapped = CompileResult(result)
        emit_launch_config_cache_notice = False
        retry_launch_config_compile = False
        if active_launch_config_key is not None:
            normalized_args = tuple(result.signature.args)
            with self._launch_config_lock:
                if active_launch_config_generation != self._launch_config_generation:
                    retry_launch_config_compile = True
                else:
                    existing = self._find_launch_config_overload_locked(
                        argtypes, active_launch_config_key
                    )
                    if existing is None and normalized_args != argtypes:
                        existing = self._find_launch_config_overload_locked(
                            normalized_args, active_launch_config_key
                        )
                    if existing is not None:
                        # A duplicate compile can lose a race to another thread that
                        # published an equivalent overload. Discard this result
                        # without publishing callbacks from an unused module.
                        return _result(existing)
                    self._launch_config_cache_misses[(argtypes, active_launch_config_key)] += 1
                    emit_launch_config_cache_notice = not self._launch_config_cache_notice_emitted
                    self._launch_config_cache_notice_emitted = True
                    self._propagate_compile_callbacks(result.metadata)
                    self._apply_shared_memory_carveout(wrapped)
                    # Preserve both the native dispatcher request types and the
                    # compiler-normalized signature for launch-qualified lookups.
                    self._launch_config_overloads.setdefault(
                        (argtypes, active_launch_config_key), wrapped
                    )
                    self._launch_config_overloads.setdefault(
                        (normalized_args, active_launch_config_key), wrapped
                    )
            if retry_launch_config_compile:
                if launch_config_retry_budget <= 0:
                    raise RuntimeError(
                        "Kernel launch configuration was invalidated repeatedly during compile(); "
                        "retry after concurrent recompile() calls finish."
                    )
                return self._compile_impl(args, launch_config_retry_budget - 1)
            if emit_launch_config_cache_notice:
                message = (
                    "Persistent disk cache is disabled for launch-config-specialized "
                    "compiles because the disk cache key does not include launch metadata."
                )
                warnings.warn(message, NumbaPerformanceWarning, stacklevel=3)
                trace(message)
        else:
            self._propagate_compile_callbacks(result.metadata)
            self._apply_shared_memory_carveout(wrapped)
            self.overloads[result.signature.args] = wrapped

        # Save to disk cache
        if disk_cache_eligible:
            with _planner_registry._lock:
                if not _planner_registry._planners:
                    self._cache.save_overload(sig, result)

        return _result(wrapped)

    def compile(self, sig, abi_info=None, output=None):
        self._ensure_dispatcher_state()
        launch_lock = self._launch_lock
        if launch_lock is None:
            return self._compile_public(sig, abi_info=abi_info, output=output)
        with launch_lock:
            return self._compile_public(sig, abi_info=abi_info, output=output)

    def _compile_public(self, sig, abi_info=None, output=None):
        from numba_cuda_mlir.mlir_optimization import optimize
        from numba_cuda_mlir import mlir_compiler
        from numba_cuda_mlir.compiler import CompileResult

        argtypes, return_type = sigutils.normalize_signature(sig)

        if output is not None:
            self.targetoptions["_compile_output"] = output
            self.targetoptions["lto"] = output == "ltoir"

        # Check in-memory overloads first
        if argtypes in self.overloads:
            return self.overloads[argtypes]

        self._resolve_target_options()
        disk_cache_miss = False

        # Publish cache hits atomically with respect to planner registration.
        mlir_target.ensure_initialized()
        with _planner_registry._lock:
            if not _planner_registry._planners:
                cres = self._cache.load_overload(sig, mlir_target.target_context)
                if not _planner_registry._planners:
                    if cres is not None:
                        self._cache_hits[argtypes] += 1
                        wrapped = CompileResult(cres)
                        self.overloads[argtypes] = wrapped
                        return wrapped
                    disk_cache_miss = True

        # Cache miss - need to compile
        if disk_cache_miss:
            self._cache_misses[argtypes] += 1

        if abi_info is not None:
            self.targetoptions["abi_info"] = abi_info

        self._is_compiling = True
        try:
            with self._compile_profiler():
                cres = mlir_compiler.compile_mlir(
                    self.py_func,
                    return_type,
                    argtypes,
                    targetoptions=self.targetoptions,
                )
                optimize(cres)

            cres.target_context.insert_user_function(cres.entry_point, cres.fndesc, [cres.library])
        except _RequireLaunchConfig:
            raise RuntimeError(
                "whole-function planner requires launch metadata; compile through a "
                "configured kernel launch"
            ) from None
        finally:
            self._is_compiling = False

        # Propagate callbacks discovered during lowering/optimization
        self._propagate_compile_callbacks(cres.metadata)

        # Save to cache
        with _planner_registry._lock:
            if not _planner_registry._planners:
                self._cache.save_overload(sig, cres)

        # Wrap in CompileResult for compatibility attributes
        wrapped = CompileResult(cres)

        self._apply_shared_memory_carveout(wrapped)

        # Store in overloads
        self.overloads[argtypes] = wrapped

        return wrapped

    def compile_for(self, *args):
        return_type = types.none
        args = [a if isinstance(a, types.Type) else typeof(a) for a in args]
        sig = typing.signature(return_type, *args)
        return self.compile(sig)

    def _compile_device_callee(self, sig):
        """Compile enough of a device function to inline/link it into a kernel.

        Device callees are cloned from their MLIR into the parent module, so
        eagerly finalizing every callee to cubin only adds linker work and
        retained cubin metadata that the parent compilation does not use.
        """
        from numba_cuda_mlir import mlir_compiler
        from numba_cuda_mlir.compiler import CompileResult

        argtypes, return_type = sigutils.normalize_signature(sig)

        if argtypes in self.overloads:
            return self.overloads[argtypes]

        self._resolve_target_options()
        self._cache_misses[argtypes] += 1

        self._is_compiling = True
        try:
            with self._compile_profiler():
                cres = mlir_compiler.compile_mlir(
                    self.py_func,
                    return_type,
                    argtypes,
                    targetoptions=self.targetoptions,
                )

            cres.target_context.insert_user_function(cres.entry_point, cres.fndesc, [cres.library])
        except _RequireLaunchConfig:
            raise RuntimeError(
                "whole-function planner requires launch metadata; device-function "
                "compilation has no configured kernel launch"
            ) from None
        finally:
            self._is_compiling = False

        for cb in cres.metadata.get("setup_callbacks", []):
            if cb not in self._module_setup_callbacks:
                self._module_setup_callbacks.append(cb)
        for cb in cres.metadata.get("teardown_callbacks", []):
            if cb not in self._module_teardown_callbacks:
                self._module_teardown_callbacks.append(cb)

        wrapped = CompileResult(cres)
        self.overloads[argtypes] = wrapped
        return wrapped

    def _compile_as_device_callee(self, sig):
        """Compile this dispatcher through the lightweight device-callee path."""
        opts = self.targetoptions.copy()
        opts["device"] = True
        opts["lto"] = False
        if self.targetoptions.get("device", False):
            self.targetoptions.update(opts)
            return self._compile_device_callee(sig)

        if not hasattr(self, "_device_dispatcher") or self._device_dispatcher.targetoptions != opts:
            self._device_dispatcher = MLIRDispatcher(self.py_func, targetoptions=opts)
        cres = self._device_dispatcher._compile_device_callee(sig)
        argtypes, _ = sigutils.normalize_signature(sig)
        if argtypes not in self.overloads:
            self.overloads[argtypes] = cres
        return cres

    def inspect_lto_ptx(self, args=None):
        """Get generated LTO PTX.

        With no signature, launch-specialized entries are keyed by
        LaunchConfigInspectableKey and generic entries keep their argtypes tuple.
        """
        if args is None:
            return {sig: self.inspect_lto_ptx(sig) for sig in self._inspectable_overloads()}
        cres = self._find_overload(args)
        ptx = cres.metadata.get("lto_ptx")
        if ptx:
            return ptx

        from numba_cuda_mlir.mlir_optimization import get_lto_ptx

        ptx = get_lto_ptx(cres)
        cres.metadata["lto_ptx"] = ptx
        return ptx

    def forall(self, ntasks, tpb=0, stream=0, sharedmem=0):
        if ntasks < 0:
            raise ValueError("Can't create ForAll with negative task count: %s" % ntasks)
        if ntasks == 0:
            return lambda *args, **kwargs: None
        return _ForAll(self, ntasks, tpb, stream, sharedmem)

    def specialize(self, *args):
        """Create a new instance specialized for the given *args*."""
        from numba_cuda_mlir.numba_cuda import get_current_device

        cc = get_current_device().compute_capability
        argtypes = tuple(a if isinstance(a, types.Type) else typeof(a) for a in args)
        if self.specialized:
            raise RuntimeError("Dispatcher already specialized")

        specialization = self.specializations.get((cc, argtypes))
        if specialization:
            return specialization

        specialization = MLIRDispatcher(self.py_func, targetoptions=self.targetoptions)
        specialization.compile_for(*argtypes)
        specialization.disable_compile()
        specialization._specialized = True
        self.specializations[cc, argtypes] = specialization
        return specialization

    @property
    def specialized(self):
        """True if the Dispatcher has been specialized."""
        return self._specialized

    def _specialized_compile_result(self):
        if self.overloads:
            return next(iter(self.overloads.values()))
        launch_results = self._launch_config_compile_results()
        if launch_results:
            return launch_results[0]
        return None

    def _get_kernel_attr(self, attr, default, sig=None):
        """Return a kernel attribute, lazily querying the CUDA driver.

        With no signature, launch-specialized entries are keyed by
        LaunchConfigInspectableKey and generic entries keep their argtypes tuple.
        """
        if sig is not None:
            cres = self._find_overload(sig)
            cres._ensure_kernel_attrs()
            return cres.metadata.get(attr, default)
        if self.specialized:
            cres = self._specialized_compile_result()
            if cres is not None:
                cres._ensure_kernel_attrs()
                return cres.metadata.get(attr, default)
        inspectable = self._inspectable_overloads()
        return {s: self._get_kernel_attr(attr, default, s) for s in inspectable}

    def get_regs_per_thread(self, sig=None):
        return self._get_kernel_attr("regs_per_thread", 0, sig)

    def get_max_threads_per_block(self, sig=None):
        return self._get_kernel_attr("max_threads_per_block", 1024, sig)

    def get_shared_mem_per_block(self, sig=None):
        return self._get_kernel_attr("shared_mem_per_block", 0, sig)

    def get_const_mem_size(self, sig=None):
        return self._get_kernel_attr("const_mem_size", 0, sig)

    def get_local_mem_per_thread(self, sig=None):
        return self._get_kernel_attr("local_mem_per_thread", 0, sig)

    def inspect_sass(self, sig=None):
        """Return the SASS assembly for the given signature.

        Requires nvdisasm to be available on the PATH.
        With no signature, launch-specialized entries are keyed by
        LaunchConfigInspectableKey and generic entries keep their argtypes tuple.
        """
        if self.targetoptions.get("device"):
            raise RuntimeError("Cannot inspect SASS of a device function")

        if sig is not None:
            cres = self._find_overload(sig)
            return self._disassemble_cubin(cres)
        if self.specialized:
            cres = self._specialized_compile_result()
            if cres is not None:
                return self._disassemble_cubin(cres)
        inspectable = self._inspectable_overloads()
        return {s: self.inspect_sass(s) for s in inspectable}

    @staticmethod
    def _disassemble_cubin(cres):
        """Disassemble a cubin from a compile result using nvdisasm."""
        import subprocess
        import tempfile

        cubin = cres.metadata.get("cubin")
        if cubin is None:
            raise RuntimeError("No cubin available for disassembly")

        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            f.write(cubin)
            f.flush()
            try:
                cp = subprocess.run(
                    ["nvdisasm", "-gi", f.name],
                    check=True,
                    capture_output=True,
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    "nvdisasm has not been found. You may need "
                    "to install the CUDA toolkit and ensure that "
                    "it is available on your PATH.\n"
                ) from e
            return cp.stdout.decode("utf-8")

    def compile_device(self, sig):
        """Compile as a device function, injecting device=True if needed
        so that the function is not treated as a kernel."""
        opts = self.targetoptions.copy()
        opts["device"] = True
        opts["lto"] = False
        if self.targetoptions.get("device", False):
            self.targetoptions.update(opts)
            return self.compile(sig)
        if not hasattr(self, "_device_dispatcher") or self._device_dispatcher.targetoptions != opts:
            self._device_dispatcher = MLIRDispatcher(self.py_func, targetoptions=opts)
        cres = self._device_dispatcher.compile(sig)
        argtypes, _ = sigutils.normalize_signature(sig)
        if argtypes not in self.overloads:
            self.overloads[argtypes] = cres
        return cres

    def get_call_template(self, args, kws):
        """Resolve return type when this dispatcher is called from another
        jit function. Always compile as a device function so we don't
        emit kernel metadata for callees."""
        pysig, args = self._compiler.fold_argument_types(args, kws)
        kws = {}
        if self._can_compile:
            self._compile_as_device_callee(tuple(args))
        func_name = self.py_func.__name__
        name = "CallTemplate({0})".format(func_name)
        call_template = typing.make_concrete_template(
            name, key=func_name, signatures=self.nopython_signatures
        )
        return call_template, pysig, args, kws

    def recompile(self):
        """Recompile all signatures afresh.

        This clears all cached compilations so the next launch will trigger
        a fresh compile. Useful when global variables captured by the kernel
        have changed and you want the kernel to use the new values.
        """
        self._ensure_dispatcher_state()
        launch_lock = self._launch_lock
        if launch_lock is None:
            return self._recompile_impl()
        with launch_lock:
            return self._recompile_impl()

    def _recompile_impl(self):
        # Clear Python-side overloads
        self.overloads.clear()

        # Clear cache counters
        self._cache_hits.clear()
        self._cache_misses.clear()
        self._launch_config_cache_hits.clear()
        self._launch_config_cache_misses.clear()

        # Flush disk cache
        self._cache.flush()

        # Keep old dispatchers alive up to _OLD_DISPATCHER_RETAIN_LIMIT to reduce
        # CUDA cleanup crashes without allowing unbounded growth.
        with self._launch_config_lock:
            self._retain_old_dispatcher(self._c)
            self._launch_config_generation += 1
            self._launch_config_overloads.clear()
            old_launch_config_dispatchers = list(self._launch_config_dispatchers.values())
            self._launch_config_dispatchers = collections.OrderedDict()
            for dispatcher in old_launch_config_dispatchers:
                self._retain_old_dispatcher(dispatcher)
            self._configure_cache.clear()
            inflight = tuple(self._configure_cache_inflight.values())
            self._configure_cache_inflight.clear()
            for event in inflight:
                event.set()
            self._launch_config_cache_notice_emitted = False
            self._c = self._new_kernel_dispatcher()


target_registry["numba_cuda_mlir"] = MLIR
dispatcher_registry[MLIR] = MLIRDispatcher
jit_registry[MLIR] = mlir_jit
