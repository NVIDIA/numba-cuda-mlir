# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import collections
from contextlib import contextmanager
from functools import cached_property, lru_cache
from typing import override
from cusimt.errors import UserFacingInternalCompilerError
from cusimt.logging import trace
import inspect

from cusimt import codegen
from cusimt.decorators import mlir_jit
from cusimt.numba_cuda.errors import normalize_kernel_dimensions
from cusimt.numba_cuda.core.errors import NumbaPerformanceWarning
from cusimt.numba_cuda.core import config as cuda_config
from cusimt.numba_cuda.cudadrv import driver as numba_cuda_driver
from importlib.util import find_spec
from cusimt.numba_cuda.core import errors, sigutils
from cusimt.numba_cuda import types
from cusimt.numba_cuda.typing.typeof import typeof
from cusimt.numba_cuda.cudadecl import registry as cuda_registry
from cusimt.numba_cuda import serialize, typing
from cusimt.numba_cuda.core.base import BaseContext
from cusimt.numba_cuda.core.callconv import MinimalCallConv
from cusimt.numba_cuda.core.descriptors import TargetDescriptor
from cusimt.numba_cuda.core.compiler_lock import global_compiler_lock
from cusimt.numba_cuda.dispatcher import Dispatcher
from cusimt.numba_cuda.core.options import TargetOptions
from cusimt.numba_cuda.core.target_extension import (
    CPU,
    target_registry,
    dispatcher_registry,
    jit_registry,
)
from cusimt.type_defs.builtin_types import Namespace
from cusimt.caching import MLIRCache, NullCache
import logging
import warnings
import numpy as np
from cusimt.errors import ExtensionError
from cusimt import _cext
from cusimt._cext import LaunchConfiguration
import threading
import cProfile
import pstats
import sys

# Thread-local storage for passing original tuple arg types to _compile
_compile_arg_types = threading.local()


def _is_strided_memory_view(arg):
    """Check if arg is a cuda.core StridedMemoryView."""
    try:
        from cuda.core._memoryview import StridedMemoryView

        return isinstance(arg, StridedMemoryView)
    except ImportError:
        return False


def _strided_memory_view_to_device_array(smv):
    """Convert a StridedMemoryView to a DeviceNDArray for kernel dispatch."""
    from cusimt.numba_cuda.cudadrv import driver, devices
    from cusimt.numba_cuda.cudadrv.devicearray import DeviceNDArray

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


def _call_module_teardown_callback(cb, object_code):
    try:
        cb(object_code)
    except Exception:
        logging.exception("teardown callback %r failed for %r", cb, object_code)


class _ModuleCallbackRef:
    """Prevent GC of ObjectCode and invoke teardown callbacks when this ref is collected."""

    __slots__ = ("_teardown_cbs", "_object_code")

    def __init__(self, teardown_cbs, object_code):
        self._teardown_cbs = list(teardown_cbs)
        self._object_code = object_code

    def __del__(self):
        for cb in self._teardown_cbs:
            _call_module_teardown_callback(cb, self._object_code)
        self._teardown_cbs = []
        self._object_code = None


def _safe_typeof(arg):
    try:
        return typeof(arg)
    except (ValueError, TypeError):
        return None


def _flatten_arg(arg):
    """Recursively flatten a tuple/list argument."""
    if isinstance(arg, (tuple, list)):
        result = []
        for elem in arg:
            result.extend(_flatten_arg(elem))
        return result
    return [arg]


def _flatten_args(args):
    """Flatten all tuple/list arguments in args."""
    result = []
    for arg in args:
        result.extend(_flatten_arg(arg))
    return result


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
    from cusimt.numba_cuda.cudadrv import devices

    return devices.get_context()


class _ArgMarshaller:
    """
    Handles copying numpy arrays to the device and back to the host
    before and after the kernel is launched.
    NOTE: This adds overhead to the kernel launch!! This is not good!
    But numba-cuda supports it, so we must, for now.
    """

    def __init__(self, launcher, extensions=None, dispatcher=None):
        self._launcher = launcher
        self._callbacks = []
        self._extensions = extensions or []
        self._dispatcher = dispatcher

    def _maybe_copy_to_device_item(self, arg):
        from cusimt import cuda

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
                return int(arg)
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
                from cusimt.numba_cuda.args import In

                host_val = arg.value
                device_arg = cuda.to_device(host_val)
                if not isinstance(arg, In):

                    def copy_back(host_arg=host_val, dev_arg=device_arg):
                        host_array = dev_arg.copy_to_host()
                        np.copyto(host_arg, host_array)

                    self._callbacks.append(copy_back)
                return device_arg
            case _ if hasattr(arg, "__cuda_array_interface__") and not isinstance(arg, np.ndarray):
                # Third-party objects implementing __cuda_array_interface__.
                # Convert to DeviceNDArray so typeof() works.
                from cusimt.numba_cuda.cudadrv.devicearray import DeviceNDArrayBase

                if isinstance(arg, DeviceNDArrayBase):
                    return arg
                from cusimt.numba_cuda.api import from_cuda_array_interface

                return from_cuda_array_interface(arg.__cuda_array_interface__, owner=arg)
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

    def _maybe_copy_to_device(self, args):
        return list(map(self._maybe_copy_to_device_item, args))

    def _apply_extensions(self, args):
        if not self._extensions:
            return list(args)

        transformed_vals = []
        for arg in args:
            try:
                ty = typeof(arg)
            except (ValueError, TypeError) as e:
                raise ExtensionError(
                    f"Could not get type of argument: {arg}. Please register a typeof_impl for this type."
                ) from e
            val = arg
            for extension in reversed(self._extensions):
                ty, val = extension.prepare_args(ty, val, stream=None, retr=self._callbacks)
            transformed_vals.append(val)
        return transformed_vals

    def _coerce_to_overload(self, device_args, argtypes):
        """Coerce scalar arguments to match a pre-compiled overload signature."""
        disp = self._dispatcher
        if disp is None or disp._can_compile or not disp.overloads:
            return device_args, argtypes

        for sig_args in disp.overloads:
            if len(sig_args) == len(argtypes) and all(
                disp._can_reuse_overload(s, r) for s, r in zip(sig_args, argtypes)
            ):
                return device_args, argtypes

        best, has_match = disp._resolve_overload(argtypes, allow_unsafe=True)
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

    def __call__(self, *args):
        original_types = [_safe_typeof(arg) for arg in args]

        args = self._apply_extensions(args)

        device_args = self._maybe_copy_to_device(args)
        # Store original types for _compile to use (before flattening)
        argtypes = [typeof(arg) for arg in device_args]
        device_args, argtypes = self._coerce_to_overload(device_args, argtypes)
        _compile_arg_types.types = argtypes
        try:
            # Flatten tuple/list arguments for C++ kernel launcher
            flat_args = _flatten_args(device_args)
            result = self._launcher(*flat_args)
        except UserFacingInternalCompilerError as e:
            import os

            if os.environ.get("CUSIMT_ICE_FULL_TB", "0").strip() == "1":
                raise
            raise e.with_traceback(None) from None
        except RuntimeError as e:
            _raise_as_cuda_error(e)
        finally:
            _compile_arg_types.types = None
        for callback in self._callbacks:
            callback()
        self._callbacks.clear()  # Clear callbacks to prevent accumulation
        return result


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

            cres = next(iter(self.dispatcher.overloads.values()))
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
        from cusimt.typing.numpy import registry as npydecl_registry
        from cusimt.typing.libdevice import registry as libdevice_registry
        from cusimt.typing.externals import registry as externals_registry
        from cusimt.typing.ctypes import registry as ctypes_registry
        from cusimt.typing.cuda import registry as cusimt_cuda_registry
        from cusimt.typing.builtin import registry as cusimt_builtins_registry
        from cusimt.typing.struct import registry as struct_registry
        from cusimt.typing.cmath import registry as cmath_registry
        from cusimt.typing.math import registry as math_registry
        from cusimt.typing.half_precision import (
            registry as half_precision_registry,
            register_bf16_globals,
        )
        from cusimt.typing.exotic_float import register_fp8_globals
        from cusimt.typing.exotic_float import (
            registry as exotic_float_typing_registry,
        )
        from cusimt.typing.vector import registry as vector_registry
        from cusimt.typing.cuda_vector_types import (
            registry as cuda_vector_types_registry,
        )
        from cusimt.extending import typing_registry as extending_typing_registry
        import cusimt.lowering.unicode  # noqa: F401 - registers string overloads
        from cusimt.numba_cuda.typing import enumdecl, cffi_utils
        from cusimt.numba_cuda.typing.templates import builtin_registry

        # Install cusimt registries first to give them priority
        self.install_registry(ctypes_registry)
        self.install_registry(libdevice_registry)
        self.install_registry(npydecl_registry)
        self.install_registry(cusimt_cuda_registry)
        self.install_registry(externals_registry)
        self.install_registry(cusimt_builtins_registry)
        self.install_registry(struct_registry)
        self.install_registry(cmath_registry)
        self.install_registry(math_registry)
        self.install_registry(half_precision_registry)
        self.install_registry(exotic_float_typing_registry)
        self.install_registry(vector_registry)
        self.install_registry(cuda_vector_types_registry)
        self.install_registry(extending_typing_registry)

        # Install numba-cuda registries after cusimt ones
        self.install_registry(cuda_registry)
        self.install_registry(builtin_registry)
        register_bf16_globals()
        register_fp8_globals()

        # Install numba-cuda's bf16 typing registry (includes operators)
        from cusimt.numba_cuda._internal.cuda_bf16 import (
            typing_registry as bf16_typing_registry,
        )

        self.install_registry(bf16_typing_registry)

        self.install_registry(enumdecl.registry)
        self.install_registry(cffi_utils.registry)
        if find_spec("torch") is not None:
            from cusimt.type_defs.torch_types import registry as torch_registry

            self.install_registry(torch_registry)

        if find_spec("cupy") is not None:
            import cusimt.type_defs.cupy_types  # noqa: F401

    def refresh(self):
        super().refresh()
        self._filter_conflicting_overload_methods()

    def _filter_conflicting_overload_methods(self):
        """Remove upstream overload_method templates that conflict with
        cusimt's own Array method typing (min, max, etc.).  Upstream
        numba-cuda registers @overload_method(types.Array, "min") / "max"
        whose bodies use numpy_take which cannot handle 0-d arrays under
        the cusimt pipeline."""
        from cusimt.numba_cuda import types as nb_types

        cusimt_methods = {"min", "max", "sum", "prod", "mean", "std", "var"}
        arr_templates = self._attributes.get(nb_types.Array, [])
        self._attributes[nb_types.Array] = [
            t
            for t in arr_templates
            if not (
                hasattr(type(t), "_attr")
                and getattr(type(t), "_attr", None) in cusimt_methods
                and getattr(type(t), "__module__", "").startswith("cusimt.numba_cuda")
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
        from cusimt.numba_cuda.typing.typeof import typeof as cuda_typeof

        try:
            return cuda_typeof(val)
        except ValueError:
            return super().resolve_value_type(val)


class MLIRCallConv(MinimalCallConv):
    """Use simple default call convention for now"""


# The MLIRTargetContext allows us to use the data models that we registered.
class MLIRTargetContext(BaseContext):
    strict_alignment = True

    def __init__(self, typingctx, target="cusimt"):
        super().__init__(typingctx, target)
        from cusimt.models import mlir_data_manager

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
            "cusimt.cuda",
            "cusimt.numba_cuda",
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
        from cusimt.install_registry import setup_lowering_patches

        setup_lowering_patches()

        from cusimt.models import register_fp8_models

        register_fp8_models()

        # Import individual lowering registries
        from cusimt.lowering.builtins import registry as builtins_lowering_registry
        from cusimt.lowering.math import registry as math_lowering_registry
        from cusimt.lowering.cmath import registry as cmath_lowering_registry
        from cusimt.lowering.numpy import registry as numpy_lowering_registry
        from cusimt.lowering.libdevice import registry as libdevice_lowering_registry
        from cusimt.lowering.cuda import registry as cuda_lowering_registry
        from cusimt.lowering.print import registry as print_lowering_registry
        from cusimt.lowering.ctypes import registry as ctypes_lowering_registry
        from cusimt.lowering.struct import registry as struct_lowering_registry
        from cusimt.lowering.union import registry as union_lowering_registry
        from cusimt.lowering.half_precision import (
            registry as half_precision_lowering_registry,
        )
        from cusimt.lowering.exotic_float import (
            registry as exotic_float_lowering_registry,
        )
        from cusimt.lowering.vector import registry as vector_lowering_registry
        from cusimt.lowering.cuda_vector_types import (
            registry as cuda_vector_types_lowering_registry,
        )
        from cusimt.lowering.mlir.types import registry as mlir_types_lowering_registry
        from cusimt.lowering.record import registry as record_lowering_registry
        from cusimt.lowering.unicode import registry as unicode_lowering_registry
        from cusimt.lowering.nrt import registry as nrt_lowering_registry
        from cusimt.extending import lowering_registry as extending_lowering_registry
        from cusimt.lowering.enum import registry as enum_lowering_registry
        from cusimt.lowering.datetime import registry as datetime_lowering_registry

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
        self.install_registry(mlir_types_lowering_registry)
        self.install_registry(enum_lowering_registry)
        self.install_registry(unicode_lowering_registry)
        self.install_registry(nrt_lowering_registry)
        self.install_registry(datetime_lowering_registry)
        self.install_registry(extending_lowering_registry)

    @override
    def refresh(self):
        self.load_additional_registries()
        super().refresh()
        self.typing_context.refresh()

    def get_overload_builder(self, fn, sig):
        """Return an MLIR builder for an overloaded function, or None.

        Searches the typing templates for an overload Dispatcher that
        can be compiled through cusimt's MLIR pipeline. For BoundFunction
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
                if args == match_args:
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
            "cusimt.cuda",
            "cusimt.numba_cuda",
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
                    from cusimt.lowering_utilities import convert

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


mlir_target = MLIRTarget("cusimt")


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
    "_CompileStats", ("cache_path", "cache_hits", "cache_misses")
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
        if targetoptions is None:
            targetoptions = {}
        # AST transforms now happen at compile time (in compile_mlir) when we
        # have the signature (argtypes) available. This allows consteval to
        # access argument types and target options.
        super().__init__(py_func, targetoptions=targetoptions)
        self._c = _cext.KernelDispatcher(
            self._compile,
            get_constant_args(py_func),
            _ensure_numba_cuda_context,
        )
        self.extensions = targetoptions.get("extensions") or []
        self._specialized = False
        self.specializations = {}
        self._module_setup_callbacks = []
        self._module_teardown_callbacks = []
        for link_item in targetoptions.get("link", []):
            if hasattr(link_item, "setup_callback") and link_item.setup_callback:
                self._module_setup_callbacks.append(link_item.setup_callback)
            if hasattr(link_item, "teardown_callback") and link_item.teardown_callback:
                self._module_teardown_callbacks.append(link_item.teardown_callback)

        # Caching support
        self._cache = NullCache()
        self._cache_hits = collections.Counter()
        self._cache_misses = collections.Counter()

        # Checked by type inferer (numba-cuda's typeinfer.py) to detect self-recursive calls
        self._is_compiling = False

    @property
    def is_compiling(self):
        return self._is_compiling

    @property
    def _numba_type_(self):
        return MLIRDispatcherType(self)

    @lru_cache(maxsize=128)
    def configure(self, griddim, blockdim, stream=None, sharedmem=None, cluster=None):
        griddim, blockdim = normalize_kernel_dimensions(griddim, blockdim)
        if cluster is not None:
            cluster = normalize_kernel_dimensions(cluster, (1, 1, 1))[0]

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

        # Numba-CUDA now holds a reference to a cuda-bindings CUstream instead
        # of a ctypes object. The easiest way to pass it to cext is as an
        # integer.
        if isinstance(stream, numba_cuda_driver.Stream):
            stream = int(stream.handle)

        return _ArgMarshaller(
            LaunchConfiguration(self._c, griddim, blockdim, stream, sharedmem, cluster),
            extensions=self.extensions,
            dispatcher=self,
        )

    def __getitem__(self, args):
        assert isinstance(args, tuple)
        assert len(args) in (2, 3, 4, 5)
        return self.configure(*args)

    def enable_caching(self):
        """Enable on-disk caching for this dispatcher."""
        self._cache = MLIRCache(self.py_func)

    @property
    def stats(self):
        """Return cache statistics."""
        return _CompileStats(
            cache_path=self._cache.cache_path,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
        )

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

        Returns the compile result if found, or compiles and returns it.
        """
        # Direct lookup
        if sig in self.overloads:
            return self.overloads[sig]

        # Extract args to search for matching overload
        search_args = sig.args if hasattr(sig, "args") else sig if isinstance(sig, tuple) else None
        if search_args is not None:
            for key, cres in self.overloads.items():
                key_args = (
                    key.args if hasattr(key, "args") else key if isinstance(key, tuple) else None
                )
                if key_args == search_args:
                    return cres

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

    def inspect_llvm(self, sig=None):
        raise NotImplementedError(
            "inspect_llvm is not supported. "
            "Use inspect_mlir() to inspect the MLIR module or inspect_asm() to inspect the PTX."
        )

    def inspect_asm(self, signature=None):
        """Get the generated PTX assembly code."""
        if signature is None:
            return {sig: self.inspect_asm(sig) for sig in self.overloads}
        cres = self._find_overload(signature)
        return cres.metadata["ptx"]

    def inspect_mlir(self, sig=None):
        if sig is None:
            return {sig: self.inspect_mlir(sig) for sig in self.overloads}
        cres = self._find_overload(sig)
        return cres.metadata["mlir_module_str"]

    def inspect_mlir_optimized(self, sig=None):
        if sig is None:
            return {sig: self.inspect_mlir_optimized(sig) for sig in self.overloads}
        cres = self._find_overload(sig)
        return cres.metadata["mlir_module_optimized"]

    def inspect_transformed_source(self, sig=None):
        """Return the AST-transformed source code for the given signature.

        Args:
            sig: The signature to get the transformed source for. If None, returns
                a dict mapping all compiled signatures to their transformed sources.

        Returns:
            The transformed source string, or None if no transforms were applied.
            If sig is None, returns a dict of {signature: transformed_source}.
        """
        if sig is None:
            return {sig: self.inspect_transformed_source(sig) for sig in self.overloads}
        cres = self._find_overload(sig)
        return cres.metadata.get("transformed_source")

    def inspect_ptx(self, sig=None):
        if sig is None:
            return {sig: self.inspect_ptx(sig) for sig in self.overloads}
        cres = self._find_overload(sig)
        return cres.metadata["ptx"]

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

    def _resolve_overload(self, argtypes, allow_unsafe=True):
        """Find the best-matching overload(s) for argtypes.
        Returns (tied_best_sig_args_list, has_any_match)."""
        candidates = []
        for sig_args in self.overloads:
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

    def _raise_ambiguous(self, argtypes, tied):
        sigs_str = "\n".join(f"{sig_args} -> none" for sig_args in tied)
        raise TypeError(f"Ambiguous overloading for {self.py_func!r} {argtypes}:\n{sigs_str}")

    def _make_post_load_hook(self):
        """Create a callback for C++ to invoke after loading the CUlibrary."""
        setup_cbs = self._module_setup_callbacks
        teardown_cbs = self._module_teardown_callbacks

        def post_load(lib_handle_int):
            from cuda.bindings.driver import CUlibrary
            from cusimt.numba_cuda.cudadrv import devices

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

    def _compile_impl(self, args):
        from cusimt import mlir_compiler
        from cusimt.compiler import CompileResult

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

        def _result(cres):
            return (
                cres.metadata["cubin"],
                cres.metadata["func_name"],
                cres.metadata.get("use_cooperative", False),
            )

        # Try to reuse a pre-compiled overload with ABI-compatible types
        if self.overloads:
            for sig_args, cres in self.overloads.items():
                if len(sig_args) == len(argtypes) and all(
                    self._can_reuse_overload(s, r) for s, r in zip(sig_args, argtypes)
                ):
                    return _result(cres)

        if not self._can_compile:
            best, has_any = self._resolve_overload(argtypes, allow_unsafe=True)
            if not has_any:
                raise TypeError(f"No matching definition for argument type(s) {argtypes}")
            if len(best) > 1:
                self._raise_ambiguous(argtypes, best)
            cres = self.overloads[best[0]]
            return _result(cres)

        # Try to load from disk cache
        sig = typing.signature(types.none, *argtypes)
        cres = self._cache.load_overload(sig, mlir_target.target_context)
        if cres is not None:
            self._cache_hits[argtypes] += 1
            wrapped = CompileResult(cres)
            self.overloads[argtypes] = wrapped
            return _result(wrapped)

        # Cache miss - need to compile
        self._cache_misses[argtypes] += 1

        # Compile using mlir_compiler_entry which handles annotations and AST transforms
        with self._compile_profiler():
            result = mlir_compiler.mlir_compiler_entry(
                pyfunc=self.py_func,
                func_args=list(args),
                targetoptions=self.targetoptions,
                override_argtypes=override_argtypes,
            )
        wrapped = CompileResult(result)
        self.overloads[result.signature.args] = wrapped

        # Propagate callbacks discovered during lowering (e.g. from
        # declare_device(link=[...])) up to the dispatcher level.
        for cb in result.metadata.get("setup_callbacks", []):
            if cb not in self._module_setup_callbacks:
                self._module_setup_callbacks.append(cb)
        for cb in result.metadata.get("teardown_callbacks", []):
            if cb not in self._module_teardown_callbacks:
                self._module_teardown_callbacks.append(cb)

        # Save to disk cache
        self._cache.save_overload(sig, result)

        return _result(wrapped)

    def compile(self, sig, abi_info=None, output=None):
        from cusimt.mlir_optimization import optimize
        from cusimt import mlir_compiler
        from cusimt.compiler import CompileResult

        argtypes, return_type = sigutils.normalize_signature(sig)

        # Check in-memory overloads first
        if argtypes in self.overloads:
            return self.overloads[argtypes]

        # Try to load from disk cache
        cres = self._cache.load_overload(sig, mlir_target.target_context)
        if cres is not None:
            self._cache_hits[argtypes] += 1
            wrapped = CompileResult(cres)
            self.overloads[argtypes] = wrapped
            return wrapped

        # Cache miss - need to compile
        self._cache_misses[argtypes] += 1

        if output is not None:
            self.targetoptions["output"] = output
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
        finally:
            self._is_compiling = False

        # Save to cache
        self._cache.save_overload(sig, cres)

        # Wrap in CompileResult for compatibility attributes
        wrapped = CompileResult(cres)

        # Apply shared memory carveout if specified
        carveout = self.targetoptions.get("shared_memory_carveout")
        if carveout is not None:
            if isinstance(carveout, str):
                carveout_map = {"default": -1, "maxl1": 0, "maxshared": 100}
                carveout = carveout_map[carveout.lower()]
            wrapped._codelibrary.get_cufunc().set_shared_memory_carveout(carveout)

        # Store in overloads
        self.overloads[argtypes] = wrapped

        return wrapped

    def compile_for(self, *args):
        return_type = types.none
        args = [a if isinstance(a, types.Type) else typeof(a) for a in args]
        sig = typing.signature(return_type, *args)
        return self.compile(sig)

    def inspect_lto_ptx(self, args=None):
        return self.inspect_ptx(args)

    def forall(self, ntasks, tpb=0, stream=0, sharedmem=0):
        if ntasks < 0:
            raise ValueError("Can't create ForAll with negative task count: %s" % ntasks)
        if ntasks == 0:
            return lambda *args, **kwargs: None
        return _ForAll(self, ntasks, tpb, stream, sharedmem)

    def specialize(self, *args):
        """Create a new instance specialized for the given *args*."""
        from cusimt.numba_cuda import get_current_device

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

    def _get_kernel_attr(self, attr, default, sig=None):
        """Return a kernel attribute, lazily querying the CUDA driver."""
        if sig is not None:
            cres = self._find_overload(sig)
            cres._ensure_kernel_attrs()
            return cres.metadata.get(attr, default)
        if self.specialized:
            cres = next(iter(self.overloads.values()))
            cres._ensure_kernel_attrs()
            return cres.metadata.get(attr, default)
        return {s: self._get_kernel_attr(attr, default, s) for s in self.overloads}

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
        """
        if self.targetoptions.get("device"):
            raise RuntimeError("Cannot inspect SASS of a device function")

        if sig is not None:
            cres = self._find_overload(sig)
            return self._disassemble_cubin(cres)
        if self.specialized:
            cres = next(iter(self.overloads.values()))
            return self._disassemble_cubin(cres)
        return {s: self.inspect_sass(s) for s in self.overloads}

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
        if not self.targetoptions.get("device", False):
            if not hasattr(self, "_device_dispatcher"):
                opts = self.targetoptions.copy()
                opts["device"] = True
                self._device_dispatcher = MLIRDispatcher(self.py_func, targetoptions=opts)
            cres = self._device_dispatcher.compile(sig)
            argtypes, _ = sigutils.normalize_signature(sig)
            if argtypes not in self.overloads:
                self.overloads[argtypes] = cres
            return cres
        return self.compile(sig)

    def get_call_template(self, args, kws):
        """Resolve return type when this dispatcher is called from another
        jit function. Always compile as a device function so we don't
        emit kernel metadata for callees."""
        pysig, args = self._compiler.fold_argument_types(args, kws)
        kws = {}
        if self._can_compile:
            self.compile_device(tuple(args))
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
        # Clear Python-side overloads
        self.overloads.clear()

        # Clear cache counters
        self._cache_hits.clear()
        self._cache_misses.clear()

        # Flush disk cache
        self._cache.flush()

        # Clear the configure LRU cache so old LaunchConfiguration refs are dropped
        self.configure.cache_clear()

        # Keep old dispatcher alive to prevent CUDA resource cleanup issues,
        # then create a new one. This is a memory leak but avoids crashes.
        if not hasattr(self, "_old_dispatchers"):
            self._old_dispatchers = []
        self._old_dispatchers.append(self._c)
        self._c = _cext.KernelDispatcher(
            self._compile,
            get_constant_args(self.py_func),
            _ensure_numba_cuda_context,
        )


target_registry["cusimt"] = MLIR
dispatcher_registry[MLIR] = MLIRDispatcher
jit_registry[MLIR] = mlir_jit
