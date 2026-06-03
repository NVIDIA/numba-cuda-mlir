# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import re
from functools import cached_property
import warnings
import importlib.util
import numpy as np

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda import HAS_NUMBA
from numba_cuda_mlir.numba_cuda.core.callconv import CUDACallConv
from numba_cuda_mlir.numba_cuda.core.compiler_lock import global_compiler_lock
from numba_cuda_mlir.numba_cuda.core.errors import NumbaWarning
from numba_cuda_mlir.numba_cuda.core.base import BaseContext
from numba_cuda_mlir.numba_cuda.typing import cmathdecl
from numba_cuda_mlir.numba_cuda import datamodel

from .cudadrv import nvvm
from numba_cuda_mlir.numba_cuda import (
    cgutils,
    itanium_mangler,
    compiler,
    codegen,
    ufuncs,
    typing,
)
from numba_cuda_mlir.numba_cuda.debuginfo import CUDADIBuilder
from numba_cuda_mlir.numba_cuda.flags import CUDAFlags
from numba_cuda_mlir.numba_cuda.models import cuda_data_manager
from numba_cuda_mlir.numba_cuda.core import config, targetconfig


# -----------------------------------------------------------------------------
# Typing


class CUDATypingContext(typing.BaseContext):
    def load_additional_registries(self):
        from . import (
            cudadecl,
            cudamath,
            fp16,
            bf16,
            fp8,
            libdevicedecl,
            vector_types,
        )
        from numba_cuda_mlir.numba_cuda.typing import enumdecl, cffi_utils, npydecl
        from numba_cuda_mlir.extending import (
            typing_registry as extending_typing_registry,
        )

        self.install_registry(cudadecl.registry)
        self.install_registry(cffi_utils.registry)
        self.install_registry(cudamath.registry)
        self.install_registry(cmathdecl.registry)
        self.install_registry(libdevicedecl.registry)
        self.install_registry(npydecl.registry)
        self.install_registry(enumdecl.registry)
        self.install_registry(vector_types.typing_registry)
        self.install_registry(fp16.typing_registry)
        self.install_registry(bf16.typing_registry)
        self.install_registry(fp8.typing_registry)
        self.install_registry(extending_typing_registry)

    def resolve_value_type(self, val):
        # treat other dispatcher object as another device function
        from numba_cuda_mlir.numba_cuda.dispatcher import CUDADispatcher
        from numba_cuda_mlir.numba_cuda.dispatcher import Dispatcher

        if HAS_NUMBA:
            if isinstance(val, Dispatcher) and not isinstance(val, CUDADispatcher):
                try:
                    # use cached device function
                    val = val.__dispatcher
                except AttributeError:
                    if not val._can_compile:
                        raise ValueError(
                            "using cpu function on device but its compilation is disabled"
                        )
                    targetoptions = val.targetoptions.copy()
                    targetoptions["device"] = True
                    targetoptions["debug"] = targetoptions.get("debug", False)
                    targetoptions["opt"] = targetoptions.get("opt", True)
                    disp = CUDADispatcher(val.py_func, targetoptions)
                    # cache the device function for future use and to avoid
                    # duplicated copy of the same function.
                    val.__dispatcher = disp
                    val = disp

        # continue with parent logic
        return super().resolve_value_type(val)

    def can_convert(self, fromty, toty):
        """
        Check whether conversion is possible from *fromty* to *toty*.
        If successful, return a numba.cuda.typeconv.Conversion instance;
        otherwise None is returned.
        """

        # This implementation works around the issue addressed in Numba PR
        # #10047, "Fix IntEnumMember.can_convert_to() when no conversions
        # found", https://github.com/numba/numba/pull/10047.
        #
        # This should be gated on the version of Numba that the fix is
        # incorporated into, and eventually removed when the minimum supported
        # Numba version includes the fix.

        try:
            return super().can_convert(fromty, toty)
        except TypeError:
            if isinstance(fromty, types.IntEnumMember):
                # IntEnumMember fails to correctly handle impossible
                # conversions - in this scenario the correct thing to do is to
                # return None to signal that the conversion was not possible
                return None
            else:
                # Any failure involving conversion from a non-IntEnumMember is
                # almost certainly a real and separate issue
                raise


# -----------------------------------------------------------------------------
# Implementation


VALID_CHARS = re.compile(r"[^a-z0-9]", re.I)


def load_cuda_target_registration_modules():
    from numba_cuda_mlir.numba_cuda.cpython import (
        numbers,
        slicing,
        iterators,
        listobj,
        unicode,
        charseq,
        cmathimpl,
        mathimpl,
        tupleobj,
        rangeobj,
        enumimpl,
    )
    from numba_cuda_mlir.numba_cuda.cpython import builtins as cpython_builtins
    from numba_cuda_mlir.numba_cuda.core import optional
    from numba_cuda_mlir.numba_cuda.misc import cffiimpl
    from numba_cuda_mlir.numba_cuda.np import (
        arrayobj,
        npdatetime,
        polynomial,
        arraymath,
    )
    from numba_cuda_mlir.numba_cuda.np.unsafe import ndarray

    return (
        numbers,
        slicing,
        iterators,
        listobj,
        unicode,
        charseq,
        cmathimpl,
        mathimpl,
        tupleobj,
        rangeobj,
        enumimpl,
        cpython_builtins,
        optional,
        cffiimpl,
        arrayobj,
        npdatetime,
        polynomial,
        arraymath,
        ndarray,
    )


class CUDATargetContext(BaseContext):
    implement_powi_as_math_call = True
    strict_alignment = True

    def __init__(self, typingctx, target="cuda"):
        super().__init__(typingctx, target)
        self.data_model_manager = cuda_data_manager.chain(datamodel.default_manager)

    @property
    def enable_nrt(self):
        return getattr(config, "CUDA_ENABLE_NRT", False)

    @property
    def DIBuilder(self):
        return CUDADIBuilder

    @property
    def enable_boundscheck(self):
        # Unconditionally disabled
        return False

    # Overrides
    def create_module(self, name):
        return self._internal_codegen._create_empty_module(name)

    def init(self):
        self._internal_codegen = codegen.JITCUDACodegen("numba.cuda.jit")
        self._target_data = None

    def load_additional_registries(self):
        (
            numbers,
            slicing,
            iterators,
            listobj,
            unicode,
            charseq,
            cmathimpl,
            mathimpl,
            tupleobj,
            rangeobj,
            enumimpl,
            cpython_builtins,
            optional,
            cffiimpl,
            arrayobj,
            npdatetime,
            polynomial,
            arraymath,
            _,
        ) = load_cuda_target_registration_modules()

        from . import (
            cudaimpl,
            fp16,
            mathimpl as cuda_mathimpl,
            vector_types,
            bf16,
            fp8,
        )

        self.install_registry(cudaimpl.registry)
        self.install_registry(cffiimpl.registry)
        self.install_registry(cmathimpl.registry)
        self.install_registry(mathimpl.registry)
        self.install_registry(numbers.registry)
        self.install_registry(optional.registry)
        self.install_registry(cuda_mathimpl.registry)
        self.install_registry(vector_types.impl_registry)
        self.install_registry(fp16.target_registry)
        self.install_registry(bf16.target_registry)
        self.install_registry(fp8.target_registry)
        self.install_registry(slicing.registry)
        self.install_registry(iterators.registry)
        self.install_registry(listobj.registry)
        self.install_registry(unicode.registry)
        self.install_registry(charseq.registry)
        self.install_registry(tupleobj.registry)
        self.install_registry(rangeobj.registry)
        self.install_registry(enumimpl.registry)
        self.install_registry(cpython_builtins.registry)

        # install np registries
        self.install_registry(polynomial.registry)
        self.install_registry(npdatetime.registry)
        self.install_registry(arrayobj.registry)
        self.install_registry(arraymath.registry)

    def codegen(self):
        return self._internal_codegen

    @property
    def target_data(self):
        # The MLIR pipeline does not use an llvmlite TargetData; this property
        # is only reachable on the dead llvmlite codegen path.
        raise NotImplementedError("CUDATargetContext.target_data is not available on the MLIR path")

    def build_list(self, builder, list_type, items):
        """
        Build a list from the Numba *list_type* and its initial *items*.
        """
        from numba_cuda_mlir.numba_cuda.cpython import listobj

        return listobj.build_list(self, builder, list_type, items)

    @cached_property
    def nonconst_module_attrs(self):
        """
        Some CUDA intrinsics are at the module level, but cannot be treated as
        constants, because they are loaded from a special register in the PTX.
        These include threadIdx, blockDim, etc.
        """
        from numba_cuda_mlir import numba_cuda as cuda

        nonconsts = (
            "threadIdx",
            "blockDim",
            "blockIdx",
            "gridDim",
            "laneid",
            "warpsize",
        )
        nonconsts_with_mod = tuple([(types.Module(cuda), nc) for nc in nonconsts])
        return nonconsts_with_mod

    @property
    def call_conv(self):
        warnings.warn(
            "Context.call_conv is deprecated. Use FunctionDescriptor.call_conv instead.",
            DeprecationWarning,
        )
        return self.fndesc.call_conv

    def make_constant_array(self, builder, aryty, arr):
        # Builds an llvmlite constant array global; only reachable on the dead
        # llvmlite codegen path. The MLIR pipeline materializes constant arrays
        # itself, so this is never called.
        raise NotImplementedError(
            "CUDATargetContext.make_constant_array is not available on the MLIR path"
        )

    def insert_const_string(self, mod, string):
        # Builds an llvmlite constant string global; only reachable on the dead
        # llvmlite codegen path.
        raise NotImplementedError(
            "CUDATargetContext.insert_const_string is not available on the MLIR path"
        )

    def insert_string_const_addrspace(self, builder, string):
        # Builds an llvmlite constant string global; only reachable on the dead
        # llvmlite codegen path.
        raise NotImplementedError(
            "CUDATargetContext.insert_string_const_addrspace is not available on the MLIR path"
        )

    def optimize_function(self, func):
        """Run O1 function passes"""
        pass
        ## XXX skipped for now
        # fpm = lp.FunctionPassManager.new(func.module)
        #
        # lp.PassManagerBuilder.new().populate(fpm)
        #
        # fpm.initialize()
        # fpm.run(func)
        # fpm.finalize()

    def get_ufunc_info(self, ufunc_key):
        return ufuncs.get_ufunc_info(ufunc_key)

    def _compile_subroutine_no_cache(self, builder, impl, sig, locals=None, flags=None):
        # Overrides numba.core.base.BaseContext._compile_subroutine_no_cache().
        # Modified to use flags from the context stack if they are not provided
        # (pending a fix in Numba upstream).

        if locals is None:
            locals = {}

        with global_compiler_lock:
            codegen = self.codegen()
            library = codegen.create_library(impl.__name__)
            if flags is None:
                cstk = targetconfig.ConfigStack()
                if cstk:
                    flags = cstk.top().copy()
                else:
                    msg = "There should always be a context stack; none found."
                    warnings.warn(msg, NumbaWarning)
                    flags = CUDAFlags()

            flags.no_compile = True
            flags.no_cpython_wrapper = True
            flags.no_cfunc_wrapper = True

            # compile_subroutine always uses CUDACallConv
            call_conv = CUDACallConv(self)
            abi_info = {}

            cres = compiler.compile_internal(
                self.typing_context,
                self,
                library,
                impl,
                sig.args,
                sig.return_type,
                flags,
                locals=locals,
                call_conv=call_conv,
                abi_info=abi_info,
            )

            # Allow inlining the function inside callers
            self.active_code_library.add_linking_library(cres.library)
            return cres
