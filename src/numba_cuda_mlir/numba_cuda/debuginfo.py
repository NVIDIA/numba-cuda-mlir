# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import abc
from contextlib import contextmanager
from enum import IntEnum

from numba_cuda_mlir.numba_cuda.core import config
from cuda.bindings import runtime


# Check if the CUDA Toolkit supports polymorphic debug info
def _check_polymorphic_debug_info_support():
    """Check if the CTK supports polymorphic debug info.

    Returns:
        tuple: (supported: bool, use_typed_const: bool)
        - supported: Whether feature is supported at all
        - use_typed_const: True for typed constant,
                           False for node reference
    """
    # runtime.getLocalRuntimeVersion() returns (cudaError_t, version_int)
    # Example: 13010 = CTK 13.1, 13020 = CTK 13.2
    _, ctk_version_number = runtime.getLocalRuntimeVersion()
    ctk_major = ctk_version_number // 1000
    ctk_minor = (ctk_version_number % 1000) // 10
    ctk_version = (ctk_major, ctk_minor)

    # Support not available with CTK 13.1 or older
    if ctk_version <= (13, 1):
        return (False, False)

    # This used to be gated on the llvmlite version (typed constant for
    # llvmlite > 0.45, node reference otherwise). The MLIR path translates
    # through a modern LLVM, which corresponds to the typed-constant format.
    use_typed_const = True
    return (True, use_typed_const)


# Check support and determine mode
(DEBUG_POLY_SUPPORTED, DEBUG_POLY_USE_TYPED_CONST) = _check_polymorphic_debug_info_support()

# Set config based on polymorphic debug info support
if not hasattr(config, "CUDA_DEBUG_POLY"):
    config.CUDA_DEBUG_POLY = DEBUG_POLY_SUPPORTED
if not hasattr(config, "CUDA_DEBUG_POLY_USE_TYPED_CONST"):
    config.CUDA_DEBUG_POLY_USE_TYPED_CONST = DEBUG_POLY_USE_TYPED_CONST


class DwarfAddressClass(IntEnum):
    GENERIC = 0x00
    GLOBAL = 0x01
    REGISTER = 0x02
    CONSTANT = 0x05
    LOCAL = 0x06
    PARAMETER = 0x07
    SHARED = 0x08


@contextmanager
def suspend_emission(builder):
    """Suspends the emission of debug_metadata for the duration of the context
    managed block."""
    ref = builder.debug_metadata
    builder.debug_metadata = None
    try:
        yield
    finally:
        builder.debug_metadata = ref


class AbstractDIBuilder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def mark_variable(
        self,
        builder,
        allocavalue,
        name,
        lltype,
        size,
        line,
        datamodel=None,
        argidx=None,
    ):
        """Emit debug info for the variable."""
        pass

    @abc.abstractmethod
    def mark_location(self, builder, line):
        """Emit source location information to the given IRBuilder."""
        pass

    @abc.abstractmethod
    def mark_subprogram(self, function, qualname, argnames, argtypes, line):
        """Emit source location information for the given function."""
        pass

    @abc.abstractmethod
    def initialize(self):
        """Initialize the debug info. An opportunity for the debuginfo to
        prepare any necessary data structures.
        """

    @abc.abstractmethod
    def finalize(self):
        """Finalize the debuginfo by emitting all necessary metadata."""
        pass


class DummyDIBuilder(AbstractDIBuilder):
    def __init__(self, module, filepath, cgctx, directives_only):
        pass

    def mark_variable(
        self,
        builder,
        allocavalue,
        name,
        lltype,
        size,
        line,
        datamodel=None,
        argidx=None,
    ):
        pass

    def mark_location(self, builder, line):
        pass

    def mark_subprogram(self, function, qualname, argnames, argtypes, line):
        pass

    def initialize(self):
        pass

    def finalize(self):
        pass


# The real DWARF DIBuilder / CUDADIBuilder built llvmlite debug-info metadata
# (DISubprogram, DILocalVariable, DWARF type mappings, ...). They are dead on the
# MLIR path: debug info is emitted by numba_cuda_mlir.mlir_debuginfo.DIBuilder
# (see MLIRLower). The names are still referenced as classes by base.py
# (BaseContext.DIBuilder) and target.py (CUDATargetContext.DIBuilder), so they
# are aliased to the no-op DummyDIBuilder.
class DIBuilder(DummyDIBuilder):
    pass


class CUDADIBuilder(DIBuilder):
    pass
