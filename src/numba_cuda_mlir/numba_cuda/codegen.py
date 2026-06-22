# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from .cudadrv import devices, runtime
from numba_cuda_mlir.numba_cuda.core.codegen import Codegen, CodeLibrary
from numba_cuda_mlir.numba_cuda.cudadrv.linkable_code import LinkableCode
import os
import subprocess
import tempfile


def run_nvdisasm(cubin, flags):
    # nvdisasm only accepts input from a file, so we need to write out to a
    # temp file and clean up afterwards.
    fd = None
    fname = None
    try:
        fd, fname = tempfile.mkstemp()
        with open(fname, "wb") as f:
            f.write(cubin.code)

        try:
            cp = subprocess.run(
                ["nvdisasm", *flags, fname],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            msg = (
                "nvdisasm has not been found. You may need "
                "to install the CUDA toolkit and ensure that "
                "it is available on your PATH.\n"
            )
            raise RuntimeError(msg) from e
        return cp.stdout.decode("utf-8")
    finally:
        if fd is not None:
            os.close(fd)
        if fname is not None:
            os.unlink(fname)


def disassemble_cubin(cubin):
    # Request lineinfo in disassembly
    flags = ["-gi"]
    return run_nvdisasm(cubin, flags)


def disassemble_cubin_for_cfg(cubin):
    # Request control flow graph in disassembly
    flags = ["-cfg"]
    return run_nvdisasm(cubin, flags)


class ExternalCodeLibrary(CodeLibrary):
    """Holds code produced externally, for linking with generated code."""

    def __init__(self, codegen, name):
        super().__init__(codegen, name)
        # Files to link
        self._linking_files = set()
        # Setup and teardown functions for the module.
        # The order is determined by the order they are added to the codelib.
        self._setup_functions = []
        self._teardown_functions = []

        self.use_cooperative = False

    @property
    def modules(self):
        # There are no LLVM IR modules in an ExternalCodeLibrary
        return set()

    def add_linking_file(self, path_or_obj):
        # Adding new files after finalization is prohibited, in case the list
        # of libraries has already been added to another code library; the
        # newly-added files would be omitted from their linking process.
        self._raise_if_finalized()

        if isinstance(path_or_obj, LinkableCode):
            if path_or_obj.setup_callback:
                self._setup_functions.append(path_or_obj.setup_callback)
            if path_or_obj.teardown_callback:
                self._teardown_functions.append(path_or_obj.teardown_callback)

        self._linking_files.add(path_or_obj)

    def add_ir_module(self, module):
        raise NotImplementedError("Cannot add LLVM IR to external code")

    def add_linking_library(self, library):
        raise NotImplementedError("Cannot add libraries to external code")

    def finalize(self):
        self._raise_if_finalized()
        self._finalized = True

    def get_asm_str(self):
        raise NotImplementedError("No assembly for external code")

    def get_llvm_str(self):
        raise NotImplementedError("No LLVM IR for external code")

    def get_function(self, name):
        raise NotImplementedError("Cannot get function from external code")


class JITCUDACodegen(Codegen):
    """
    Scaffolding codegen retained for the device-function declaration path
    (see numba_cuda_mlir.numba_cuda.compiler). It does not generate code -
    the MLIR pipeline produces code via numba_cuda_mlir.codegen instead.
    """

    def __init__(self, module_name):
        pass

    def _create_empty_module(self, name):
        raise NotImplementedError(
            "JITCUDACodegen does not generate LLVM IR modules; the MLIR "
            "pipeline produces code via numba_cuda_mlir.codegen instead."
        )

    def _add_module(self, module):
        pass

    def magic_tuple(self):
        """
        Return a tuple unambiguously describing the codegen behaviour.
        """
        ctx = devices.get_context()
        cc = ctx.device.compute_capability
        return (runtime.runtime.get_version(), cc)
