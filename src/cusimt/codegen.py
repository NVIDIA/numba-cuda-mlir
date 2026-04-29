# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt.numba_cuda import serialize
from cusimt.numba_cuda.codegen import Codegen, CodeLibrary
from cusimt._mlir.ir import Context, Location, Module


class MLIRCodeLibrary(serialize.ReduceMixin, CodeLibrary):
    @classmethod
    def _rebuild(cls, codegen, name):
        return cls(codegen, name)

    def _reduce_states(self):
        return dict(codegen=None, name=self.name)

    def add_ir_module(self, module):
        pass

    def add_linking_library(self, library):
        pass

    def finalize(self):
        pass

    def get_asm_str(self):
        return "ASSEMBLY"

    def get_llvm_str(self):
        return "LLVM"

    def get_function(self, name):
        raise KeyError(f"No functions available (asked for {name})")


class JITMLIRCodegen(Codegen):
    _library_class = MLIRCodeLibrary

    def __init__(self, module_name):
        pass

    def _add_module(self, module):
        pass

    def _create_empty_module(self, name):
        with Context():
            return Module.create(loc=Location.name(name))

    def magic_tuple(self):
        """
        Return a tuple unambiguously describing the codegen behaviour.
        Used for cache key generation.
        """
        from cusimt.tools import get_gpu_compute_capability, get_cuda_runtime_version

        cc = get_gpu_compute_capability(tuple)
        return (get_cuda_runtime_version(), cc)
