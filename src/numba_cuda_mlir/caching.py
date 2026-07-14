# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Caching support for numba_cuda_mlir.

This module provides cache implementations for MLIRDispatcher, allowing
compiled CUDA kernels to be cached to disk and reloaded in subsequent runs.
"""

from numba_cuda_mlir.numba_cuda import typing
from numba_cuda_mlir.numba_cuda.core.caching import Cache, CacheImpl, NullCache


class CachedCompileResult:
    """
    A minimal compile result reconstructed from cached data.

    This provides just enough interface to be used by MLIRDispatcher
    without requiring the full compilation infrastructure.
    """

    def __init__(self, signature, metadata):
        self.signature = signature
        self.metadata = metadata
        # Provide a codegen attribute for cache key computation
        self.codegen = _CachedCodegen()
        # entry_point is needed for recompile() to work
        self.entry_point = metadata.get("func_name")


class _CachedCodegen:
    """Minimal codegen for cache key computation."""

    def magic_tuple(self):
        # Return a consistent tuple for MLIR-compiled code
        from numba_cuda_mlir.tools import (
            get_gpu_compute_capability,
            get_cuda_runtime_version,
        )

        cc = get_gpu_compute_capability(tuple)
        return (get_cuda_runtime_version(), cc)


class MLIRCacheImpl(CacheImpl):
    """
    Cache implementation for MLIR-compiled CUDA kernels.

    Handles serialization and deserialization of compile results.
    """

    def reduce(self, cres):
        """Serialize a compile result for caching."""
        return {
            "signature_args": cres.signature.args,
            "signature_return_type": cres.signature.return_type,
            "cubin": cres.metadata.get("cubin"),
            "ptx": cres.metadata.get("ptx"),
            "ltoir": cres.metadata.get("ltoir"),
            "func_name": cres.metadata.get("func_name"),
            "mlir_module_optimized": cres.metadata.get("mlir_module_optimized"),
            "needs_nrt": cres.metadata.get("needs_nrt"),
            "nrt_inline": cres.metadata.get("nrt_inline"),
            "targetoptions": cres.metadata.get("targetoptions", {}),
            "gpu_target": cres.metadata.get("gpu_target"),
            "check_error_code": cres.metadata.get("check_error_code"),
        }

    def rebuild(self, target_context, payload):
        """Deserialize a compile result from cache."""
        signature_args = payload["signature_args"]
        signature_return_type = payload["signature_return_type"]
        cubin = payload["cubin"]
        ptx = payload["ptx"]
        ltoir = payload.get("ltoir")
        func_name = payload["func_name"]
        mlir_module_optimized = payload.get("mlir_module_optimized")
        needs_nrt = payload.get("needs_nrt")
        nrt_inline = payload.get("nrt_inline")
        targetoptions = payload.get("targetoptions", {})
        gpu_target = payload.get("gpu_target")
        check_error_code = payload.get("check_error_code", True)

        signature = typing.signature(signature_return_type, *signature_args)

        return CachedCompileResult(
            signature=signature,
            metadata={
                "cubin": cubin,
                "ptx": ptx,
                "ltoir": ltoir,
                "func_name": func_name,
                "mlir_module_optimized": mlir_module_optimized,
                "needs_nrt": needs_nrt,
                "nrt_inline": nrt_inline,
                "targetoptions": targetoptions,
                "gpu_target": gpu_target,
                "check_error_code": check_error_code,
            },
        )

    def check_cachable(self, cres):
        """Check if a compile result can be cached."""
        targetoptions = cres.metadata.get("targetoptions", {})
        link = targetoptions.get("link", [])
        if link:
            raise RuntimeError("Cannot pickle CUDACodeLibrary with linking files")
        return True


class MLIRCache(Cache):
    """
    Cache for MLIR-compiled CUDA kernels.

    Uses the standard numba caching infrastructure with MLIR-specific
    serialization.
    """

    _impl_class = MLIRCacheImpl

    def __init__(self, py_func, targetoptions=None):
        self._targetoptions = targetoptions if targetoptions is not None else {}
        super().__init__(py_func)

    def _index_key(self, sig, codegen):
        key = super()._index_key(sig, codegen)
        targetoptions = self._targetoptions
        from numba_cuda_mlir.tools import resolve_gpu_target

        gpu_target = resolve_gpu_target(targetoptions)
        option_key = (
            ("lto_explicit", targetoptions.get("_lto_explicit", False)),
            ("lto", targetoptions.get("lto")),
            ("chip", gpu_target["chip"]),
            ("launch_bounds", targetoptions.get("launch_bounds")),
        )
        return (*key, option_key)


# Re-export NullCache for convenience
__all__ = ["MLIRCache", "MLIRCacheImpl", "NullCache", "CachedCompileResult"]
