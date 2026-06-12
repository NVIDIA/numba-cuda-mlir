# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba_cuda_mlir.numba_cuda.core.options import TargetOptions
from .target import CUDATargetContext, CUDATypingContext


class CUDATargetOptions(TargetOptions):
    pass


class CUDATarget:
    def __init__(self, name):
        self.options = CUDATargetOptions
        # The typing and target contexts are initialized only when needed -
        # this prevents an attempt to load CUDA libraries at import time on
        # systems that might not have them present.
        self._typingctx = None
        self._targetctx = None
        self._typingctx_initialized = False
        self._targetctx_initialized = False
        self._typingctx_initializing = False
        self._targetctx_initializing = False
        self._initializing = False
        self._target_name = name

    @property
    def typing_context(self):
        if self._typingctx is None:
            self._typingctx = CUDATypingContext()
        return self._typingctx

    @property
    def target_context(self):
        if self._targetctx is None:
            self._targetctx = CUDATargetContext(self.typing_context)
        return self._targetctx

    def _seed_target_registry(self):
        if self.target_context._registries:
            return
        from numba_cuda_mlir.numba_cuda.core.imputils import builtin_registry

        self.target_context.install_registry(builtin_registry)

    def ensure_initialized(self):
        if self._typingctx_initialized and self._targetctx_initialized:
            return
        if self._initializing:
            return

        self._initializing = True
        try:
            self._seed_target_registry()

            if not self._typingctx_initialized:
                self._typingctx_initializing = True
                try:
                    self.typing_context.refresh()
                except Exception:
                    self._typingctx_initialized = False
                    raise
                else:
                    self._typingctx_initialized = True
                finally:
                    self._typingctx_initializing = False

            if not self._targetctx_initialized:
                self._targetctx_initializing = True
                try:
                    self.target_context.refresh()
                except Exception:
                    self._targetctx_initialized = False
                    raise
                else:
                    self._targetctx_initialized = True
                finally:
                    self._targetctx_initializing = False

            from numba_cuda_mlir.device_declarations import (
                apply_device_declarations,
            )
            from numba_cuda_mlir.numba_cuda.typing.templates import builtin_registry

            self.typing_context.install_registry(builtin_registry)
            apply_device_declarations(self.typing_context, self.target_context)
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
            if typing and target:
                from numba_cuda_mlir.device_declarations import (
                    apply_device_declarations,
                )

                apply_device_declarations(self.typing_context, self.target_context)
        finally:
            self._initializing = False


cuda_target = CUDATarget("cuda")
