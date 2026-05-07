# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.numba_cuda.extending import intrinsic
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir.dialects import nvvm


@intrinsic
def breakpoint(typingctx):
    def codegen(ctx, builder, sig, args):
        nvvm.inline_ptx([], [], [], "brkpt;")
        return None

    return types.void(), codegen


@intrinsic
def nanosleep(typingctx, seconds):
    if isinstance(seconds, types.Number):

        def codegen(builder, target, args, kwargs):
            arg = args[0]
            arg = builder.load_var(arg)
            arg = builder.mlir_convert(arg, T.i32())
            nvvm.inline_ptx(
                write_only_args=[],
                read_only_args=[arg],
                read_write_args=[],
                ptx_code=f"nanosleep.u32 $0;",
            )
            return None

        return types.void(seconds), codegen
