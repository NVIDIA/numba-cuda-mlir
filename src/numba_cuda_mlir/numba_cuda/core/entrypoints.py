# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause


def init_all():
    """Execute all `numba_cuda_mlir_extensions` entry points with the name `init`

    This compatibility entry point delegates to the compiler-owned, lazy,
    process-wide extension bootstrap.
    """

    from numba_cuda_mlir._extension_bootstrap import initialize_extensions

    return initialize_extensions()
