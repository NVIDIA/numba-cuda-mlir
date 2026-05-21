# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NRT (Numba Runtime) constants and utilities.

NRT device functions are emitted as MLIR LLVM dialect ops by nrt_mlir.py.
The old NVRTC compilation path (nrt.cu) has been removed.
"""

from pathlib import Path

_NRT_DIR = Path(__file__).parent


def get_include():
    """Return the include path for the NRT headers."""
    return str(_NRT_DIR)


NRT_FUNCTIONS = frozenset(
    [
        "NRT_Allocate",
        "NRT_MemInfo_alloc",
        "NRT_MemInfo_init",
        "NRT_MemInfo_new",
        "NRT_Free",
        "NRT_dealloc",
        "NRT_MemInfo_destroy",
        "NRT_MemInfo_call_dtor",
        "NRT_MemInfo_data_fast",
        "NRT_MemInfo_alloc_aligned",
        "NRT_Allocate_External",
        "NRT_decref",
        "NRT_incref",
    ]
)


def needs_nrt_linking(asm: str) -> bool:
    """Check if the given assembly/PTX references NRT functions."""
    return any(fn in asm for fn in NRT_FUNCTIONS)
