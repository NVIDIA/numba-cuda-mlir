# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt import types

float16 = types.float16

_FUNCS = {
    "habs": (float16,),
    "hadd": (float16, float16),
    "hceil": (float16,),
    "hcos": (float16,),
    "hdiv": (float16, float16),
    "heq": (float16, float16),
    "hexp": (float16,),
    "hexp10": (float16,),
    "hexp2": (float16,),
    "hfloor": (float16,),
    "hfma": (float16, float16, float16),
    "hge": (float16, float16),
    "hgt": (float16, float16),
    "hle": (float16, float16),
    "hlog": (float16,),
    "hlog10": (float16,),
    "hlog2": (float16,),
    "hlt": (float16, float16),
    "hmax": (float16, float16),
    "hmin": (float16, float16),
    "hmul": (float16, float16),
    "hne": (float16, float16),
    "hneg": (float16,),
    "hrcp": (float16,),
    "hrint": (float16,),
    "hrsqrt": (float16,),
    "hsin": (float16,),
    "hsqrt": (float16,),
    "hsub": (float16, float16),
    "htanh": (float16,),
    "htanh_approx": (float16,),
    "htrunc": (float16,),
}

if __name__ == "__main__":
    print("from cusimt.types import float16")
    for func, sig in _FUNCS.items():
        args = [f"arg{i}: float16" for i in range(len(sig))]
        print(f"def {func}({', '.join(args)}) -> float16:\n    ...\n")
