# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
import numpy as np
import pytest


@pytest.mark.parametrize(
    "case,args,expected",
    [
        ("all_defaults", (2.0,), 21.5),
        ("override_int", (2.0, 5), 11.5),
        ("override_float", (2.0, 3.0), 23.0),
        ("override_bool_false", (2.0, False), 3.5),
        ("override_opt", (2.0, 2.0), 43.0),
        ("override_all", (2.0, 5, 2.0, False, 3.0), 12.0),
    ],
)
def test_device_defaults(case, args, expected):
    @cuda.jit(device=True)
    def compute(x, int_val=10, float_val=1.5, do_scale=True, opt_mult=None):
        if do_scale:
            result = x * int_val
        else:
            result = x
        result = result + float_val
        if opt_mult is not None:
            result = result * opt_mult
        return result

    if case == "all_defaults":

        @cuda.jit
        def k(r, x):
            r[0] = compute(x)

    elif case == "override_int":

        @cuda.jit
        def k(r, x, int_val):
            r[0] = compute(x, int_val)

    elif case == "override_float":

        @cuda.jit
        def k(r, x, float_val):
            r[0] = compute(x, float_val=float_val)

    elif case == "override_bool_false":

        @cuda.jit
        def k(r, x, do_scale):
            r[0] = compute(x, do_scale=do_scale)

    elif case == "override_opt":

        @cuda.jit
        def k(r, x, opt_mult):
            r[0] = compute(x, opt_mult=opt_mult)

    elif case == "override_all":

        @cuda.jit
        def k(r, x, int_val, float_val, do_scale, opt_mult):
            r[0] = compute(x, int_val, float_val, do_scale, opt_mult)

    r = np.zeros(1, dtype=np.float64)
    k[1, 1](r, *args)
    assert r[0] == expected


if __name__ == "__main__":
    for case, args, expected in [
        ("all_defaults", (2.0,), 21.5),
        ("override_int", (2.0, 5), 11.5),
        ("override_float", (2.0, 3.0), 23.0),
        ("override_bool_false", (2.0, False), 3.5),
        ("override_opt", (2.0, 2.0), 43.0),
        ("override_all", (2.0, 5, 2.0, False, 3.0), 12.0),
    ]:
        test_device_defaults(case, args, expected)
