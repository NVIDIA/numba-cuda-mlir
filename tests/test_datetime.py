# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import itertools
import operator

import cuda.simt as cuda
import numpy as np
import pytest
from cusimt.numba_cuda.np import npdatetime_helpers

date_units = ("Y", "M")
time_units = ("W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as")
all_units = date_units + time_units


def _has_safe_factor(a, b):
    best = npdatetime_helpers.get_best_unit(a, b)
    coarser = b if best == a else a
    factor = npdatetime_helpers.get_timedelta_conversion_factor(coarser, best)
    return factor is not None and factor <= 10**15


td_cross_pairs = [
    (a, b) for a, b in itertools.permutations(date_units, 2) if _has_safe_factor(a, b)
] + [(a, b) for a, b in itertools.permutations(time_units, 2) if _has_safe_factor(a, b)]

dt_td_cross_pairs = [
    (dt_u, td_u)
    for dt_u, td_u in itertools.product(all_units, all_units)
    if dt_u != td_u
    and npdatetime_helpers.combine_datetime_timedelta_units(dt_u, td_u) is not None
    and _has_safe_factor(dt_u, td_u)
]

dt_dt_cross_pairs = [
    (a, b) for a, b in itertools.permutations(date_units, 2) if _has_safe_factor(a, b)
] + [(a, b) for a, b in itertools.permutations(time_units, 2) if _has_safe_factor(a, b)]

cmp_ops = [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]


def _run_binop(op, a, b):
    expected = op(a, b)

    @cuda.jit
    def kernel(out, x, y):
        out[0] = op(x, y)

    out = np.zeros(1, dtype=expected.dtype)
    kernel[1, 1](out, a, b)
    if np.issubdtype(out.dtype, np.floating):
        np.testing.assert_almost_equal(out[0], expected)
    else:
        assert out[0] == expected


@pytest.mark.parametrize("unit", all_units)
@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_timedelta_arith(op, unit):
    _run_binop(op, np.timedelta64(7, unit), np.timedelta64(3, unit))


@pytest.mark.parametrize("ua,ub", td_cross_pairs)
@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_timedelta_arith_cross_unit(op, ua, ub):
    _run_binop(op, np.timedelta64(3, ua), np.timedelta64(2, ub))


def test_timedelta_add_commutative():
    @cuda.jit
    def kernel(out1, out2, x, y):
        out1[0] = x + y
        out2[0] = y + x

    a, b = np.timedelta64(3, "D"), np.timedelta64(7, "D")
    out1 = np.zeros(1, dtype="timedelta64[D]")
    out2 = np.zeros(1, dtype="timedelta64[D]")
    kernel[1, 1](out1, out2, a, b)
    assert out1[0] == out2[0] == a + b


@pytest.mark.parametrize("unit", all_units)
@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_datetime_timedelta_arith(op, unit):
    _run_binop(op, np.datetime64(100, unit), np.timedelta64(5, unit))


@pytest.mark.parametrize("dt_u,td_u", dt_td_cross_pairs)
def test_datetime_add_timedelta_cross_unit(dt_u, td_u):
    _run_binop(operator.add, np.datetime64(100, dt_u), np.timedelta64(5, td_u))


def test_datetime_add_timedelta_commutative():
    @cuda.jit
    def kernel(out1, out2, d, t):
        out1[0] = d + t
        out2[0] = t + d

    dt = np.datetime64("2014-01-01")
    td = np.timedelta64(5, "D")
    out1 = np.zeros(1, dtype="datetime64[D]")
    out2 = np.zeros(1, dtype="datetime64[D]")
    kernel[1, 1](out1, out2, dt, td)
    assert out1[0] == out2[0] == dt + td


@pytest.mark.parametrize("unit", all_units)
def test_datetime_difference(unit):
    _run_binop(operator.sub, np.datetime64(100, unit), np.datetime64(37, unit))


@pytest.mark.parametrize("ua,ub", dt_dt_cross_pairs)
def test_datetime_difference_cross_unit(ua, ub):
    _run_binop(operator.sub, np.datetime64(100, ua), np.datetime64(50, ub))


@pytest.mark.parametrize("unit", all_units)
@pytest.mark.parametrize("op", cmp_ops)
def test_timedelta_comparisons(op, unit):
    _run_binop(op, np.timedelta64(3, unit), np.timedelta64(7, unit))
    _run_binop(op, np.timedelta64(5, unit), np.timedelta64(5, unit))


@pytest.mark.parametrize("unit", all_units)
@pytest.mark.parametrize("op", cmp_ops)
def test_datetime_comparisons(op, unit):
    _run_binop(op, np.datetime64(100, unit), np.datetime64(200, unit))
    _run_binop(op, np.datetime64(50, unit), np.datetime64(50, unit))


@pytest.mark.parametrize("unit", all_units)
def test_timedelta_mul_int(unit):
    _run_binop(operator.mul, np.timedelta64(3, unit), 2)


@pytest.mark.parametrize("unit", all_units)
def test_timedelta_rmul_int(unit):
    _run_binop(operator.mul, 2, np.timedelta64(3, unit))


@pytest.mark.parametrize("unit", all_units)
def test_timedelta_mul_float(unit):
    _run_binop(operator.mul, np.timedelta64(7, unit), 1.5)


@pytest.mark.parametrize("unit", all_units)
def test_timedelta_rmul_float(unit):
    _run_binop(operator.mul, 1.5, np.timedelta64(7, unit))


@pytest.mark.parametrize("unit", all_units)
def test_timedelta_floordiv_int(unit):
    _run_binop(operator.floordiv, np.timedelta64(7, unit), 2)


@pytest.mark.parametrize("unit", all_units)
def test_timedelta_floordiv_negative(unit):
    _run_binop(operator.floordiv, np.timedelta64(-7, unit), 2)


@pytest.mark.parametrize("unit", all_units)
def test_timedelta_truediv_float(unit):
    _run_binop(operator.truediv, np.timedelta64(7, unit), 2.0)


@pytest.mark.parametrize("unit", all_units)
def test_timedelta_div_timedelta(unit):
    _run_binop(operator.truediv, np.timedelta64(7, unit), np.timedelta64(3, unit))


@pytest.mark.parametrize("unit", all_units)
def test_timedelta_neg(unit):
    a = np.timedelta64(3, unit)
    expected = -a

    @cuda.jit
    def kernel(out, x):
        out[0] = -x

    out = np.zeros(1, dtype=a.dtype)
    kernel[1, 1](out, a)
    assert out[0] == expected


@pytest.mark.parametrize("unit", all_units)
def test_timedelta_abs(unit):
    a = np.timedelta64(-4, unit)
    expected = abs(a)

    @cuda.jit
    def kernel(out, x):
        out[0] = abs(x)

    out = np.zeros(1, dtype=a.dtype)
    kernel[1, 1](out, a)
    assert out[0] == expected


@pytest.mark.parametrize("unit", all_units)
def test_array_datetime_sub(unit):
    @cuda.jit
    def kernel(start, end, delta):
        for i in range(cuda.grid(1), delta.size, cuda.gridsize(1)):
            delta[i] = end[i] - start[i]

    n = 10
    dt = f"datetime64[{unit}]"
    td = f"timedelta64[{unit}]"
    a = (np.arange(n) + 10).astype(dt)
    b = (np.arange(n) + 20).astype(dt)
    out = np.zeros(n, dtype=td)
    kernel[1, 32](a, b, out)
    np.testing.assert_array_equal(out, b - a)


def test_scalar_datetime_args():
    @cuda.jit
    def foo(dates, target, delta, matches, outdelta):
        for i in range(cuda.grid(1), matches.size, cuda.gridsize(1)):
            matches[i] = dates[i] == target
            outdelta[i] = dates[i] - delta

    arr1 = np.arange("2005-02", "2006-02", dtype="datetime64[D]")
    target = arr1[5]
    delta = arr1[6] - arr1[5]
    matches = np.zeros_like(arr1, dtype=np.bool_)
    outdelta = np.zeros_like(arr1, dtype="datetime64[D]")

    foo[1, 32](arr1, target, delta, matches, outdelta)
    where = matches.nonzero()
    assert list(where) == [5]
    np.testing.assert_array_equal(outdelta, arr1 - delta)
