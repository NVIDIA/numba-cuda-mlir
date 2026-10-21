# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A launch whose argument count disagrees with the kernel's parameters is refused.

Surplus arguments used to be dropped: the kernel ran, the launch reported
success, and an array the caller passed was never written.  Too few arguments
escaped as a bare ``IndexError`` from inside the compiler entry point, naming
neither the kernel nor the mismatch.
"""

import numpy as np
import pytest

from numba_cuda_mlir import cuda


def test_surplus_argument_is_refused():
    """One extra argument must not be silently dropped."""

    @cuda.jit
    def k(out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = i + 100

    out = np.zeros(4, np.int32)
    extra = np.zeros(4, np.int32)
    with pytest.raises(TypeError, match=r"k\(\).*too many positional arguments.*2 given"):
        k[1, 4](out, extra)


def test_several_surplus_arguments_are_refused():
    @cuda.jit
    def k(out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = i + 100

    out = np.zeros(4, np.int32)
    with pytest.raises(TypeError, match=r"k\(\).*too many positional arguments.*4 given"):
        k[1, 4](out, 1, 2, 3)


def test_missing_argument_raises_type_error_not_index_error():
    """The shortfall is reported as a TypeError, not a bare IndexError."""

    @cuda.jit
    def k(out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = i + 100

    with pytest.raises(TypeError, match=r"k\(\).*missing a required argument.*out.*0 given"):
        k[1, 4]()


def test_exact_argument_count_still_launches():
    @cuda.jit
    def k(out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = i + 100

    out = cuda.to_device(np.zeros(4, np.int32))
    k[1, 4](out)
    cuda.synchronize()
    np.testing.assert_equal(out.copy_to_host(), np.arange(100, 104, dtype=np.int32))


def test_omitting_a_defaulted_argument_is_not_refused():
    """Defaults make a shorter call legal, so the floor is the required count.

    Only the count is asserted here.  Omitting a defaulted argument currently
    leaves the parameter holding garbage rather than the default, which is a
    separate defect in how the launcher passes values; this test must not start
    passing or failing on account of it.
    """

    @cuda.jit
    def k(out, addend=7):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = i + addend

    out = cuda.to_device(np.zeros(4, np.int32))
    k[1, 4](out)  # must not raise
    cuda.synchronize()

    supplied = cuda.to_device(np.zeros(4, np.int32))
    k[1, 4](supplied, 20)
    cuda.synchronize()
    np.testing.assert_equal(supplied.copy_to_host(), np.arange(20, 24, dtype=np.int32))


def test_varargs_kernel_accepts_any_trailing_count():
    """A ``*args`` kernel has no upper bound, so the check must not impose one."""

    @cuda.jit
    def k(out, *rest):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = len(rest)

    for count in (0, 1, 3):
        out = cuda.to_device(np.zeros(4, np.int32))
        k[1, 4](out, *range(count))
        cuda.synchronize()
        np.testing.assert_equal(out.copy_to_host(), np.full(4, count, np.int32))
