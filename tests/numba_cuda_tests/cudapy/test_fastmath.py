# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import sys
from typing import List
from dataclasses import dataclass, field
import numba_cuda_mlir
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.types import float32
from numba_cuda_mlir.compiler import compile_ptx
from math import cos, sin, tan, exp, log, log10, log2, pow, tanh, nextafter
from operator import truediv
import numpy as np
from numba_cuda_mlir.testing import NumbaCUDATestCase
import pytest


@dataclass
class FastMathCriterion:
    fast_expected: List[str] = field(default_factory=list)
    fast_unexpected: List[str] = field(default_factory=list)
    prec_expected: List[str] = field(default_factory=list)
    prec_unexpected: List[str] = field(default_factory=list)

    def check(self, test: NumbaCUDATestCase, fast: str, prec: str):
        for i in self.fast_expected:
            test.assertIn(i, fast)
        for i in self.fast_unexpected:
            test.assertNotIn(i, fast)
        for i in self.prec_expected:
            test.assertIn(i, prec)
        for i in self.prec_unexpected:
            test.assertNotIn(i, prec)


def _first_asm(dispatcher, sig):
    asm = dispatcher.inspect_asm(sig)
    if isinstance(asm, dict):
        return next(iter(asm.values()))
    return asm


class TestFastMathOption(NumbaCUDATestCase):
    def _test_fast_math_common(self, pyfunc, sig, device, criterion, fastmath=True):
        # Test jit code path
        fastver = numba_cuda_mlir.cuda.jit(sig, device=device, fastmath=fastmath)(pyfunc)
        precver = numba_cuda_mlir.cuda.jit(sig, device=device)(pyfunc)

        criterion.check(self, _first_asm(fastver, sig), _first_asm(precver, sig))

        # Test compile_ptx code path
        fastptx, _ = compile_ptx(pyfunc, sig, device=device, fastmath=fastmath)
        precptx, _ = compile_ptx(pyfunc, sig, device=device)

        criterion.check(self, fastptx, precptx)

    def _test_fast_math_unary(self, op, criterion: FastMathCriterion):
        def kernel(r, x):
            r[0] = op(x)

        def device_function(x):
            return op(x)

        self._test_fast_math_common(
            kernel, (float32[::1], float32), device=False, criterion=criterion
        )
        self._test_fast_math_common(device_function, (float32,), device=True, criterion=criterion)

    def _test_fast_math_binary(self, op, criterion: FastMathCriterion):
        def kernel(r, x, y):
            r[0] = op(x, y)

        def device(x, y):
            return op(x, y)

        self._test_fast_math_common(
            kernel,
            (float32[::1], float32, float32),
            device=False,
            criterion=criterion,
        )
        self._test_fast_math_common(device, (float32, float32), device=True, criterion=criterion)

    def test_cosf(self):
        self._test_fast_math_unary(
            cos,
            FastMathCriterion(
                fast_expected=["cos.approx.ftz.f32 "],
                prec_unexpected=["cos.approx.ftz.f32 "],
            ),
        )

    def test_sinf(self):
        self._test_fast_math_unary(
            sin,
            FastMathCriterion(
                fast_expected=["sin.approx.ftz.f32 "],
                prec_unexpected=["sin.approx.ftz.f32 "],
            ),
        )

    def test_tanf(self):
        self._test_fast_math_unary(
            tan,
            FastMathCriterion(
                fast_expected=[
                    "sin.approx.ftz.f32 ",
                    "cos.approx.ftz.f32 ",
                    "div.approx.ftz.f32 ",
                ],
                prec_unexpected=["sin.approx.ftz.f32 "],
            ),
        )

    def test_tanhf(self):
        self._test_fast_math_unary(
            tanh,
            FastMathCriterion(
                fast_expected=["tanh.approx.f32 "],
                prec_unexpected=["tanh.approx.f32 "],
            ),
        )

    def test_tanhf_compile_ptx(self):
        def tanh_kernel(r, x):
            r[0] = tanh(x)

        fastptx, _ = compile_ptx(tanh_kernel, (float32[::1], float32), fastmath=True)
        precptx, _ = compile_ptx(tanh_kernel, (float32[::1], float32))

        criterion = FastMathCriterion(
            fast_expected=["tanh.approx.f32 "],
            prec_unexpected=["tanh.approx.f32 "],
        )

        criterion.check(self, fastptx, precptx)

    def test_expf(self):
        self._test_fast_math_unary(
            exp,
            FastMathCriterion(fast_unexpected=["fma.rn.f32 "], prec_expected=["fma.rn.f32 "]),
        )

    @pytest.mark.skipif(sys.version_info < (3, 11), reason="Python 3.11+ required")
    def test_exp2f(self):
        from math import exp2

        self._test_fast_math_unary(
            exp2,
            FastMathCriterion(
                fast_expected=["ex2.approx.ftz.f32 "],
                prec_expected=["ex2.approx.f32 "],
                prec_unexpected=["ex2.approx.ftz.f32 "],
            ),
        )

    def test_logf(self):
        # Look for constant used to convert from log base 2 to log base e
        self._test_fast_math_unary(
            log,
            FastMathCriterion(
                fast_expected=["lg2.approx.ftz.f32 ", "0f3F317218"],
                prec_unexpected=["lg2.approx.ftz.f32 "],
            ),
        )

    def test_log10f(self):
        # Look for constant used to convert from log base 2 to log base 10
        self._test_fast_math_unary(
            log10,
            FastMathCriterion(
                fast_expected=["lg2.approx.ftz.f32 ", "0f3E9A209B"],
                prec_unexpected=["lg2.approx.ftz.f32 "],
            ),
        )

    def test_log2f(self):
        self._test_fast_math_unary(
            log2,
            FastMathCriterion(
                fast_expected=["lg2.approx.ftz.f32 "],
                prec_unexpected=["lg2.approx.ftz.f32 "],
            ),
        )

    def test_powf(self):
        self._test_fast_math_binary(
            pow,
            FastMathCriterion(
                fast_expected=["lg2.approx.ftz.f32 "],
                prec_unexpected=["lg2.approx.ftz.f32 "],
            ),
        )

    def test_nextafterf(self):
        self._test_fast_math_binary(
            nextafter,
            FastMathCriterion(
                fast_expected=[".ftz.f32 "],
                prec_unexpected=[".ftz.f32 "],
            ),
        )

    def test_divf(self):
        self._test_fast_math_binary(
            truediv,
            FastMathCriterion(
                fast_expected=["div.approx.ftz.f32 "],
                fast_unexpected=["div.rn.f32"],
                prec_expected=["div.rn.f32"],
                prec_unexpected=["div.approx.ftz.f32 "],
            ),
        )

    def test_divf_arcp_only(self):
        # Selective fastmath: arcp alone opts into approximate division.
        def kernel(r, x, y):
            r[0] = x / y

        sig = (float32[::1], float32, float32)
        arcpptx, _ = compile_ptx(kernel, sig, fastmath={"arcp"})
        self.assertIn("div.approx", arcpptx)
        self.assertNotIn("div.rn.f32", arcpptx)

    def test_divf_arcp_only_jit(self):
        # The selective form must also survive the dispatcher path, not
        # just compile_ptx.
        def kernel(r, x, y):
            r[0] = x / y

        sig = (float32[::1], float32, float32)
        fastver = cuda.jit(sig, fastmath={"arcp"})(kernel)
        self.assertIn("div.approx", _first_asm(fastver, sig))

    def test_divf_nsz_only_stays_precise_division(self):
        # Selective fastmath: a subset without arcp/fast must not use the
        # fastest approximate division.
        def kernel(r, x, y):
            r[0] = x / y

        sig = (float32[::1], float32, float32)
        nszptx, _ = compile_ptx(kernel, sig, fastmath={"nsz"})
        self.assertNotIn("div.approx", nszptx)

    def test_sinf_afn_only(self):
        # afn opts into approximate functions without touching division.
        def kernel(r, x, y):
            r[0] = sin(x) / y

        sig = (float32[::1], float32, float32)
        afnptx, _ = compile_ptx(kernel, sig, fastmath={"afn"})
        self.assertIn("sin.approx", afnptx)
        self.assertNotIn("div.approx", afnptx)

    def test_fastmath_flags_in_mlir(self):
        # Selective flags are stamped per-op as #arith.fastmath attributes.
        from numba_cuda_mlir.compiler import compile_mlir

        def kernel(r, x, y):
            r[0] = x / y

        sig = (float32[::1], float32, float32)
        m = str(compile_mlir(kernel, sig, fastmath={"nnan", "arcp"}))
        self.assertIn("fastmath<nnan,arcp>", m)
        m = str(compile_mlir(kernel, sig, fastmath=True))
        self.assertIn("fastmath<fast>", m)
        m = str(compile_mlir(kernel, sig))
        self.assertNotIn("fastmath<", m)

    def test_fastmath_invalid_flag_rejected(self):
        def kernel(r, x, y):
            r[0] = x / y

        sig = (float32[::1], float32, float32)
        with self.assertRaises(Exception) as cm:
            compile_ptx(kernel, sig, fastmath={"bogus"})
        self.assertIn("bogus", str(cm.exception))

    def test_fastmath_with_lineinfo(self):
        # Regression test: with debug info present, the LLVM70 fast-math
        # text round-trip used to fail re-parsing on the compile unit's
        # emission kind ("expected emission kind").
        def kernel(r, x, y):
            r[0] = x / y

        sig = (float32[::1], float32, float32)
        ptx, _ = compile_ptx(kernel, sig, fastmath=True, lineinfo=True)
        self.assertIn("div.approx", ptx)
        self.assertIn(".file", ptx)

    def test_fastmath_with_debug(self):
        # Full debug info takes the same LLVM70 text round-trip as lineinfo;
        # both fast-math and the debug line table must survive it.
        def kernel(r, x, y):
            r[0] = x / y

        sig = (float32[::1], float32, float32)
        ptx, _ = compile_ptx(kernel, sig, fastmath=True, debug=True, opt=False)
        self.assertIn("div.approx", ptx)
        self.assertIn(".file", ptx)

    def test_nvvm_knobs_follow_flags(self):
        # The module-level libnvvm/ptxas knobs are implied per-flag, with
        # 'fast' (the bool form) enabling all four as numba-cuda does.
        from numba_cuda_mlir.fastmath import nvvm_fastmath_options

        self.assertEqual(
            nvvm_fastmath_options(True),
            {"ftz": True, "fma": True, "prec_div": False, "prec_sqrt": False},
        )
        self.assertEqual(nvvm_fastmath_options(False), {})
        self.assertEqual(nvvm_fastmath_options({"arcp"}), {"prec_div": False})
        self.assertEqual(nvvm_fastmath_options({"afn"}), {"prec_sqrt": False})
        self.assertEqual(nvvm_fastmath_options({"contract"}), {"fma": True})
        self.assertEqual(nvvm_fastmath_options({"nnan", "nsz"}), {})

    @pytest.mark.xfail(
        True,
        reason="mlir backend does not yet raise ZeroDivisionError for float "
        "division under debug=True (python error model); unrelated to fastmath",
    )
    def test_divf_exception(self):
        def f10(r, x, y):
            r[0] = x / y

        sig = (float32[::1], float32, float32)
        fastver = numba_cuda_mlir.cuda.jit(sig, fastmath=True, debug=True, opt=False)(f10)
        precver = numba_cuda_mlir.cuda.jit(sig, debug=True, opt=False)(f10)
        nelem = 10
        ary = np.empty(nelem, dtype=np.float32)
        with self.assertRaises(ZeroDivisionError):
            precver[1, nelem](ary, 10.0, 0.0)

        try:
            fastver[1, nelem](ary, 10.0, 0.0)
        except ZeroDivisionError:
            self.fail("Divide in fastmath should not throw ZeroDivisionError")

    @pytest.mark.xfail(True, reason="fastmath option doesn't propagate")
    def test_device_fastmath_propagation(self):
        # The fastmath option doesn't presently propagate to device functions
        # from their callees - arguably it should do, so this test is presently
        # an xfail.
        @numba_cuda_mlir.cuda.jit("float32(float32, float32)", device=True)
        def foo(a, b):
            return a / b

        def bar(arr, val):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] = foo(i, val)

        sig = (float32[::1], float32)
        fastver = numba_cuda_mlir.cuda.jit(sig, fastmath=True)(bar)
        precver = numba_cuda_mlir.cuda.jit(sig)(bar)

        # Variants of the div instruction are further documented at:
        # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div

        # The fast version should use the "fast, approximate divide" variant
        self.assertIn("div.approx.f32", fastver.inspect_asm(sig))
        # The precise version should use the "IEEE 754 compliant rounding"
        # variant, and neither of the "approximate divide" variants.
        self.assertIn("div.rn.f32", precver.inspect_asm(sig))
        self.assertNotIn("div.approx.f32", precver.inspect_asm(sig))
        self.assertNotIn("div.full.f32", precver.inspect_asm(sig))
