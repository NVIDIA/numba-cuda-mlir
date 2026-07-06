# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pytest
import io
import pathlib
import unittest
import unittest.mock
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.testing import (
    skip_if_cuda_includes_missing,
    skip_if_nvjitlink_missing,
)
from numba_cuda_mlir.testing import NumbaCUDATestCase
from numba_cuda_mlir.linker import Linker as NumbaCudaMLIRLinker
from numba_cuda_mlir.numba_cuda.cudadrv import driver as cuda_driver
from numba_cuda_mlir.numba_cuda.cudadrv.driver import _Linker, LinkerError
from numba_cuda_mlir.numba_cuda import require_context
from numba_cuda_mlir.numba_cuda import void, float64, int64, int32, float32
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof
from numba_cuda_mlir.numba_cuda.cudadrv.linkable_code import CUSource
from cuda.core._utils.cuda_utils import CUDAError

CONST1D = np.arange(10, dtype=np.float64)

test_data_dir = pathlib.Path(__file__).parent.parent / "data"


def simple_const_mem(A):
    C = cuda.const.array_like(CONST1D)
    i = cuda.grid(1)

    A[i] = C[i] + 1.0


def func_with_lots_of_registers(x, a, b, c, d, e, f):
    a1 = 1.0
    a2 = 1.0
    a3 = 1.0
    a4 = 1.0
    a5 = 1.0
    b1 = 1.0
    b2 = 1.0
    b3 = 1.0
    b4 = 1.0
    b5 = 1.0
    c1 = 1.0
    c2 = 1.0
    c3 = 1.0
    c4 = 1.0
    c5 = 1.0
    d1 = 10
    d2 = 10
    d3 = 10
    d4 = 10
    d5 = 10
    for i in range(a):
        a1 += b
        a2 += c
        a3 += d
        a4 += e
        a5 += f
        b1 *= b
        b2 *= c
        b3 *= d
        b4 *= e
        b5 *= f
        c1 /= b
        c2 /= c
        c3 /= d
        c4 /= e
        c5 /= f
        d1 <<= b
        d2 <<= c
        d3 <<= d
        d4 <<= e
        d5 <<= f
    x[cuda.grid(1)] = a1 + a2 + a3 + a4 + a5
    x[cuda.grid(1)] += b1 + b2 + b3 + b4 + b5
    x[cuda.grid(1)] += c1 + c2 + c3 + c4 + c5
    x[cuda.grid(1)] += d1 + d2 + d3 + d4 + d5


def simple_smem(ary, dty):
    sm = cuda.shared.array(100, dty)
    i = cuda.grid(1)
    if i == 0:
        for j in range(100):
            sm[j] = j
    cuda.syncthreads()
    ary[i] = sm[i]


def coop_smem2d(ary):
    i, j = cuda.grid(2)
    sm = cuda.shared.array((10, 20), float32)
    sm[i, j] = (i + 1) / (j + 1)
    cuda.syncthreads()
    ary[i, j] = sm[i, j]


def simple_maxthreads(ary):
    i = cuda.grid(1)
    ary[i] = i


LMEM_SIZE = 1000


def simple_lmem(A, B, dty):
    C = cuda.local.array(LMEM_SIZE, dty)
    for i in range(C.shape[0]):
        C[i] = A[i]
    for i in range(C.shape[0]):
        B[i] = C[i]


@pytest.fixture
def add_from_numba_lto(request, numba_cuda_test_binaries):
    request.instance.add_from_numba_lto = cuda.declare_device(
        "add_from_numba",
        "int32(int32, int32)",
        link=[numba_cuda_test_binaries.test_device_functions_ltoir],
    )
    yield
    del request.instance.add_from_numba_lto


class TestLinker(NumbaCUDATestCase):
    @require_context
    def test_linker_basic(self):
        """Simply go through the constructor and destructor"""
        linker = _Linker(max_registers=0, cc=(7, 5))
        del linker

    def test_variables_used_linker_option(self):
        linker = _Linker(
            max_registers=0,
            cc=(7, 5),
            variables_used=["retained_global", "another_global"],
        )

        self.assertEqual(linker.variables_used, ["retained_global", "another_global"])
        self.assertEqual(
            linker._get_linker_options(ptx=False).variables_used,
            ["retained_global", "another_global"],
        )

        linker.variables_used = "updated_global"

        self.assertEqual(linker.variables_used, "updated_global")
        self.assertEqual(
            linker._get_linker_options(ptx=False).variables_used,
            "updated_global",
        )

    def test_variables_used_passed_to_cuda_core_linker(self):
        captured = {}

        class FakeCudaCoreLinker:
            def __init__(self, *object_codes, options):
                captured["object_codes"] = object_codes
                captured["options"] = options

            def link(self, kind):
                captured["kind"] = kind
                return object()

            def get_info_log(self):
                return ""

            def get_error_log(self):
                return ""

            def close(self):
                captured["closed"] = True

        linker = _Linker(
            max_registers=0,
            cc=(7, 5),
            variables_used=["retained_global"],
        )

        with unittest.mock.patch.object(cuda_driver, "Linker", FakeCudaCoreLinker):
            linker.complete()

        self.assertEqual(captured["options"].variables_used, ["retained_global"])
        self.assertEqual(captured["kind"], "cubin")
        self.assertTrue(captured["closed"])

    def test_public_linker_preserves_variables_used_when_recreated_with_lto(self):
        linker = NumbaCudaMLIRLinker(
            cc=(7, 5),
            variables_used=["retained_global"],
        )

        recreated = linker.recreate_with_lto()

        self.assertEqual(recreated.variables_used, ["retained_global"])
        self.assertEqual(
            recreated._get_linker_options(ptx=False).variables_used,
            ["retained_global"],
        )

    def _test_linking(self, eager):
        global bar  # must be a global; other it is recognized as a freevar
        bar = cuda.declare_device("bar", "int32(int32)")

        link = str(test_data_dir / "jitlink.ptx")

        if eager:
            args = ["void(int32[:], int32[:])"]
        else:
            args = []

        @cuda.jit(*args, link=[link])
        def foo(x, y):
            i = cuda.grid(1)
            x[i] += bar(y[i])

        A = np.array([123], dtype=np.int32)
        B = np.array([321], dtype=np.int32)

        foo[1, 1](A, B)

        self.assertTrue(A[0] == 123 + 2 * 321)

    def test_linking_lazy_compile(self):
        self._test_linking(eager=False)

    def test_linking_eager_compile(self):
        self._test_linking(eager=True)

    def test_linking_cu(self):
        bar = cuda.declare_device("bar", "int32(int32)")

        link = str(test_data_dir / "jitlink.cu")

        @cuda.jit(link=[link])
        def kernel(r, x):
            i = cuda.grid(1)

            if i < len(r):
                r[i] = bar(x[i])

        x = np.arange(10, dtype=np.int32)
        r = np.zeros_like(x)

        kernel[1, 32](r, x)

        # Matches the operation of bar() in jitlink.cu
        expected = x * 2
        np.testing.assert_array_equal(r, expected)

    def test_linking_cu_log_warning(self):
        bar = cuda.declare_device("bar", "int32(int32)")

        link = str(test_data_dir / "warn.cu")

        with pytest.warns(UserWarning) as w:

            @cuda.jit("void(int32)", link=[link])
            def kernel(x):
                bar(x)

        nvrtc_log_warnings = [wi for wi in w if "NVRTC log messages" in str(wi.message)]
        self.assertEqual(len(nvrtc_log_warnings), 1, "Expected warnings from NVRTC")
        # Check the warning refers to the log messages
        self.assertIn("NVRTC log messages", str(nvrtc_log_warnings[0].message))
        # Check the message pertaining to the unused variable is provided
        self.assertIn("declared but never referenced", str(nvrtc_log_warnings[0].message))

    def test_linking_cu_error(self):
        bar = cuda.declare_device("bar", "int32(int32)")

        link = str(test_data_dir / "error.cu")

        from cuda.core._utils.cuda_utils import NVRTCError

        errty = NVRTCError
        with self.assertRaises(errty) as e:

            @cuda.jit("void(int32)", link=[link])
            def kernel(x):
                bar(x)

        msg = e.exception.args[0]
        # Check the error message refers to the NVRTC compile
        nvrtc_err_str = "NVRTC_ERROR_COMPILATION"
        self.assertIn(nvrtc_err_str, msg)
        # Check the expected error in the CUDA source is reported
        self.assertIn('identifier "SYNTAX" is undefined', msg)
        # Check the filename is reported correctly
        self.assertIn('in the compilation of "error.cu"', msg)

    def test_linking_unknown_filetype_error(self):
        expected_err = "Don't know how to link file with extension .cuh"
        with self.assertRaisesRegex(RuntimeError, expected_err):

            @cuda.jit("void()", link=["header.cuh"], annotations_as_signatures=False)
            def kernel():
                pass

    def test_linking_file_with_no_extension_error(self):
        expected_err = "Don't know how to link file with no extension"
        with self.assertRaisesRegex(RuntimeError, expected_err):

            @cuda.jit("void()", link=["data"], annotations_as_signatures=False)
            def kernel():
                pass

    @skip_if_cuda_includes_missing
    def test_linking_cu_cuda_include(self):
        link = str(test_data_dir / "cuda_include.cu")

        # An exception will be raised when linking this kernel due to the
        # compile failure if CUDA includes cannot be found by Nvrtc.
        @cuda.jit("void()", link=[link], annotations_as_signatures=False)
        def kernel():
            pass

    def test_try_to_link_nonexistent(self):
        with self.assertRaises(LinkerError) as e:

            @cuda.jit("void(int32[::1])", link=["nonexistent.a"])
            def f(x):
                x[0] = 0

        self.assertIn("nonexistent.a not found", str(e.exception))

    def test_set_registers_no_max(self):
        """Ensure that the jitted kernel used in the test_set_registers_* tests
        uses more than 57 registers - this ensures that test_set_registers_*
        are really checking that they reduced the number of registers used from
        something greater than the maximum."""
        compiled = cuda.jit(func_with_lots_of_registers)
        compiled = compiled.specialize(np.empty(32), *range(6))
        self.assertGreater(compiled.get_regs_per_thread(), 57)

    def test_register_pressure_launch_diagnostic(self):
        compiled = cuda.jit(func_with_lots_of_registers)
        compiled = compiled.specialize(np.empty(32), *range(6))

        registers_per_thread = compiled.get_regs_per_thread()
        device = cuda.get_current_device()
        max_registers_per_block = device.MAX_REGISTERS_PER_BLOCK
        threads_per_block = max_registers_per_block // registers_per_thread + 1
        if threads_per_block > device.MAX_THREADS_PER_BLOCK:
            self.skipTest("Kernel does not use enough registers to exceed the device limit")

        required_registers = registers_per_thread * threads_per_block
        suggested_max_registers = max_registers_per_block // threads_per_block
        out = cuda.device_array(threads_per_block, dtype=np.float64)

        with self.assertRaises(CUDAError) as raises:
            compiled[1, threads_per_block](out, 1, 1, 1, 1, 1, 1)

        message = str(raises.exception)
        self.assertIn("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES", message)
        self.assertIn(f"uses {registers_per_thread} registers per thread", message)
        self.assertIn(f"launch block has {threads_per_block} threads", message)
        self.assertIn(f"requiring {required_registers} registers per block", message)
        self.assertIn(f"device limit is {max_registers_per_block}", message)
        self.assertIn(f"cuda.jit(max_registers={suggested_max_registers})", message)

    def test_set_registers_57(self):
        compiled = cuda.jit(max_registers=57)(func_with_lots_of_registers)
        compiled = compiled.specialize(np.empty(32), *range(6))
        self.assertLessEqual(compiled.get_regs_per_thread(), 57)

    def test_set_registers_38(self):
        compiled = cuda.jit(max_registers=38)(func_with_lots_of_registers)
        compiled = compiled.specialize(np.empty(32), *range(6))
        self.assertLessEqual(compiled.get_regs_per_thread(), 38)

    def test_set_registers_eager(self):
        sig = void(float64[::1], int64, int64, int64, int64, int64, int64)
        compiled = cuda.jit(sig, max_registers=38)(func_with_lots_of_registers)
        self.assertLessEqual(compiled.get_regs_per_thread(), 38)

    @pytest.mark.xfail(reason="const memory not supported")
    def test_get_const_mem_size(self):
        sig = void(float64[::1])
        compiled = cuda.jit(sig)(simple_const_mem)
        const_mem_size = compiled.get_const_mem_size()
        self.assertGreaterEqual(const_mem_size, CONST1D.nbytes)

    def test_get_no_shared_memory(self):
        compiled = cuda.jit(func_with_lots_of_registers)
        compiled = compiled.specialize(np.empty(32), *range(6))
        shared_mem_size = compiled.get_shared_mem_per_block()
        self.assertEqual(shared_mem_size, 0)

    def test_get_shared_mem_per_block(self):
        sig = void(int32[::1], typeof(np.int32))
        compiled = cuda.jit(sig)(simple_smem)
        shared_mem_size = compiled.get_shared_mem_per_block()
        self.assertEqual(shared_mem_size, 400)

    def test_get_shared_mem_per_specialized(self):
        compiled = cuda.jit(simple_smem)
        compiled_specialized = compiled.specialize(np.zeros(100, dtype=np.int32), np.float64)
        shared_mem_size = compiled_specialized.get_shared_mem_per_block()
        self.assertEqual(shared_mem_size, 800)

    def test_get_max_threads_per_block(self):
        compiled = cuda.jit("void(float32[:,::1])")(coop_smem2d)
        max_threads = compiled.get_max_threads_per_block()
        self.assertGreater(max_threads, 0)

    def test_max_threads_exceeded(self):
        compiled = cuda.jit("void(int32[::1])")(simple_maxthreads)
        max_threads = compiled.get_max_threads_per_block()
        nelem = max_threads + 1
        ary = np.empty(nelem, dtype=np.int32)
        with self.assertRaisesRegex(CUDAError, "CUDA_ERROR_INVALID_VALUE"):
            compiled[1, nelem](ary)

    def test_get_local_mem_per_thread(self):
        sig = void(int32[::1], int32[::1], typeof(np.int32))
        compiled = cuda.jit(sig)(simple_lmem)
        local_mem_size = compiled.get_local_mem_per_thread()
        calc_size = np.dtype(np.int32).itemsize * LMEM_SIZE
        self.assertGreaterEqual(local_mem_size, calc_size)

    def test_get_local_mem_per_specialized(self):
        compiled = cuda.jit(simple_lmem)
        compiled_specialized = compiled.specialize(
            np.zeros(LMEM_SIZE, dtype=np.int32),
            np.zeros(LMEM_SIZE, dtype=np.int32),
            np.float64,
        )
        local_mem_size = compiled_specialized.get_local_mem_per_thread()
        calc_size = np.dtype(np.float64).itemsize * LMEM_SIZE
        self.assertGreaterEqual(local_mem_size, calc_size)

    @pytest.mark.numba_cuda_test_binaries("ltoir")
    @pytest.mark.usefixtures("add_from_numba_lto")
    def test_debug_kernel_with_lto(self):
        add_from_numba = self.add_from_numba_lto

        def debuggable_kernel(result):
            i = cuda.grid(1)
            result[i] = add_from_numba(i, i)

        cuda.jit("void(int32[::1])", debug=True, opt=False)(debuggable_kernel)

    @skip_if_nvjitlink_missing("nvJitLink not installed or new enough (>12.3)")
    def test_link_for_different_cc(self):
        linker = _Linker(max_registers=0, cc=(7, 5), lto=True)
        code = """
__device__ int foo(int x) {
    return x + 1;
}
"""
        linker.add_cu(code, "foo")
        ptx = linker.get_linked_ptx().decode()
        assert "target sm_75" in ptx

    def test_add_cu_defers_nvrtc_compile(self):
        linker = _Linker(max_registers=0, cc=(7, 5))

        with unittest.mock.patch.object(cuda_driver.nvrtc, "compile") as compile:
            linker.add_cu("__device__ int foo() { return 1; }", "foo.cu")

        self.assertEqual(compile.call_count, 0)

    def test_complete_materializes_deferred_cu(self):
        captured = {}

        class FakeCudaCoreLinker:
            def __init__(self, *object_codes, options):
                captured["object_codes"] = object_codes
                captured["options"] = options

            def link(self, kind):
                captured["kind"] = kind
                return object()

            def get_info_log(self):
                return ""

            def get_error_log(self):
                return ""

            def close(self):
                captured["closed"] = True

        fake_obj = unittest.mock.Mock()
        linker = _Linker(max_registers=0, cc=(7, 5))

        with (
            unittest.mock.patch.object(
                cuda_driver.nvrtc, "compile", return_value=(fake_obj, "")
            ) as compile,
            unittest.mock.patch.object(cuda_driver, "Linker", FakeCudaCoreLinker),
        ):
            linker.add_cu("__device__ int foo() { return 1; }", "foo.cu")
            self.assertEqual(compile.call_count, 0)
            linker.complete()

        self.assertEqual(compile.call_count, 1)
        self.assertEqual(
            compile.call_args.args[:3],
            ("__device__ int foo() { return 1; }", "foo.cu", (7, 5)),
        )
        self.assertIs(captured["object_codes"][0], fake_obj)
        self.assertEqual(captured["kind"], "cubin")
        self.assertTrue(captured["closed"])

    def test_get_linked_ptx_materializes_deferred_cu(self):
        captured = {}

        class FakeCudaCoreLinker:
            def __init__(self, *object_codes, options):
                captured["object_codes"] = object_codes
                captured["options"] = options

            def link(self, kind):
                captured["kind"] = kind
                result = unittest.mock.Mock()
                result.code = b"linked-ptx"
                return result

            def get_info_log(self):
                return ""

            def get_error_log(self):
                return ""

            def close(self):
                captured["closed"] = True

        fake_obj = unittest.mock.Mock()
        linker = _Linker(max_registers=0, cc=(7, 5))

        with (
            unittest.mock.patch.object(
                cuda_driver.nvrtc, "compile", return_value=(fake_obj, "")
            ) as compile,
            unittest.mock.patch.object(cuda_driver, "Linker", FakeCudaCoreLinker),
        ):
            linker.add_cu("__device__ int foo() { return 1; }", "foo.cu")
            ptx = linker.get_linked_ptx()

        self.assertEqual(compile.call_count, 1)
        self.assertIs(captured["object_codes"][0], fake_obj)
        self.assertEqual(captured["kind"], "ptx")
        self.assertEqual(ptx, b"linked-ptx")
        self.assertTrue(captured["closed"])

    def test_cusource_stream_is_snapshotted_at_materialization(self):
        stream = io.StringIO()
        stream.write("__device__ int a() { return 1; }")
        source = CUSource(stream, name="shim.cu")

        fake_obj = unittest.mock.Mock()
        linker = _Linker(max_registers=0, cc=(7, 5))

        with unittest.mock.patch.object(cuda_driver.nvrtc, "compile") as compile:
            linker.add_file_guess_ext(source)
            stream.write("\n__device__ int b() { return 2; }")
            self.assertEqual(compile.call_count, 0)
            compile.return_value = (fake_obj, "")
            linker._materialize_pending_cu()

        compiled_source = compile.call_args.args[0]
        self.assertIn("int a()", compiled_source)
        self.assertIn("int b()", compiled_source)

    def test_cusource_data_is_not_read_until_complete(self):
        class TrackingCUSource(CUSource):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.data_reads = 0

            @property
            def data(self):
                self.data_reads += 1
                return super().data

        captured = {}

        class FakeCudaCoreLinker:
            def __init__(self, *object_codes, options):
                captured["object_codes"] = object_codes
                captured["options"] = options

            def link(self, kind):
                captured["kind"] = kind
                return object()

            def get_info_log(self):
                return ""

            def get_error_log(self):
                return ""

            def close(self):
                captured["closed"] = True

        stream = io.StringIO("__device__ int a() { return 1; }")
        source = TrackingCUSource(stream, name="shim.cu")
        fake_obj = unittest.mock.Mock()
        linker = _Linker(max_registers=0, cc=(7, 5))

        with (
            unittest.mock.patch.object(
                cuda_driver.nvrtc, "compile", return_value=(fake_obj, "")
            ) as compile,
            unittest.mock.patch.object(cuda_driver, "Linker", FakeCudaCoreLinker),
        ):
            linker.add_file_guess_ext(source)
            self.assertEqual(source.data_reads, 0)
            self.assertEqual(compile.call_count, 0)

            stream.write("\n__device__ int b() { return 2; }")
            linker.complete()

        self.assertEqual(source.data_reads, 1)
        self.assertEqual(compile.call_args.args[0], stream.getvalue())
        self.assertIs(captured["object_codes"][0], fake_obj)
        self.assertEqual(captured["kind"], "cubin")

    def test_deferred_cu_uses_registration_lto(self):
        fake_obj = unittest.mock.Mock()
        linker = _Linker(max_registers=0, cc=(7, 5), lto=False)
        linker.add_cu("__device__ int foo() { return 1; }", "foo.cu")
        linker.lto = True

        with unittest.mock.patch.object(
            cuda_driver.nvrtc, "compile", return_value=(fake_obj, "")
        ) as compile:
            linker._materialize_pending_cu()

        self.assertFalse(compile.call_args.kwargs["ltoir"])

    def test_recreate_with_lto_materializes_pending_cu(self):
        fake_obj = unittest.mock.Mock()
        fake_obj.code_type = "unknown"
        linker = NumbaCudaMLIRLinker(cc=(7, 5))
        linker.add_cu("__device__ int foo() { return 1; }", "foo.cu")

        with unittest.mock.patch.object(
            cuda_driver.nvrtc, "compile", return_value=(fake_obj, "")
        ) as compile:
            recreated = linker.recreate_with_lto()

        self.assertEqual(compile.call_count, 1)
        self.assertEqual(linker._pending_cu, [])
        self.assertEqual(recreated._pending_cu, [])
        self.assertIs(recreated._object_codes[0], fake_obj)

    def test_recreate_with_lto_tracks_materialized_cu_ltoir(self):
        fake_obj = unittest.mock.Mock()
        fake_obj.code_type = "ltoir"
        fake_obj.code = b"ltoir"
        linker = NumbaCudaMLIRLinker(cc=(7, 5), lto=True)
        linker.add_cu("__device__ int foo() { return 1; }", "foo.cu")

        with unittest.mock.patch.object(cuda_driver.nvrtc, "compile", return_value=(fake_obj, "")):
            recreated = linker.recreate_with_lto()

        self.assertEqual(recreated._ltoirs, {hash(fake_obj.code): fake_obj.code})


if __name__ == "__main__":
    unittest.main()
