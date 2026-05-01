# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda import get_current_device, config
from numba_cuda_mlir.numba_cuda.cudadrv.driver import _Linker, _have_nvjitlink
from numba_cuda_mlir.testing import NumbaCUDATestCase
from numba_cuda_mlir.numba_cuda.testing import skip_on_cudasim

import os
import io
import contextlib


@unittest.skipIf(not _have_nvjitlink(), "nvJitLink not installed or new enough (>12.3)")
@skip_on_cudasim("Linking unsupported in the simulator")
class TestLinker(NumbaCUDATestCase):
    @pytest.mark.numba_cuda_test_binaries("a", "cubin", "cu", "fatbin", "o", "ptx")
    def test_nvjitlink_add_file_guess_ext_linkable_code(self):
        binaries = self.numba_cuda_test_binaries
        files = (
            binaries.test_device_functions_a,
            binaries.test_device_functions_cubin,
            binaries.test_device_functions_cu,
            binaries.test_device_functions_fatbin,
            binaries.test_device_functions_o,
            binaries.test_device_functions_ptx,
        )
        for file in files:
            with self.subTest(file=file):
                linker = _Linker(cc=get_current_device().compute_capability)
                linker.add_file_guess_ext(file)

    @pytest.mark.numba_cuda_test_binaries("cubin")
    def test_nvjitlink_test_add_file_guess_ext_invalid_input(self):
        with open(self.numba_cuda_test_binaries.test_device_functions_cubin, "rb") as f:
            content = f.read()

        linker = _Linker(cc=get_current_device().compute_capability)
        with self.assertRaisesRegex(
            TypeError, "Expected path to file or a LinkableCode"
        ):
            # Feeding raw data as bytes to add_file_guess_ext should raise,
            # because there's no way to know what kind of file to treat it as
            linker.add_file_guess_ext(content)

    @pytest.mark.numba_cuda_test_binaries("a", "cubin", "cu", "fatbin", "o", "ptx")
    def test_nvjitlink_jit_with_linkable_code(self):
        binaries = self.numba_cuda_test_binaries
        files = (
            binaries.test_device_functions_a,
            binaries.test_device_functions_cubin,
            binaries.test_device_functions_cu,
            binaries.test_device_functions_fatbin,
            binaries.test_device_functions_o,
            binaries.test_device_functions_ptx,
        )
        for lto in [True, False]:
            for file in files:
                with self.subTest(file=file):
                    sig = "uint32(uint32, uint32)"
                    add_from_numba = cuda.declare_device("add_from_numba", sig)

                    @cuda.jit(link=[file], lto=lto)
                    def kernel(result):
                        result[0] = add_from_numba(1, 2)

                    result = cuda.device_array(1)
                    kernel[1, 1](result)
                    assert result[0] == 3

    @pytest.mark.numba_cuda_test_binaries("cubin")
    def test_nvjitlink_jit_with_invalid_linkable_code(self):
        with open(self.numba_cuda_test_binaries.test_device_functions_cubin, "rb") as f:
            content = f.read()
        with self.assertRaisesRegex(
            TypeError, "Expected path to file or a LinkableCode"
        ):

            @cuda.jit("void()", link=[content])
            def kernel():
                pass


@unittest.skipIf(not _have_nvjitlink(), "nvJitLink not installed or new enough (>12.3)")
@skip_on_cudasim("Linking unsupported in the simulator")
class TestLinkerDumpAssembly(NumbaCUDATestCase):
    def setUp(self):
        super().setUp()
        self._prev_dump_assembly = config.DUMP_ASSEMBLY
        config.DUMP_ASSEMBLY = True

    def tearDown(self):
        config.DUMP_ASSEMBLY = self._prev_dump_assembly
        super().tearDown()

    @pytest.mark.numba_cuda_test_binaries("cu", "ltoir", "fatbin_multi")
    @pytest.mark.skip(reason="sporadic CI failures")
    def test_nvjitlink_jit_with_linkable_code_lto_dump_assembly(self):
        binaries = self.numba_cuda_test_binaries
        files = (
            binaries.test_device_functions_cu,
            binaries.test_device_functions_ltoir,
            binaries.test_device_functions_fatbin_multi,
        )

        for file in files:
            with self.subTest(file=file):
                if (
                    file in binaries.require_cuobjdump
                    and os.getenv("NUMBA_CUDA_MLIR_TEST_WHEEL_ONLY") is not None
                ):
                    self.skipTest("wheel-only environments do not have cuobjdump")

                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    sig = "uint32(uint32, uint32)"
                    add_from_numba = cuda.declare_device("add_from_numba", sig)

                    @cuda.jit(link=[file], lto=True)
                    def kernel(result):
                        result[0] = add_from_numba(1, 2)

                    result = cuda.device_array(1)
                    kernel[1, 1](result)
                    assert result[0] == 3

                self.assertTrue("ASSEMBLY (AFTER LTO)" in f.getvalue())

    @pytest.mark.numba_cuda_test_binaries("a", "cubin", "fatbin", "o", "ptx")
    def test_nvjitlink_jit_with_linkable_code_lto_dump_assembly_warn(self):
        binaries = self.numba_cuda_test_binaries
        files = (
            binaries.test_device_functions_a,
            binaries.test_device_functions_cubin,
            binaries.test_device_functions_fatbin,
            binaries.test_device_functions_o,
            binaries.test_device_functions_ptx,
        )

        for file in files:
            with self.subTest(file=file):
                if (
                    file in binaries.require_cuobjdump
                    and os.getenv("NUMBA_CUDA_MLIR_TEST_WHEEL_ONLY") is not None
                ):
                    self.skipTest("wheel-only environments do not have cuobjdump")

                sig = "uint32(uint32, uint32)"
                add_from_numba = cuda.declare_device("add_from_numba", sig)

                @cuda.jit(link=[file], lto=True)
                def kernel(result):
                    result[0] = add_from_numba(1, 2)

                result = cuda.device_array(1)
                func = kernel[1, 1]
                with pytest.warns(
                    UserWarning,
                    match="it is not optimizable at link time, and `ignore_nonlto == True`",
                ):
                    func(result)
                assert result[0] == 3


if __name__ == "__main__":
    unittest.main()
