# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

from numba_cuda_mlir.numba_cuda.cudadrv import nvrtc, nvvm
from numba_cuda_mlir.numba_cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM


class TestNvvmDriver(unittest.TestCase):
    def get_nvvmir(self):
        versions = NVVM().get_ir_version()
        data_layout = NVVM().data_layout
        return nvvmir_generic.format(data_layout=data_layout, v=versions)

    def test_nvvm_compile_simple(self):
        nvvmir = self.get_nvvmir()
        ptx = nvvm.compile_ir(nvvmir).decode("utf8")
        self.assertTrue("simple" in ptx)
        self.assertTrue("ave" in ptx)

    def test_nvvm_compile_nullary_option(self):
        # Tests compilation with an option that doesn't take an argument
        # ("-gen-lto") - all other NVVM options are of the form
        # "-<name>=<value>"

        nvvmir = self.get_nvvmir()
        arch = "compute_%d%d" % nvrtc.get_lowest_supported_cc()
        ltoir = nvvm.compile_ir(nvvmir, opt=3, gen_lto=None, arch=arch)

        # Verify we correctly passed the option by checking if we got LTOIR
        # from NVVM (by looking for the expected magic number for LTOIR)
        self.assertEqual(ltoir[:4], b"\xed\x43\x4e\x7f")

    def test_nvvm_bad_option(self):
        # Ensure that unsupported / non-existent options are reported as such
        # to the user / caller
        msg = "-made-up-option=2 is an unsupported option"
        with self.assertRaisesRegex(NvvmError, msg):
            nvvm.compile_ir("", made_up_option=2)

    def _test_nvvm_support(self, arch):
        compute_xx = "compute_{0}{1}".format(*arch)
        nvvmir = self.get_nvvmir()
        ptx = nvvm.compile_ir(nvvmir, arch=compute_xx, ftz=1, prec_sqrt=0, prec_div=0).decode(
            "utf8"
        )
        self.assertIn(".target sm_{0}{1}".format(*arch), ptx)
        self.assertIn("simple", ptx)
        self.assertIn("ave", ptx)

    def test_nvvm_support(self):
        """Test supported CC by NVVM"""
        for arch in nvrtc.get_supported_ccs():
            self._test_nvvm_support(arch=arch)


class TestLibDevice(unittest.TestCase):
    def test_libdevice_load(self):
        # Test that constructing LibDevice gives a bitcode file
        libdevice = LibDevice()
        self.assertEqual(libdevice.bc[:4], b"BC\xc0\xde")


nvvmir_generic = """\
target triple="nvptx64-nvidia-cuda"
target datalayout = "{data_layout}"

define i32 @ave(i32 %a, i32 %b) {{
entry:
%add = add nsw i32 %a, %b
%div = sdiv i32 %add, 2
ret i32 %div
}}

define void @simple(i32* %data) {{
entry:
%0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%mul = mul i32 %0, %1
%2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%add = add i32 %mul, %2
%call = call i32 @ave(i32 %add, i32 %add)
%idxprom = sext i32 %add to i64
%arrayidx = getelementptr inbounds i32, i32* %data, i64 %idxprom
store i32 %call, i32* %arrayidx, align 4
ret void
}}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

!nvvmir.version = !{{!1}}
!1 = !{{i32 {v[0]}, i32 {v[1]}, i32 {v[2]}, i32 {v[3]}}}

!nvvm.annotations = !{{!2}}
!2 = !{{void (i32*)* @simple, !"kernel", i32 1}}

@"llvm.used" = appending global [1 x i8*] [i8* bitcast (void (i32*)* @simple to i8*)], section "llvm.metadata"
"""  # noqa: E501


if __name__ == "__main__":
    unittest.main()
