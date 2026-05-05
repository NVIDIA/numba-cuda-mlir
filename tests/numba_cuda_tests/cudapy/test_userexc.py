# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numba_cuda_mlir
from numba_cuda_mlir import cuda
from numba_cuda_mlir.testing import NumbaCUDATestCase
import pytest


class MyError(Exception):
    pass


regex_pattern = r'In function [\'"]test_exc[\'"], file [\:\.\/\\\-a-zA-Z_0-9]+, line \d+'


class TestUserExc(NumbaCUDATestCase):
    @pytest.mark.xfail(True, reason="ICE")
    def test_user_exception(self):
        @numba_cuda_mlir.cuda.jit("void(int32)", debug=True, opt=False)
        def test_exc(x):
            if x == 1:
                raise MyError
            elif x == 2:
                raise MyError("foo")

        test_exc[1, 1](0)  # no raise

        with self.assertRaises(MyError) as cm:
            test_exc[1, 1](1)

        self.assertRegex(str(cm.exception), regex_pattern)

        self.assertIn("tid=[0, 0, 0] ctaid=[0, 0, 0]", str(cm.exception))

        with self.assertRaises(MyError) as cm:
            test_exc[1, 1](2)

        self.assertRegex(str(cm.exception), regex_pattern)
        self.assertRegex(str(cm.exception), regex_pattern)
        self.assertIn("tid=[0, 0, 0] ctaid=[0, 0, 0]: foo", str(cm.exception))
