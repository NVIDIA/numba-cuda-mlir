# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.cudadrv.linkable_code import LinkableCode
from numba_cuda_mlir.testing import NumbaCUDATestCase
from numba_cuda_mlir.numba_cuda.testing import skip_on_cudasim


class TestLinkableCode(NumbaCUDATestCase):
    @skip_on_cudasim(reason="Simulator does not support linkable code")
    @pytest.mark.numba_cuda_test_binaries("a", "cubin", "cu", "fatbin", "o", "ptx", "ltoir")
    def test_linkable_code_from_path_or_obj(self):
        binaries = self.numba_cuda_test_binaries
        files_kind = [
            (binaries.test_device_functions_a, cuda.Archive),
            (binaries.test_device_functions_cubin, cuda.Cubin),
            (binaries.test_device_functions_cu, cuda.CUSource),
            (binaries.test_device_functions_fatbin, cuda.Fatbin),
            (binaries.test_device_functions_o, cuda.Object),
            (binaries.test_device_functions_ptx, cuda.PTXSource),
            (binaries.test_device_functions_ltoir, cuda.LTOIR),
        ]

        for path, kind in files_kind:
            obj = LinkableCode.from_path_or_obj(path)
            assert isinstance(obj, kind)

            # test identity of from_path_or_obj
            obj2 = LinkableCode.from_path_or_obj(obj)
            assert obj2 is obj
