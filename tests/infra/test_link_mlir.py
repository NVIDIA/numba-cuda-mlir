# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.simt as cuda
from cuda.simt import types, compiler, testing
import ctypes
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.xfail
def test_link_mlir_pass_by_ptr():
    lib_path = Path(__file__).parent / "pass_by_ptr.mlir"
    lib = compiler.declare_mlir_library(lib_path)

    @cuda.jit(dump=True)
    def test_func(a, i):
        ap = ctypes.pointer(a)
        lib.pass_by_ptr(ap, i)

    a = np.array([1, 2, 3], dtype=np.int64)
    ad = cuda.to_device(a)
    test_func[1, 1](ad, 1)
    a = ad.copy_to_host()
    assert all(a == np.array([1, 0xDEADBEEF, 3], dtype=np.int64)), a

    cres = compiler.compile_result(test_func, "(int64[:], int32)")
    testing.filecheck_with_comments(cres.mlir_module_optimized)
    # CHECK:     llvm.func @_ZN26test_link_mlir_pass_by_ptr12_3clocals_3e9test_funcE5ArrayIxLi1E1A7mutable7alignedEi(
    # CHECK-SAME:      %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i32)
    # CHECK-NEXT:             %{{.*}} = llvm.mlir.constant(3735928559 : i64) : i64
    # CHECK-NEXT:             %{{.*}} = llvm.ptrtoint %{{.*}} : !llvm.ptr to i64
    # CHECK-NEXT:             %{{.*}} = llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
    # CHECK-NEXT:             %{{.*}} = llvm.sext %{{.*}} : i32 to i64
    # CHECK-NEXT:             %{{.*}} = llvm.getelementptr %{{.*}}{{\[}}%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    # CHECK-NEXT:             llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr
    # CHECK-NEXT:             llvm.return
    # CHECK-NEXT:           }


def test_link_mlir_memref():
    lib = compiler.declare_mlir_library(
        """
        module {
            func.func private @set_element(%arg0: memref<?x?xi64>, %arg1: i64, %arg2: i64, %arg3: i64) attributes {always_inline} {
                %i1 = arith.index_cast %arg2 : i64 to index
                %i2 = arith.index_cast %arg3 : i64 to index
                memref.store %arg1, %arg0[%i1, %i2] : memref<?x?xi64>
                func.return
            }
        }
        """
    )

    @cuda.jit(dump=False, opt_level=3, print_after_all=False)
    def test_func(a):
        lib.set_element(a, 0xDEADBEEF, 1, 2)

    a = np.ones((3, 3), dtype=np.int64)
    ad = cuda.to_device(a)
    test_func[1, 1](ad)
    a = ad.copy_to_host()
    expect = np.array([[1, 1, 1], [1, 1, 0xDEADBEEF], [1, 1, 1]])
    assert np.allclose(a, expect), f"{a=} != {expect=}"

    mlir_str = compiler.compile_mlir(test_func, "void(int64[:, :])", optimized=True)
    testing.filecheck_with_comments(mlir_str)
    # CHECK-LABEL:     llvm.func @_ZN21test_link_mlir_memref12_3clocals_3e9test_funcE5ArrayIxLi2E1A7mutable7alignedE(
    # CHECK-SAME:      %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64)
    # CHECK-NEXT:             %{{.*}} = llvm.mlir.constant(2 : i64) : i64
    # CHECK-NEXT:             %{{.*}} = llvm.mlir.constant(1 : i64) : i64
    # CHECK-NEXT:             %{{.*}} = llvm.mlir.constant(3735928559 : i64) : i64
    # CHECK-NEXT:             %{{.*}} = llvm.mul %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
    # CHECK-NEXT:             %{{.*}} = llvm.add %{{.*}}, %{{.*}} overflow<nsw, nuw> : i64
    # CHECK-NEXT:             %{{.*}} = llvm.getelementptr inbounds|nuw %{{.*}}{{\[}}%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    # CHECK-NEXT:             llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr
    # CHECK-NEXT:             llvm.return
    # CHECK-NEXT:           }


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    test_link_mlir_memref()
