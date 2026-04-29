# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.simt as cs
from cuda.simt import compiler, testing
import numpy as np
import pytest


def test_numba_issue_9324():
    # https://github.com/numba/numba/issues/9324
    @cs.jit(dump=True)
    def f_gpu(array):
        i = cs.grid(1)

        array1 = array[:, 1]
        array0 = array[:, 0]

        if i < array.shape[0]:
            array1[i] += 1.0
            array1[i] = array0[i]

    array_cpu = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    f_gpu.forall(array_cpu.shape[0])(array_cpu)
    print(array_cpu)
    assert np.allclose(array_cpu, np.array([[1, 1, 3], [4, 4, 6]]))


INCOMPLETE_SLICE_CASES = (
    (
        [5, 2, 4],
        np.array(
            [
                [[1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1]],
                [[5, 5, 5, 5], [5, 5, 5, 5]],
                [[1, 1, 1, 1], [1, 1, 1, 1]],
            ],
            dtype=np.int32,
        ),
    ),
    ([4, 1, 2], np.array([[[1, 1]], [[1, 1]], [[1, 1]], [[5, 5]]], dtype=np.int32)),
)


@pytest.mark.parametrize("shape,answer", INCOMPLETE_SLICE_CASES)
def test_incomplete_slice(shape, answer):
    shape = tuple(shape)

    @cs.jit(dump=True, print_after_all=False)
    def k(array: cs.DeviceNDArray):
        array[3] = 5

    h = np.ones(shape, dtype=np.int32)
    d = cs.to_device(h)
    k[1, 1](d)
    assert np.allclose(d.copy_to_host(), answer), f"Expected {answer}, got {d.copy_to_host()}"

    # CHECK-LABEL: gpu.func
    # CHECK-SAME: (%[[ARG:.+]]: memref
    # CHECK-SAME: kernel
    # CHECK: scf.forall
    # CHECK-NEXT: memref.store %{{.+}}, %[[ARG]]

    cres = compiler.compile_for(k, d)
    mlir = cres.mlir_module_str
    testing.filecheck_with_comments(mlir)


if __name__ == "__main__":
    test_incomplete_slice(*INCOMPLETE_SLICE_CASES[0])
