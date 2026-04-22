# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import os
import tempfile

import cuda.simt as cs


def test_profile_jit_prints_stats(capfd):
    @cs.jit(profile_jit=True)
    def kernel(arr):
        i = cs.grid(1)
        if i < arr.size:
            arr[i] = i

    arr = cs.device_array(10, dtype=np.int32)
    kernel[1, 10](arr)
    np.testing.assert_array_equal(arr.copy_to_host(), np.arange(10, dtype=np.int32))

    captured = capfd.readouterr()
    assert "cProfile for compilation of" in captured.err
    assert "cumtime" in captured.err


def test_profile_jit_saves_prof_file(capfd):
    with tempfile.TemporaryDirectory() as tmpdir:
        prof_path = os.path.join(tmpdir, "compile.prof")

        @cs.jit(profile_jit=prof_path)
        def kernel(arr):
            i = cs.grid(1)
            if i < arr.size:
                arr[i] = i

        arr = cs.device_array(10, dtype=np.int32)
        kernel[1, 10](arr)
        np.testing.assert_array_equal(arr.copy_to_host(), np.arange(10, dtype=np.int32))

        captured = capfd.readouterr()
        assert "cProfile for compilation of" in captured.err
        assert f"Profile saved to: {prof_path}" in captured.err
        assert os.path.isfile(prof_path)

        import pstats

        stats = pstats.Stats(prof_path)
        assert stats.total_calls > 0
