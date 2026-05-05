# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for annotations_as_signatures feature.

When annotations_as_signatures=True (default for numba_cuda_mlir), type annotations
on kernel parameters define the signature and are pre-compiled at decoration
time. If annotations are missing or incomplete, falls back to lazy compilation.

When annotations_as_signatures=False, always uses lazy compilation regardless
of annotations.
"""

import numpy as np
import pytest
import subprocess
import sys
import textwrap

from numba_cuda_mlir import cuda
from numba_cuda_mlir import types
from numba_cuda_mlir.testing import filecheck


def _run_in_subprocess(code: str):
    """Run Python code in a subprocess and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


class TestAnnotationsAsSignatures:
    """Tests for annotations_as_signatures option."""

    def test_default_is_true(self):
        """cuda.jit has annotations_as_signatures=True by default."""

        @cuda.jit
        def kernel(result: types.Array(types.int32, 1, "A"), x: types.int32):
            result[0] = x * 2

        result = cuda.device_array(1, dtype=np.int32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42

    def test_default_false_uses_lazy_compilation(self):
        """annotations_as_signatures=False uses lazy compilation (subprocess)."""
        code = """
            from numba_cuda_mlir import cuda
            import numpy as np

            # No annotations - should work with lazy compilation
            @cuda.jit(annotations_as_signatures=False)
            def kernel(result, x):
                result[0] = x * 2

            result = cuda.device_array(1, dtype=np.int32)
            kernel[1, 1](result, 21)
            assert result.copy_to_host()[0] == 42
            print("OK")
        """
        rc, stdout, stderr = _run_in_subprocess(code)
        assert rc == 0, f"subprocess failed: {stderr}"
        assert "OK" in stdout

    def test_missing_annotations_falls_back_to_lazy(self):
        """When annotations_as_signatures=True but annotations missing, use lazy compilation."""

        @cuda.jit
        def kernel(result, x):  # No annotations - falls back to lazy compilation
            result[0] = x * 2

        result = cuda.device_array(1, dtype=np.int32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42

    def test_explicit_false_uses_lazy_compilation(self):
        """Setting annotations_as_signatures=False explicitly uses lazy compilation."""

        # Even with annotations, explicit False means lazy compilation
        @cuda.jit(annotations_as_signatures=False)
        def kernel(result: types.Array(types.int32, 1, "A"), x: types.int32):
            result[0] = x * 2

        result = cuda.device_array(1, dtype=np.int32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42

    def test_explicit_true_with_numba_cuda(self):
        """Setting annotations_as_signatures=True explicitly with numba_cuda_mlir (subprocess)."""
        code = """
            from numba_cuda_mlir import cuda
            from numba_cuda_mlir import types
            import numpy as np

            @cuda.jit(annotations_as_signatures=True)
            def kernel(result: types.Array(types.int32, 1, "A"), x: types.int32):
                result[0] = x * 2

            result = cuda.device_array(1, dtype=np.int32)
            kernel[1, 1](result, 21)
            assert result.copy_to_host()[0] == 42
            print("OK")
        """
        rc, stdout, stderr = _run_in_subprocess(code)
        assert rc == 0, f"subprocess failed: {stderr}"
        assert "OK" in stdout

    def test_partial_annotations_uses_lazy_compilation(self):
        """Partial annotations (some params annotated, some not) use lazy compilation."""

        @cuda.jit
        def kernel(result: types.Array(types.int32, 1, "A"), x):  # x is a template
            result[0] = x * 2

        result = cuda.device_array(1, dtype=np.int32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42

    def test_device_function_with_annotations(self):
        """Device functions also respect annotations_as_signatures."""

        @cuda.jit(device=True)
        def device_func(x: types.float64) -> types.float64:
            return x * 2.0

        @cuda.jit
        def kernel(result: types.Array(types.float64, 1, "A")):
            result[0] = device_func(21.0)

        result = cuda.device_array(1, dtype=np.float64)
        kernel[1, 1](result)
        assert result.copy_to_host()[0] == 42.0

    def test_return_type_annotation(self):
        """Return type annotations are respected in binding mode."""

        @cuda.jit(device=True)
        def add(a: types.int32, b: types.int32) -> types.int32:
            return a + b

        @cuda.jit
        def kernel(result: types.Array(types.int32, 1, "A")):
            result[0] = add(20, 22)

        result = cuda.device_array(1, dtype=np.int32)
        kernel[1, 1](result)
        assert result.copy_to_host()[0] == 42


class TestExplicitSignatures:
    """Tests for explicit signature handling."""

    def test_explicit_signature_without_annotations(self):
        """Explicit signature works when no annotations are present."""

        @cuda.jit("void(int32[:], int32)")
        def kernel(result, x):  # No annotations
            result[0] = x * 2

        result = cuda.device_array(1, dtype=np.int32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42

    def test_explicit_signature_on_parameterless_kernel(self):
        """Explicit signature on a parameterless kernel must not be reported as
        conflicting with annotations (regression test for issue #92)."""

        @cuda.jit("void()")
        def kernel():
            pass

        kernel[1, 1]()

    def test_conflicting_signature_and_annotations_raises_error(self):
        """Both explicit signature and annotations raises TypeError."""
        with pytest.raises(TypeError, match="Conflicting signature sources"):

            @cuda.jit("void(int32[:], int32)")
            def kernel(result: types.Array(types.int32, 1, "A"), x: types.int32):
                result[0] = x * 2

    def test_explicit_signature_with_annotations_allowed_when_disabled(self):
        """Explicit signature with annotations works when annotations_as_signatures=False."""

        # With annotations_as_signatures=False, annotations are ignored
        @cuda.jit("void(int32[:], int32)", annotations_as_signatures=False)
        def kernel(result: types.Array(types.int32, 1, "A"), x: types.int32):
            result[0] = x * 2

        result = cuda.device_array(1, dtype=np.int32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42

    def test_signature_infer_with_annotations(self):
        """signature='infer' extracts signature from annotations (subprocess)."""
        code = """
            from numba_cuda_mlir import cuda
            from numba_cuda_mlir import types
            import numpy as np

            @cuda.jit(signature="infer")
            def kernel(result: types.Array(types.int32, 1, "A"), x: types.int32):
                result[0] = x * 2

            result = cuda.device_array(1, dtype=np.int32)
            kernel[1, 1](result, 21)
            assert result.copy_to_host()[0] == 42
            print("OK")
        """
        rc, stdout, stderr = _run_in_subprocess(code)
        assert rc == 0, f"subprocess failed: {stderr}"
        assert "OK" in stdout


class TestTypeCoercion:
    """Tests for type coercion with explicit signatures (compilation disabled)."""

    def test_int_coercion_reuses_overload(self):
        """int64 runtime value is coerced to int32, reusing pre-compiled overload."""

        @cuda.jit("void(int32[:], int32)")
        def kernel(result, x):
            result[0] = x * 2

        mlir = next(iter(kernel.inspect_mlir().values()))
        filecheck(
            """
            CHECK: gpu.func @{{.*}}(%arg0: memref<?xi32{{.*}}>, %arg1: i32)
            """,
            mlir,
        )

        assert len(kernel.overloads) == 1
        pre_compiled = next(iter(kernel.overloads.values()))

        result = cuda.device_array(1, dtype=np.int32)
        kernel[1, 1](result, np.int64(21))
        assert result.copy_to_host()[0] == 42

        assert len(kernel.overloads) == 1
        post_compiled = next(iter(kernel.overloads.values()))
        assert pre_compiled is post_compiled

    def test_float_coercion_reuses_overload(self):
        """float64 runtime value is coerced to float32, reusing pre-compiled overload."""

        @cuda.jit("void(float32[:], float32)")
        def kernel(result, x):
            result[0] = x * 2.0

        mlir = next(iter(kernel.inspect_mlir().values()))
        filecheck(
            """
            CHECK: gpu.func @{{.*}}(%arg0: memref<?xf32{{.*}}>, %arg1: f32)
            """,
            mlir,
        )

        assert len(kernel.overloads) == 1
        pre_compiled = next(iter(kernel.overloads.values()))

        result = cuda.device_array(1, dtype=np.float32)
        kernel[1, 1](result, np.float64(21.0))
        assert result.copy_to_host()[0] == 42.0

        assert len(kernel.overloads) == 1
        post_compiled = next(iter(kernel.overloads.values()))
        assert pre_compiled is post_compiled

    def test_int_to_float_coercion_reuses_overload(self):
        """int runtime value is coerced to float32, reusing pre-compiled overload."""

        @cuda.jit("void(float32[:], float32)")
        def kernel(result, x):
            result[0] = x * 2.0

        mlir = next(iter(kernel.inspect_mlir().values()))
        filecheck(
            """
            CHECK: gpu.func @{{.*}}(%arg0: memref<?xf32{{.*}}>, %arg1: f32)
            """,
            mlir,
        )

        assert len(kernel.overloads) == 1
        pre_compiled = next(iter(kernel.overloads.values()))

        result = cuda.device_array(1, dtype=np.float32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42.0

        assert len(kernel.overloads) == 1
        post_compiled = next(iter(kernel.overloads.values()))
        assert pre_compiled is post_compiled

    def test_no_matching_overload_raises(self):
        """TypeError raised when no overload matches runtime types."""

        @cuda.jit("void(int32[:], int32)")
        def kernel(result, x):
            result[0] = x * 2

        result = cuda.device_array(1, dtype=np.float32)
        with pytest.raises(TypeError, match="No matching definition"):
            kernel[1, 1](result, 21)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
