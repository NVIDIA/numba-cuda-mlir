# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for annotations_as_signatures feature.

When annotations_as_signatures=True (default for cuda.simt), type annotations
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

import cuda.simt as cs
from cusimt import types
from cusimt.testing import filecheck


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

    def test_cuda_simt_default_is_true(self):
        """cuda.simt.jit has annotations_as_signatures=True by default."""

        @cs.jit
        def kernel(result: types.Array(types.int32, 1, "A"), x: types.int32):
            result[0] = x * 2

        result = cs.device_array(1, dtype=np.int32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42

    def test_default_false_uses_lazy_compilation(self):
        """annotations_as_signatures=False uses lazy compilation (subprocess)."""
        code = """
            import cuda.simt as cs
            import numpy as np

            # No annotations - should work with lazy compilation
            @cs.jit(annotations_as_signatures=False)
            def kernel(result, x):
                result[0] = x * 2

            result = cs.device_array(1, dtype=np.int32)
            kernel[1, 1](result, 21)
            assert result.copy_to_host()[0] == 42
            print("OK")
        """
        rc, stdout, stderr = _run_in_subprocess(code)
        assert rc == 0, f"subprocess failed: {stderr}"
        assert "OK" in stdout

    def test_missing_annotations_falls_back_to_lazy(self):
        """When annotations_as_signatures=True but annotations missing, use lazy compilation."""

        @cs.jit
        def kernel(result, x):  # No annotations - falls back to lazy compilation
            result[0] = x * 2

        result = cs.device_array(1, dtype=np.int32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42

    def test_explicit_false_uses_lazy_compilation(self):
        """Setting annotations_as_signatures=False explicitly uses lazy compilation."""

        # Even with annotations, explicit False means lazy compilation
        @cs.jit(annotations_as_signatures=False)
        def kernel(result: types.Array(types.int32, 1, "A"), x: types.int32):
            result[0] = x * 2

        result = cs.device_array(1, dtype=np.int32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42

    def test_explicit_true_with_numba_cuda(self):
        """Setting annotations_as_signatures=True explicitly with cuda.simt (subprocess)."""
        code = """
            import cuda.simt as cs
            from cusimt import types
            import numpy as np

            @cs.jit(annotations_as_signatures=True)
            def kernel(result: types.Array(types.int32, 1, "A"), x: types.int32):
                result[0] = x * 2

            result = cs.device_array(1, dtype=np.int32)
            kernel[1, 1](result, 21)
            assert result.copy_to_host()[0] == 42
            print("OK")
        """
        rc, stdout, stderr = _run_in_subprocess(code)
        assert rc == 0, f"subprocess failed: {stderr}"
        assert "OK" in stdout

    def test_partial_annotations_uses_lazy_compilation(self):
        """Partial annotations (some params annotated, some not) use lazy compilation."""

        @cs.jit
        def kernel(result: types.Array(types.int32, 1, "A"), x):  # x is a template
            result[0] = x * 2

        result = cs.device_array(1, dtype=np.int32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42

    def test_device_function_with_annotations(self):
        """Device functions also respect annotations_as_signatures."""

        @cs.jit(device=True)
        def device_func(x: types.float64) -> types.float64:
            return x * 2.0

        @cs.jit
        def kernel(result: types.Array(types.float64, 1, "A")):
            result[0] = device_func(21.0)

        result = cs.device_array(1, dtype=np.float64)
        kernel[1, 1](result)
        assert result.copy_to_host()[0] == 42.0

    def test_return_type_annotation(self):
        """Return type annotations are respected in binding mode."""

        @cs.jit(device=True)
        def add(a: types.int32, b: types.int32) -> types.int32:
            return a + b

        @cs.jit
        def kernel(result: types.Array(types.int32, 1, "A")):
            result[0] = add(20, 22)

        result = cs.device_array(1, dtype=np.int32)
        kernel[1, 1](result)
        assert result.copy_to_host()[0] == 42


class TestExplicitSignatures:
    """Tests for explicit signature handling."""

    def test_explicit_signature_without_annotations(self):
        """Explicit signature works when no annotations are present."""

        @cs.jit("void(int32[:], int32)")
        def kernel(result, x):  # No annotations
            result[0] = x * 2

        result = cs.device_array(1, dtype=np.int32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42

    def test_explicit_signature_on_parameterless_kernel(self):
        """Explicit signature on a parameterless kernel must not be reported as
        conflicting with annotations (regression test for issue #92)."""

        @cs.jit("void()")
        def kernel():
            pass

        kernel[1, 1]()

    def test_conflicting_signature_and_annotations_raises_error(self):
        """Both explicit signature and annotations raises TypeError."""
        with pytest.raises(TypeError, match="Conflicting signature sources"):

            @cs.jit("void(int32[:], int32)")
            def kernel(result: types.Array(types.int32, 1, "A"), x: types.int32):
                result[0] = x * 2

    def test_explicit_signature_with_annotations_allowed_when_disabled(self):
        """Explicit signature with annotations works when annotations_as_signatures=False."""

        # With annotations_as_signatures=False, annotations are ignored
        @cs.jit("void(int32[:], int32)", annotations_as_signatures=False)
        def kernel(result: types.Array(types.int32, 1, "A"), x: types.int32):
            result[0] = x * 2

        result = cs.device_array(1, dtype=np.int32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42

    def test_signature_infer_with_annotations(self):
        """signature='infer' extracts signature from annotations (subprocess)."""
        code = """
            import cuda.simt as cs
            from cusimt import types
            import numpy as np

            @cs.jit(signature="infer")
            def kernel(result: types.Array(types.int32, 1, "A"), x: types.int32):
                result[0] = x * 2

            result = cs.device_array(1, dtype=np.int32)
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

        @cs.jit("void(int32[:], int32)")
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

        result = cs.device_array(1, dtype=np.int32)
        kernel[1, 1](result, np.int64(21))
        assert result.copy_to_host()[0] == 42

        assert len(kernel.overloads) == 1
        post_compiled = next(iter(kernel.overloads.values()))
        assert pre_compiled is post_compiled

    def test_float_coercion_reuses_overload(self):
        """float64 runtime value is coerced to float32, reusing pre-compiled overload."""

        @cs.jit("void(float32[:], float32)")
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

        result = cs.device_array(1, dtype=np.float32)
        kernel[1, 1](result, np.float64(21.0))
        assert result.copy_to_host()[0] == 42.0

        assert len(kernel.overloads) == 1
        post_compiled = next(iter(kernel.overloads.values()))
        assert pre_compiled is post_compiled

    def test_int_to_float_coercion_reuses_overload(self):
        """int runtime value is coerced to float32, reusing pre-compiled overload."""

        @cs.jit("void(float32[:], float32)")
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

        result = cs.device_array(1, dtype=np.float32)
        kernel[1, 1](result, 21)
        assert result.copy_to_host()[0] == 42.0

        assert len(kernel.overloads) == 1
        post_compiled = next(iter(kernel.overloads.values()))
        assert pre_compiled is post_compiled

    def test_no_matching_overload_raises(self):
        """TypeError raised when no overload matches runtime types."""

        @cs.jit("void(int32[:], int32)")
        def kernel(result, x):
            result[0] = x * 2

        result = cs.device_array(1, dtype=np.float32)
        with pytest.raises(TypeError, match="No matching definition"):
            kernel[1, 1](result, 21)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
