import pytest
import re
import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda.vector_types import vector_types_by_name

_EXPECTED_ALIGNMENTS = {
    "int8x1": 1, "int8x2": 2, "int8x3": 1, "int8x4": 4,
    "int16x1": 2, "int16x2": 4, "int16x3": 2, "int16x4": 8,
    "int32x1": 4, "int32x2": 8, "int32x3": 4, "int32x4": 16,
    "int64x1": 8, "int64x2": 16, "int64x3": 8, "int64x4": 16,
    "uint8x1": 1, "uint8x2": 2, "uint8x3": 1, "uint8x4": 4,
    "uint16x1": 2, "uint16x2": 4, "uint16x3": 2, "uint16x4": 8,
    "uint32x1": 4, "uint32x2": 8, "uint32x3": 4, "uint32x4": 16,
    "uint64x1": 8, "uint64x2": 16, "uint64x3": 8, "uint64x4": 16,
    "float16x1": 2, "float16x2": 4, "float16x3": 2, "float16x4": 8,
    "float32x1": 4, "float32x2": 8, "float32x3": 4, "float32x4": 16,
    "float64x1": 8, "float64x2": 16, "float64x3": 8,             "float64x4": 16,
    "double4_16a": 16,
    "double4_32a": 32,
    "long4_16a": 16,
    "long4_32a": 32,
    "ulong4_16a": 16,
    "ulong4_32a": 32,
    "longlong4_16a": 16,
    "longlong4_32a": 32,
    "ulonglong4_16a": 16,
    "ulonglong4_32a": 32,
}

@pytest.mark.parametrize("name,expected_align", _EXPECTED_ALIGNMENTS.items())
def test_vector_type_alignment_doc(name, expected_align):
    """Test that vector types match the alignment described in the doc."""
    if not hasattr(cuda, name):
        pytest.skip(f"{name} not available in cuda module")

    vec_type = getattr(cuda, name)

    @cuda.jit
    def func(dest):
        local_array = cuda.local.array(shape=1, dtype=vec_type)
        i = cuda.grid(1)
        if i == 0:
            dest[0] = local_array.ctypes.data

    dest = np.zeros(1, dtype=np.uint64)
    func[1, 1](dest)
    addr = dest[0]
    
    assert addr % expected_align == 0, f"{name} address {addr} is not aligned to {expected_align}"

    # Verify the allocation instruction in PTX
    sig = (cuda.types.Array(cuda.types.uint64, 1, 'C'),)
    ptx = func.inspect_ptx(sig)
    
    # For alignment of 1, PTX might not emit an explicit .align 1
    # or it might emit .align 1. Let's check if there is an alignment explicitly emitted
    if expected_align > 1:
        match = re.search(r'\.local \.align (\d+) \.b8', ptx)
        assert match is not None, f"Could not find local allocation in PTX for {name}"
        ptx_align = int(match.group(1))
        assert ptx_align >= expected_align and ptx_align % expected_align == 0, f"{name} PTX alignment {ptx_align} is not compatible with expected {expected_align}"


@pytest.mark.parametrize("vec_type_name, ptx_load_instr, ptx_store_instr", [
    ("float32x4", r'ld\.global\.v4\.f32', r'st\.global\.v4\.f32'),
    ("int32x4", r'ld\.global\.v4\.u32', r'st\.global\.v4\.u32'),
    ("float64x2", r'ld\.global\.v2\.f64', r'st\.global\.v2\.f64'),
    ("int64x2", r'ld\.global\.v2\.u64', r'st\.global\.v2\.u64'),
])
def test_vector_type_ptx_optimal_load_store(vec_type_name, ptx_load_instr, ptx_store_instr):
    """Test to confirm that optimal load/store instructions are used for vector types."""
    if vec_type_name not in vector_types_by_name:
        pytest.skip(f"{vec_type_name} not in vector_types_by_name")

    @cuda.jit(dump_ptx=True)
    def kernel(arr_in, arr_out):
        i = cuda.threadIdx.x
        arr_out[i] = arr_in[i]

    vec_type = getattr(cuda, vec_type_name)
    sig = (cuda.types.Array(vec_type, 1, 'C'), cuda.types.Array(vec_type, 1, 'C'))
    ptx = kernel.inspect_ptx(sig)

    # Verify optimal vectorized load/store
    assert re.search(ptx_load_instr, ptx) is not None, f"Optimal load not found for {vec_type_name}"
    assert re.search(ptx_store_instr, ptx) is not None, f"Optimal store not found for {vec_type_name}"


@pytest.mark.parametrize("name,expected_align", _EXPECTED_ALIGNMENTS.items())
def test_vector_type_param_alignment(name, expected_align):
    """Test that vector types passed by value have the correct parameter ABI alignment in PTX."""
    if not hasattr(cuda, name):
        pytest.skip(f"{name} not available in cuda module")

    vec_type = getattr(cuda, name)

    @cuda.jit
    def kernel_arg_test(val):
        pass

    sig = (vec_type,)
    ptx = kernel_arg_test.inspect_ptx(sig)

    if expected_align > 1:
        match = re.search(r'\.param \.align (\d+)', ptx)
        assert match is not None, f"Could not find parameter alignment in PTX for {name}"
        ptx_align = int(match.group(1))
        assert ptx_align >= expected_align and ptx_align % expected_align == 0, f"{name} PTX parameter alignment {ptx_align} is not compatible with expected {expected_align}"
