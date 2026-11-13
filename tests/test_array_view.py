# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from numba_cuda_mlir import cuda


def _make_copy_kernel_1d(target_dtype):
    def k(inp, out):
        v = inp.view(target_dtype)
        for i in range(v.shape[0]):
            out[i] = v[i]

    return k


def _make_copy_kernel_2d(target_dtype):
    def k(inp, out):
        v = inp.view(target_dtype)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                out[i, j] = v[i, j]

    return k


def _make_copy_kernel_3d(target_dtype):
    def k(inp, out):
        v = inp.view(target_dtype)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                for m in range(v.shape[2]):
                    out[i, j, m] = v[i, j, m]

    return k


_COPY_KERNEL_FACTORIES = {1: _make_copy_kernel_1d, 2: _make_copy_kernel_2d, 3: _make_copy_kernel_3d}


def _expected_view_F(arr_F, target_dtype):
    old_size = arr_F.dtype.itemsize
    new_size = np.dtype(target_dtype).itemsize
    flat = np.frombuffer(arr_F.tobytes(order="F"), dtype=target_dtype).copy()
    new_shape = list(arr_F.shape)
    if new_size <= old_size:
        new_shape[0] = arr_F.shape[0] * (old_size // new_size)
    else:
        new_shape[0] = (arr_F.shape[0] * old_size) // new_size
    return flat.reshape(new_shape, order="F")


def _run_view_copy(src, target_dtype, expect):
    jit_kernel = cuda.jit(_COPY_KERNEL_FACTORIES[expect.ndim](target_dtype), dump=False)
    d_in = cuda.to_device(src)
    d_out = cuda.to_device(np.zeros(expect.shape, dtype=target_dtype))
    jit_kernel[1, 1](d_in, d_out)
    np.testing.assert_array_equal(d_out.copy_to_host(), expect)


@pytest.mark.parametrize(
    "arr",
    [np.arange(8, dtype=np.int32), np.arange(24, dtype=np.int64).reshape(2, 3, 4)],
    ids=["i32_1d", "i64_3d"],
)
def test_view_same_dtype_is_identity(arr):
    _run_view_copy(arr, arr.dtype.type, arr.copy())


@pytest.mark.parametrize(
    "src, target_dtype",
    [
        (np.arange(-4, 4, dtype=np.int32), np.float32),
        (np.arange(8, dtype=np.uint16), np.int16),
        (np.linspace(-1.0, 1.0, 8, dtype=np.float64), np.int64),
    ],
    ids=["i32_to_f32", "u16_to_i16", "f64_to_i64"],
)
def test_view_same_width_bitcast_1d(src, target_dtype):
    _run_view_copy(src, target_dtype, src.view(target_dtype))


def test_view_same_width_bitcast_3d():
    src = np.arange(-4, 4, dtype=np.int32).reshape(2, 2, 2)
    _run_view_copy(src, np.float32, src.view(np.float32))


@pytest.mark.parametrize(
    "src, target_dtype",
    [
        (np.arange(8, dtype=np.int64), np.int8),
        (np.arange(32, dtype=np.int8), np.int32),
        (np.linspace(-2.0, 2.0, 8, dtype=np.float64), np.float32),
    ],
    ids=["i64_to_i8", "i8_to_i32", "f64_to_f32"],
)
def test_view_different_width_1d(src, target_dtype):
    _run_view_copy(src, target_dtype, src.view(target_dtype))


@pytest.mark.parametrize(
    "src_shape, src_dtype, target_dtype",
    [((2, 4), np.int64, np.int32), ((2, 3, 4), np.int32, np.int64)],
    ids=["C_i64_to_i32_2d", "C_i32_to_i64_3d"],
)
def test_view_different_width_C_contig(src_shape, src_dtype, target_dtype):
    src = np.arange(int(np.prod(src_shape)), dtype=src_dtype).reshape(src_shape)
    assert src.flags.c_contiguous
    _run_view_copy(src, target_dtype, src.view(target_dtype))


@pytest.mark.parametrize(
    "src_shape, src_dtype, target_dtype",
    [((2, 4), np.int64, np.int32), ((2, 3, 4), np.int32, np.int64)],
    ids=["F_i64_to_i32_2d", "F_i32_to_i64_3d"],
)
def test_view_different_width_F_contig(src_shape, src_dtype, target_dtype):
    src = np.asfortranarray(np.arange(int(np.prod(src_shape)), dtype=src_dtype).reshape(src_shape))
    assert src.flags.f_contiguous
    _run_view_copy(src, target_dtype, _expected_view_F(src, target_dtype))


def test_view_bitcast_preserves_special_floats():
    src = np.array([-1.0, -0.0, np.nan, np.inf, -np.inf, 1e-30, -1e30, 0.0], dtype=np.float32)
    _run_view_copy(src, np.int32, src.view(np.int32))


def test_view_after_slice_1d():
    @cuda.jit(dump=False)
    def k(byte_arr, start, stop, out):
        v = byte_arr[start:stop].view(np.int32)
        out[0] = v[0]

    host = np.array(range(10), dtype=np.uint64)
    d = cuda.to_device(host)
    out = cuda.to_device(np.zeros(1, dtype=np.int32))
    k[1, 1](d, 1, 3, out)
    assert out.copy_to_host()[0] == host[1:3].view(np.int32)[0]


def test_view_after_slice_2d_trailing_contig():
    @cuda.jit(dump=False)
    def k(inp, out):
        v = inp[::2, :].view(np.int32)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                out[i, j] = v[i, j]

    src = np.arange(32, dtype=np.int64).reshape(8, 4)
    expected = src[::2, :].view(np.int32)
    d_in = cuda.to_device(src)
    d_out = cuda.to_device(np.zeros(expected.shape, dtype=np.int32))
    k[1, 1](d_in, d_out)
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_view_u16_to_f16_bitcast():
    src = np.array([0x3C00, 0xBC00, 0x0000, 0x4100], dtype=np.uint16)
    _run_view_copy(src, np.float16, src.view(np.float16))


def test_view_f16_widening_to_i32():
    src = np.array([1.0, -1.0, 0.0, 2.5], dtype=np.float16)
    _run_view_copy(src, np.int32, src.view(np.int32))


def test_view_of_view_changes_dtype_twice():
    @cuda.jit(dump=False)
    def k(inp, out):
        v = inp.view(np.uint16).view(np.float32)
        for i in range(v.shape[0]):
            out[i] = v[i]

    src = np.array([1.0, -2.5, 3.25, 0.0], dtype=np.float32)
    d_in = cuda.to_device(src)
    d_out = cuda.to_device(np.zeros(4, dtype=np.float32))
    k[1, 1](d_in, d_out)
    np.testing.assert_array_equal(d_out.copy_to_host(), src)


def test_view_writes_propagate_to_source():
    @cuda.jit(dump=False)
    def k(arr):
        v = arr.view(np.int32)
        for i in range(v.shape[0]):
            v[i] = -1

    d = cuda.to_device(np.zeros(2, dtype=np.int64))
    k[1, 1](d)
    np.testing.assert_array_equal(d.copy_to_host(), np.full(2, -1, dtype=np.int64))


def test_view_writes_partial_overwrite():
    @cuda.jit(dump=False)
    def k(arr):
        v = arr.view(np.int8)
        for i in range(4):
            v[i] = 0x7F

    d = cuda.to_device(np.zeros(2, dtype=np.int32))
    k[1, 1](d)
    result = d.copy_to_host()
    assert result[0] == 0x7F7F7F7F
    assert result[1] == 0


def test_view_on_shared_memory_1d():
    @cuda.jit(dump=False)
    def k(out):
        sm = cuda.shared.array((8,), dtype=np.int8)
        for i in range(8):
            sm[i] = i + 1
        v = sm.view(np.int32)
        for i in range(v.shape[0]):
            out[i] = v[i]

    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    k[1, 1](out)
    expected = np.frombuffer(np.arange(1, 9, dtype=np.int8).tobytes(), dtype=np.int32)
    np.testing.assert_array_equal(out.copy_to_host(), expected)


def test_view_on_local_array_1d():
    @cuda.jit(dump=False)
    def k(out):
        loc = cuda.local.array((8,), dtype=np.int8)
        for i in range(8):
            loc[i] = i + 1
        v = loc.view(np.int32)
        for i in range(v.shape[0]):
            out[i] = v[i]

    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    k[1, 1](out)
    expected = np.frombuffer(np.arange(1, 9, dtype=np.int8).tobytes(), dtype=np.int32)
    np.testing.assert_array_equal(out.copy_to_host(), expected)


def test_view_complex64_to_float32():
    src = np.array([1 + 2j, -1 - 2j, 0 + 0j, 3.5 - 1.5j], dtype=np.complex64)
    _run_view_copy(src, np.float32, src.view(np.float32))


def _ld_st_patterns(byte_width):
    if byte_width == 4:
        return [".u32", ".b32", ".f32", ".v2.u16", ".v2.b16", ".v2.f16"]
    if byte_width == 8:
        return [".u64", ".b64", ".f64", ".v2.u32", ".v2.b32", ".v2.f32"]
    raise ValueError(f"unsupported byte width: {byte_width}")


def _assert_aligned_ld_st(copy_jit, byte_width, label):
    asm = list(copy_jit.inspect_asm().values())
    assert len(asm) == 1
    ptx = asm[0]
    patterns = _ld_st_patterns(byte_width)
    has_load = any(("ld.global" + p) in ptx for p in patterns)
    has_store = any(("st.global" + p) in ptx for p in patterns)
    assert has_load and has_store, f"missing aligned ld/st for {label}\n\nPTX:\n{ptx}"


@pytest.mark.parametrize(
    "dtype, expected_alignment",
    [(np.int32, 4), (np.float32, 4), (np.int64, 8), (np.float64, 8)],
    ids=["i32", "f32", "i64", "f64"],
)
def test_view_preserves_ptx_alignment(dtype, expected_alignment):
    @cuda.jit
    def copy(a, b):
        bv = b.view(dtype)
        bv[0] = a.view(dtype)[0]

    a = cuda.to_device(np.zeros(1, dtype=dtype))
    b = cuda.to_device(np.zeros(1, dtype=dtype))
    copy[1, 1](a, b)
    _assert_aligned_ld_st(copy, expected_alignment, dtype.__name__)


@pytest.mark.parametrize("view_dtype", [np.int32, np.float64])
def test_view_from_offset_bytes_reads_and_writes(view_dtype):
    itemsize = np.dtype(view_dtype).itemsize

    @cuda.jit(dump=False)
    def kernel(raw, out):
        view = raw[1 : 1 + out.size * itemsize].view(view_dtype)
        for i in range(view.size):
            out[i] = view[i]
        view[1] = 23

    values = np.array([7, 11, 19], dtype=view_dtype)
    raw = np.concatenate(
        (np.array([0xA5], np.uint8), values.view(np.uint8), np.array([0x5A], np.uint8))
    )
    device = cuda.to_device(raw)
    out = cuda.device_array(values.shape, dtype=view_dtype)
    kernel[1, 1](device, out)
    np.testing.assert_array_equal(out.copy_to_host(), values)
    expected = raw.copy()
    expected[1:-1].view(view_dtype)[1] = 23
    np.testing.assert_array_equal(device.copy_to_host(), expected)


def test_packed_field_same_width_view_preserves_stride_and_alias():
    @cuda.jit(dump=False)
    def kernel(records, out):
        view = records.value.view(np.int64)
        for i in range(view.size):
            out[i] = view[i]
        view[2] = 0

    records = np.zeros(5, dtype=[("tag", np.int32), ("value", np.float64)])
    records["tag"] = np.arange(5) + 100
    records["value"] = np.arange(5) + 0.5
    device = cuda.to_device(records)
    out = cuda.device_array(5, dtype=np.int64)
    kernel[1, 1](device, out)
    np.testing.assert_array_equal(out.copy_to_host(), records["value"].view(np.int64))
    expected = records.copy()
    expected["value"][2] = 0
    np.testing.assert_array_equal(device.copy_to_host(), expected)


def test_packed_nested_field_width_changing_view():
    @cuda.jit(dump=False)
    def kernel(records, out):
        view = records.value.view(np.uint32)
        for i in range(view.shape[0]):
            for j in range(view.shape[1]):
                out[i, j] = view[i, j]
        view[1, 2] = 0

    records = np.zeros(3, dtype=[("tag", np.uint8), ("value", np.float64, (2,))])
    records["tag"] = [17, 23, 31]
    records["value"] = np.arange(6).reshape(3, 2) + 0.5
    device = cuda.to_device(records)
    out = cuda.device_array((3, 4), dtype=np.uint32)
    kernel[1, 1](device, out)
    expected_values = records["value"].view(np.uint32).copy()
    np.testing.assert_array_equal(out.copy_to_host(), expected_values)
    expected = records.copy()
    expected["value"].view(np.uint32)[1, 2] = 0
    np.testing.assert_array_equal(device.copy_to_host(), expected)


def test_packed_rank_one_width_change_rejected_without_guard_writes():
    @cuda.jit(debug=True, opt=False)
    def kernel(records):
        view = records.value.view(np.uint32)
        for i in range(view.size):
            view[i] = 0

    host = np.zeros(4, [("value", np.float64), ("guard", np.uint32)])
    host["value"] = np.arange(4) + 0.5
    host["guard"] = 0xDEADBEEF
    device = cuda.to_device(host)
    with pytest.raises(ValueError):
        kernel[1, 1](device)
    np.testing.assert_array_equal(device.copy_to_host(), host)


@pytest.mark.parametrize("step", [-1, -2, -3])
def test_packed_field_reverse_slice_omitted_bounds(step):
    @cuda.jit(dump=False)
    def kernel(records, out):
        reverse = records.value[::step]
        for i in range(reverse.size):
            out[i] = reverse[i]
        reverse[1:] = 37

    records = np.zeros(8, dtype=[("tag", np.int32), ("value", np.float64)])
    records["tag"] = np.arange(8) + 100
    records["value"] = np.arange(8) + 0.5
    device = cuda.to_device(records)
    out = cuda.device_array(records["value"][::step].shape, dtype=np.float64)
    kernel[1, 1](device, out)
    np.testing.assert_array_equal(out.copy_to_host(), records["value"][::step])
    expected = records.copy()
    expected["value"][::step][1:] = 37
    np.testing.assert_array_equal(device.copy_to_host(), expected)


@pytest.mark.parametrize(
    "start,stop,step",
    [
        (-2, None, -1),
        (None, -1, -1),
        (-2, -7, -2),
        (100, -100, -2),
        (-100, 100, 3),
        (-6, -1, 2),
        (100, None, 1),
        (-100, None, -1),
        (None, None, np.iinfo(np.int64).max),
        (None, None, -np.iinfo(np.int64).max),
        (None, None, np.iinfo(np.int64).min),
        (0, 0, np.iinfo(np.int64).max),
        (0, 0, np.iinfo(np.int64).min),
    ],
)
@pytest.mark.parametrize("packed", [False, True])
def test_slice_explicit_bounds_preserve_payload_and_writes(start, stop, step, packed):
    omit_start, omit_stop = start is None, stop is None

    @cuda.jit
    def kernel(owner, out, size, runtime_start, runtime_stop, step):
        source = owner.value if packed else owner
        start = None if omit_start else runtime_start
        stop = None if omit_stop else runtime_stop
        view = source[start:stop:step]
        size[0] = view.size
        for i in range(view.size):
            out[i] = view[i]
        source[start:stop:step] = 37

    values = np.arange(8, dtype=np.float64) + 0.5
    records = np.zeros(8, [("tag", np.int32), ("value", np.float64)])
    records["tag"] = np.arange(8) + 100
    records["value"] = values
    host = records if packed else values
    selection = slice(start, stop, step)
    expected_values = values[selection].copy()
    device = cuda.to_device(host)
    out = cuda.to_device(np.full(8, -999.0, np.float64))
    size = cuda.device_array(1, np.int64)
    kernel[1, 1](device, out, size, start or 0, stop or 0, step)
    np.testing.assert_array_equal(size.copy_to_host(), [expected_values.size])
    expected_out = np.full(8, -999.0, np.float64)
    expected_out[: expected_values.size] = expected_values
    np.testing.assert_array_equal(out.copy_to_host(), expected_out)
    expected = host.copy()
    payload = expected["value"] if packed else expected
    payload[selection] = 37
    np.testing.assert_array_equal(device.copy_to_host(), expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("operation", ["view", "assign"])
@pytest.mark.parametrize("literal", [False, True])
def test_zero_slice_step_rejected_without_writes(packed, operation, literal):
    @cuda.jit(debug=True, opt=False)
    def kernel(owner, step):
        source = owner.value if packed else owner
        step = 0 if literal else step
        if operation == "view":
            view = source[::step]
            for i in range(view.size):
                view[i] = 99
        else:
            source[::step] = 99

    values = np.arange(6, dtype=np.float64) + 0.5
    records = np.zeros(6, [("guard", np.int32), ("value", np.float64)])
    records["guard"] = 123
    records["value"] = values
    host = records if packed else values
    device = cuda.to_device(host)
    with pytest.raises(ValueError):
        kernel[1, 1](device, 0)
    np.testing.assert_array_equal(device.copy_to_host(), host)
