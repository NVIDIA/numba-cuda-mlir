#!/usr/bin/env python3
import argparse
import math
import os
import struct
import time

import cuda.coop as cudax
from numba_cuda_mlir import cuda
import cupy
import cupyx
import numpy
import nvmath.linalg

from nvmath.bindings import cublas
from nvmath.bindings import cublasLt as cublaslt

from numba_cuda_mlir.numba_cuda.types import int32
from numba_cuda_mlir.numba_cuda.core import config


class CublasState:
    cublaslt_workspace_size = 32 * 1024 * 1024
    cublaslt_workspace = None
    cublas_compute_type = None
    cublas_handle = None
    cublaslt_handle = None


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(f"CUDA error code={result[0].value}")
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


"""
--------------------- FORWARD KERNELS ---------------------
"""

cx_warp_sum = cudax.warp.sum(cuda.float32)
sum_storage_bytes = cx_warp_sum.temp_storage_bytes
cx_warp_files = cx_warp_sum.files


def max_op(a, b):
    return a if a > b else b


cx_warp_max = cudax.warp.reduce(cuda.float32, max_op)
max_storage_bytes = cx_warp_max.temp_storage_bytes
cx_warp_files += cx_warp_max.files


@cuda.jit("float32(float32)", device=True, fastmath=True)
def warp_sum(val):
    temp_storage = cuda.shared.array(shape=sum_storage_bytes, dtype=cuda.uint8)
    warp_output = cx_warp_sum(temp_storage, val)
    return cuda.shfl_sync(0xFFFFFFFF, warp_output, 0)


@cuda.jit("float32(float32)", device=True, fastmath=True)
def warp_max(val):
    temp_storage = cuda.shared.array(shape=max_storage_bytes, dtype=cuda.uint8)
    warp_output = cx_warp_max(temp_storage, val)
    return cuda.shfl_sync(0xFFFFFFFF, warp_output, 0)


fp32 = numpy.float32


@cuda.jit(
    "void(float32[:], float32[:], float32[:],float32[:], float32[:], float32[:], int32, int32)",
    fastmath=True,
    link=cx_warp_files,
)
def layernorm_forward_kernel3(out, mean, rstd, inp, weight, bias, N, C):
    # This emulates the warp cg which is not available in numba
    warp_size = 32
    meta_group_size = cuda.blockDim.x // warp_size
    meta_group_rank = cuda.threadIdx.x // warp_size
    thread_rank = cuda.threadIdx.x % warp_size
    idx = cuda.blockIdx.x * meta_group_size + meta_group_rank
    # Ensure the thread is within the bounds of the array
    if idx >= N:
        return

    # mean
    sum = fp32(0.0)
    x = inp[idx * C :]
    for i in range(thread_rank, C, warp_size):
        sum += x[i]

    m = cuda.libdevice.fast_fdividef(warp_sum(sum), fp32(C))
    if thread_rank == 0:
        mean[idx] = m

    # rstd
    sum = fp32(0.0)
    for i in range(thread_rank, C, warp_size):
        diff = x[i] - m
        sum += diff * diff

    s = warp_sum(sum)
    s = cuda.libdevice.rsqrtf(cuda.libdevice.fast_fdividef(s, fp32(C)) + fp32(1e-5))

    if thread_rank == 0:
        rstd[idx] = s

    # final normalization and scaling by weight/bias
    o = out[idx * C :]
    for c in range(thread_rank, C, warp_size):
        n = s * (x[c] - m)
        o[c] = n * weight[c] + bias[c]


def ceil_div(a, b):
    return -(a // -b)


def layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C):
    block_size = 512
    N = B * T
    grid_size = ceil_div(N * 32, block_size)
    layernorm_forward_kernel3[grid_size, block_size](
        out, mean, rstd, inp, weight, bias, N, C
    )


def matmul_forward_cublas(out, inp, weight, bias, B, T, C, OC):
    assert bias is None  # bias is not supported for this kernel
    alpha = numpy.array(1.0, dtype=numpy.float32)
    beta = numpy.array(0.0, dtype=numpy.float32)
    cublas.sgemm(
        CublasState.cublas_handle,
        cublas.Operation.T,
        cublas.Operation.N,
        OC,
        B * T,
        C,
        alpha.ctypes.data,
        weight.data.ptr,
        C,
        inp.data.ptr,
        C,
        beta.ctypes.data,
        out.data.ptr,
        OC,
    )


def cublaslt_setattr(matmul_desc, name, value):
    name = name.upper()
    DescEnum = cublaslt.MatmulDescAttribute
    scalar_attrs = [e.name for e in DescEnum]
    if name not in scalar_attrs:
        raise RuntimeError("Unknown attr")
    get_dtype = cublaslt.get_matmul_desc_attribute_dtype
    attribute_buffer = numpy.zeros((1,), dtype=get_dtype(DescEnum[name]))
    attribute_buffer[0] = value
    cublaslt.matmul_desc_set_attribute(
        matmul_desc,
        DescEnum[name].value,
        attribute_buffer.ctypes.data,
        attribute_buffer.itemsize,
    )


def cublaslt_set_preference_attr(preference, name, value):
    name = name.upper()
    PreferenceEnum = cublaslt.MatmulPreferenceAttribute
    scalar_attrs = [e.name for e in PreferenceEnum]
    if name not in scalar_attrs:
        raise RuntimeError("Unknown attr")
    get_dtype = cublaslt.get_matmul_preference_attribute_dtype
    attribute_buffer = numpy.zeros((1,), dtype=get_dtype(PreferenceEnum[name]))
    attribute_buffer[0] = value
    cublaslt.matmul_preference_set_attribute(
        preference,
        PreferenceEnum[name].value,
        attribute_buffer.ctypes.data,
        attribute_buffer.itemsize,
    )


def matmul_forward_cublaslt(out, inp, weight, bias, B, T, C, OC):
    has_bias = bias is not None

    # check bias alignment
    if bias.data.ptr % 16 != 0:
        raise RuntimeError("Bias pointer is not aligned (cuBLASLt requirement)!\n")
    # create the operation descriptor
    opNoTranspose = cublas.Operation.N
    opTranspose = cublas.Operation.T
    epilogueBias = cublaslt.Epilogue.BIAS
    cuda_r_32f = nvmath.CudaDataType.CUDA_R_32F
    operation_desc = cublaslt.matmul_desc_create(
        CublasState.cublas_compute_type, cuda_r_32f
    )
    cublaslt_setattr(operation_desc, "TRANSA", opTranspose)
    cublaslt_setattr(operation_desc, "TRANSB", opNoTranspose)
    cublaslt_setattr(operation_desc, "EPILOGUE", epilogueBias)
    if has_bias:
        cublaslt_setattr(operation_desc, "BIAS_POINTER", bias.data.ptr)
    else:
        cublaslt_setattr(operation_desc, "BIAS_POINTER", 0)
    weight_layout = cublaslt.matrix_layout_create(cuda_r_32f, C, OC, C)
    input_layout = cublaslt.matrix_layout_create(cuda_r_32f, C, B * T, C)
    output_layout = cublaslt.matrix_layout_create(cuda_r_32f, OC, B * T, OC)
    bias_layout = cublaslt.matrix_layout_create(cuda_r_32f, OC, 1, OC)
    preference = cublaslt.matmul_preference_create()
    cublaslt_set_preference_attr(
        preference, "MAX_WORKSPACE_BYTES", CublasState.cublaslt_workspace_size
    )

    # find a suitable algorithm
    algorithm_dtype = algorithm_dtype = numpy.dtype(
        [
            ("algorithm", numpy.uint64, (8,)),
            ("workspace_size", numpy.uint64),
            ("status", numpy.int32),
            ("waves_count", numpy.float32),
            ("reserved", numpy.int32, (4,)),
        ]
    )
    algorithms_buffer = numpy.zeros((1,), dtype=algorithm_dtype)
    num_algorithms = numpy.zeros((1,), dtype=numpy.int32)
    cublaslt.matmul_algo_get_heuristic(
        CublasState.cublaslt_handle,
        operation_desc,
        weight_layout,
        input_layout,
        output_layout,
        output_layout,
        preference,
        1,
        algorithms_buffer.ctypes.data,
        num_algorithms.ctypes.data,
    )
    if num_algorithms[0] == 0:
        raise RuntimeError(
            f"No cuBLASLt algorithm: B: {B}, T: {T}, C: {C}, OC: {OC}, bias: {has_bias}"
        )

    # call matmul
    alpha = numpy.array(1.0, dtype=numpy.float32)
    beta = numpy.array(0.0, dtype=numpy.float32)
    cublaslt.matmul(
        CublasState.cublaslt_handle,
        operation_desc,
        alpha.ctypes.data,
        weight.data.ptr,
        weight_layout,
        inp.data.ptr,
        input_layout,
        beta.ctypes.data,
        out.data.ptr,
        output_layout,
        out.data.ptr,
        output_layout,
        algorithms_buffer[0]["algorithm"].ctypes.data,
        CublasState.cublaslt_workspace.data.ptr,
        CublasState.cublaslt_workspace_size,
        0,
    )

    cublaslt.matmul_preference_destroy(preference)
    cublaslt.matmul_desc_destroy(operation_desc)
    cublaslt.matrix_layout_destroy(weight_layout)
    cublaslt.matrix_layout_destroy(input_layout)
    cublaslt.matrix_layout_destroy(output_layout)
    cublaslt.matrix_layout_destroy(bias_layout)


@cuda.jit(
    "void(float32[:], float32, float32[:], float32, int32, int32)",
    fastmath=True,
    link=cx_warp_files,
)
def softmax_forward_kernel5(out, inv_temperature, inp, flt_max, N, T):
    # inp, out shape: (N, T, T), where N = B * NH
    # fuses the multiplication by scale inside attention
    # directly autoregressive, so we only compute the lower triangular part
    # uses the online softmax algorithm
    assert T % 4 == 0

    # This emulates the warp cg which is not available in numba
    warp_size = 32
    meta_group_size = cuda.blockDim.x // warp_size
    meta_group_rank = cuda.threadIdx.x // warp_size
    thread_rank = cuda.threadIdx.x % warp_size
    idx = (
        cuda.gridDim.x - cuda.blockIdx.x - 1
    ) * meta_group_size + meta_group_rank  # backward order
    if idx >= (N * T):
        return

    own_pos = idx % T
    pos_by_4 = own_pos // 4
    x = inp[idx * T :]
    maxval = -flt_max
    sumval = fp32(0.0)
    for i in range(int32(thread_rank), int32(pos_by_4), int32(warp_size)):
        v = x[i * 4 : i * 4 + 4]
        old_maxval = maxval
        for k in range(int32(4)):
            maxval = cuda.libdevice.fmaxf(maxval, v[k])
        sumval *= math.exp(inv_temperature * (old_maxval - maxval))
        for k in range(int32(4)):
            sumval += math.exp(inv_temperature * (v[k] - maxval))

    if (4 * pos_by_4 + thread_rank) <= own_pos:
        old_maxval = maxval
        maxval = cuda.libdevice.fmaxf(maxval, x[4 * pos_by_4 + thread_rank])
        sumval *= math.exp(inv_temperature * (old_maxval - maxval))
        sumval += math.exp(inv_temperature * (x[4 * pos_by_4 + thread_rank] - maxval))

    global_maxval = warp_max(maxval)
    sumval *= math.exp(inv_temperature * (maxval - global_maxval))

    # reduce sumval
    sum = warp_sum(sumval)
    # divide the whole row by the sum
    norm = fp32(1.0) / fp32(sum)
    for i in range(int32(thread_rank), int32(own_pos + 1), int32(warp_size)):
        # recalculation is faster than doing the round-trip through memory
        ev = math.exp(inv_temperature * (x[i] - global_maxval))
        out[idx * T + i] = ev * norm


@cuda.jit("UniTuple(int32, 4)(int32, int32, int32, int32)", device=True, fastmath=True)
def i2n_4(idx, E1, E2, E3):
    b = idx // (E1 * E2 * E3)
    rest = idx % (E1 * E2 * E3)
    nh_ = rest // (E2 * E3)
    rest = rest % (E2 * E3)
    t = rest // E3
    hs = rest % E3
    return (b, t, nh_, hs)


# @cuda.jit(fastmath=True)
@cuda.jit(
    "void(float32[:], float32[:], float32[:], float32[:,:,:,:,:], int32, int32, int32, int32)",
    fastmath=True,
)
def qkv_inp_kernel(q, k, v, inp, NH, T, HS, size):
    idx = cuda.grid(1)
    # Ensure the thread is within the bounds of the array
    if idx < size:
        b, t, nh_, hs = i2n_4(idx, NH, T, HS)
        q[idx] = inp[b, t, 0, nh_, hs]
        k[idx] = inp[b, t, 1, nh_, hs]
        v[idx] = inp[b, t, 2, nh_, hs]


def qkv_inp(q, k, v, inp, NH, T, HS, size):
    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    qkv_inp_kernel[blocks_per_grid, threads_per_block](q, k, v, inp, NH, T, HS, size)


@cuda.jit("void(float32[:], float32[:], int32, int32, int32, int32)", fastmath=True)
def scatter_kernel(out, vaccum, NH, T, HS, size):
    idx = cuda.grid(1)
    # Ensure the thread is within the bounds of the array
    if idx < size:
        b, n, nh_, d_ = i2n_4(idx, NH, T, HS)
        out[(b * NH * T * HS) + (n * NH * HS) + (nh_ * HS) + d_] = vaccum[idx]


def scatter(out, vaccum, NH, T, HS, size):
    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    scatter_kernel[blocks_per_grid, threads_per_block](out, vaccum, NH, T, HS, size)


def attention_forward(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH):
    softmax_block_size = 256
    HS = C // NH  # head size

    q = qkvr[0 * B * T * C :]
    k = qkvr[1 * B * T * C :]
    v = qkvr[2 * B * T * C :]

    inp_md = inp[: B * T * 3 * NH * HS].reshape((B, T, 3, NH, HS))
    size = B * NH * T * HS
    # Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    qkv_inp(q, k, v, inp_md, NH, T, HS, size)
    # batched matrix multiply using cuBLAS
    alpha = numpy.array(1.0, dtype=numpy.float32)
    beta = numpy.array(0.0, dtype=numpy.float32)
    cublas.sgemm_strided_batched(
        CublasState.cublas_handle,
        cublas.Operation.T,
        cublas.Operation.N,
        T,
        T,
        HS,
        alpha.ctypes.data,
        k.data.ptr,
        HS,
        T * HS,
        q.data.ptr,
        HS,
        T * HS,
        beta.ctypes.data,
        preatt.data.ptr,
        T,
        T * T,
        B * NH,
    )

    scale = numpy.array(1.0, dtype=numpy.float32) / numpy.sqrt(
        numpy.array(HS, dtype=numpy.int32)
    )
    grid_size = ceil_div(B * NH * T * 32, softmax_block_size)
    flt_max = fp32(numpy.finfo(numpy.float32).max)  # FLT_MAX
    softmax_forward_kernel5[grid_size, softmax_block_size](
        att, scale, preatt, flt_max, B * NH, T
    )
    # new approach: first cuBLAS another batched matmul
    # y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublas.sgemm_strided_batched(
        CublasState.cublas_handle,
        cublas.Operation.N,
        cublas.Operation.N,
        HS,
        T,
        T,
        alpha.ctypes.data,
        v.data.ptr,
        HS,
        T * HS,
        att.data.ptr,
        T,
        T * T,
        beta.ctypes.data,
        vaccum.data.ptr,
        HS,
        T * HS,
        B * NH,
    )

    # now unpermute
    # y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    scatter(out, vaccum, NH, T, HS, B * T * C)


# @cuda.jit('UniTuple(int32, 3)(int32, int32, int32)', device=True, fastmath=True)
@cuda.jit(device=True, fastmath=True)
def i2n_2(idx, E1, E2):
    bt = idx // E1
    b = bt // E2
    t = bt % E2
    c = idx % E1
    return (b, t, c)


@cuda.jit("void(float32[:], float32[:], float32[:])", fastmath=True)
def residual_forward_kernel(out, x, y):
    idx = cuda.grid(1)
    out[idx] = x[idx] + y[idx]


def residual_forward(out, inp1, inp2, N):
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    residual_forward_kernel[blocks_per_grid, threads_per_block](out, inp1, inp2)


@cuda.jit(fastmath=True)
def encoder_forward_kernel(out_md, wte_md, wpe_md, inp_md, C, T):
    idx = cuda.grid(1)
    # Ensure the thread is within the bounds of the array
    if idx < out_md.size:
        b, t, c = i2n_2(idx, C, T)
        out_md[b, t, c] = wte_md[inp_md[b, t], c] + wpe_md[t, c]


def encoder_forward(out, inpv, wte, wpe, B, T, C, V):
    out_md = out.reshape(B, T, C)
    wte_md = wte.reshape(V, C)
    wpe_md = wpe.reshape(T, C)
    inp_md = inpv.reshape(B, T)
    threads_per_block = 256
    blocks_per_grid = (out.size + threads_per_block - 1) // threads_per_block
    encoder_forward_kernel[blocks_per_grid, threads_per_block](
        out_md, wte_md, wpe_md, inp_md, C, T
    )


@cuda.jit("void(float32[:], float32[:], int32)", fastmath=True)
def gelu_forward_kernel(out, inp, N):
    idx = cuda.grid(1)
    if idx >= N:
        return
    xi = inp[idx]
    cube = fp32(0.044715) * xi * xi * xi
    scaling_factor = math.sqrt(fp32(2.0) / fp32(math.pi))
    out[idx] = (
        fp32(0.5)
        * xi
        * (fp32(1.0) + cuda.libdevice.tanhf(scaling_factor * (xi + cube)))
    )


def gelu_forward(out, inp, N):
    threads_per_block = 256
    blocks_per_grid = (out.size + threads_per_block - 1) // threads_per_block
    gelu_forward_kernel[blocks_per_grid, threads_per_block](out, inp, N)


@cuda.jit(
    "UniTuple(float32, 2)(int32, float32[:], int32, int32)",
    device=True,
    fastmath=True,
    link=cx_warp_files,
)
def prepare_softmax_blockwide_nofloat4(idx, inp, V, P):
    x = inp[idx * P :]
    thread_maxval = fp32(-math.inf)
    thread_sumval = fp32(0.0)

    # do the loop in reverse to maximise probability of L2 cache hits
    # so even small L2s get some hits on the 2nd read of the same thread
    for i in range(V + cuda.threadIdx.x - cuda.blockDim.x, -1, -cuda.blockDim.x):
        v = x[i]
        old_maxval = thread_maxval
        thread_maxval = cuda.libdevice.fmaxf(thread_maxval, v)
        thread_sumval *= math.exp(old_maxval - thread_maxval)
        thread_sumval += math.exp(v - thread_maxval)

    # two reductions of up to 1024 threads:
    # 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    # this results in much cleaner assembly than a multi-warp cg::reduce
    shared_maxval = cuda.shared.array(shape=32, dtype=cuda.float32)
    shared_sumval = cuda.shared.array(shape=32, dtype=cuda.float32)
    num_warps = cuda.blockDim.x // 32
    warp_id = cuda.threadIdx.x // 32
    lane_id = cuda.threadIdx.x % 32

    # reduce maxval within each warp
    warp_maxval = warp_max(thread_maxval)
    # thread 0 in each warp writes to shared memory
    if lane_id == 0:
        shared_maxval[warp_id] = warp_maxval
    cuda.syncthreads()

    # each thread now loads the maxval across previous warps
    # if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (
        shared_maxval[lane_id] if (lane_id < num_warps) else fp32(-3.402823e38)
    )  # FLT_MAX
    block_maxval = warp_max(warp_maxval)
    # each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= math.exp(thread_maxval - block_maxval)
    # (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    warp_sumval = warp_sum(thread_sumval)
    if lane_id == 0:
        shared_sumval[warp_id] = warp_sumval
    cuda.syncthreads()
    # same strategy, now reduce sumval across warps
    warp_sumval = shared_sumval[lane_id] if (lane_id < num_warps) else fp32(0.0)
    block_sumval = warp_sum(warp_sumval)
    return (fp32(1.0) / fp32(block_sumval), block_maxval)


@cuda.jit(
    "void(float32[:], float32[:], int32[:], int32, int32, int32, int32)",
    fastmath=True,
    link=cx_warp_files,
)
def fused_classifier_kernel3(logits, losses, targets, B, T, V, P):
    idx = cuda.blockIdx.x
    ix = targets[idx]
    # softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    scale, offset = prepare_softmax_blockwide_nofloat4(idx, logits, V, P)
    # calculate the probability needed for the loss and update (single-threaded)
    if cuda.threadIdx.x == 0:
        prob = math.exp(fp32(logits[idx * P + ix] - offset)) * scale
        losses[idx] = -math.log(prob)
    # very sensible default for dlosses is 1/(B*T), which is the uniform loss
    dloss = fp32(1.0) / fp32(B * T)
    # calculate the gradients directly, saves bandwidth from probs during training
    # but also supports writing probs for inference-only and debugging
    logits_vec = logits[idx * P :]
    for i in range(int32(cuda.threadIdx.x), int32(V), int32(cuda.blockDim.x)):
        # this is the 2nd read of logits after the one in prepare_softmax2
        # this data will never be needed again, so we reduce cache persistence
        v = logits_vec[i]
        prob = math.exp(fp32(v - offset)) * scale
        indicator = fp32(1.0) if i == ix else fp32(0.0)
        logits[idx * P + i] = (prob - indicator) * dloss


# replaces logits with logit gradients
def fused_classifier3(logits, losses, targets, B, T, V, P):
    block_size = 1024
    N = B * T
    grid_size = N
    fused_classifier_kernel3[grid_size, block_size](logits, losses, targets, B, T, V, P)


@cuda.jit(
    "void(float32[:], float32[:], int32, int32)",
    fastmath=True,
    link=cx_warp_files,
)
def softmax_forward_kernel7(out, inp, N, C):
    warp_size = 32
    meta_group_size = cuda.blockDim.x // warp_size
    meta_group_rank = cuda.threadIdx.x // warp_size
    thread_rank = cuda.threadIdx.x % warp_size

    # same as kernel4, but optimised for very large Cs with advanced unrolling

    # The trick is to read into a register array (all indices known at compile time)
    # and always read UNROLL_FACTOR values to maximise memory level parallelism
    # even if we would be out of bounds, we set the index to min(C-1, idx)
    # so we just do some unnecessary reads (obviously bad for small C)
    # the writes are in a separate loop with a conditional check for out of bounds
    # making it separate is necessary to convince the compiler to do the right thing
    UNROLL_FACTOR = 8
    warps_per_block = meta_group_size
    shared_size = (
        2 * 512 // 32 * 4
    )  # extern __shared__ 2 * block_size / 32 * sizeof(float)
    shared = cuda.shared.array(shape=shared_size, dtype=cuda.float32)
    idx = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    warpId = meta_group_rank
    laneId = thread_rank

    maxvals = shared
    sumvals = shared[warps_per_block:]

    if tid >= C:
        maxvals[warpId] = -math.inf
        sumvals[warpId] = fp32(0.0)
        return

    x = inp[idx * C :]
    y = out[idx * C :]

    # first, thread coarsening by directly accessing global memory in series
    maxval = -math.inf
    for i in range(tid, C, cuda.blockDim.x * UNROLL_FACTOR):
        for u in range(UNROLL_FACTOR):
            maxval = cuda.libdevice.fmaxf(
                maxval, x[min(C - 1, i + u * cuda.blockDim.x)]
            )

    maxval = warp_max(maxval)
    # now within-warp reductions for maxval
    # the 0th thread of each warp writes the maxval of that warp to shared memory
    if laneId == 0:
        maxvals[warpId] = maxval

    cuda.syncthreads()
    # now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if tid == 0:
        val = maxvals[tid]
        for i in range(1, warps_per_block):
            val = cuda.libdevice.fmaxf(val, maxvals[i])
        # store the final max in the first position
        maxvals[0] = val
    cuda.syncthreads()

    # broadcast the max to all threads
    offset = maxvals[0]

    # compute expf and write the result to global memory
    # + thread coarsening for sum
    sumval = fp32(0.0)
    for i in range(tid, C, cuda.blockDim.x * UNROLL_FACTOR):
        reg_array = cuda.local.array(shape=UNROLL_FACTOR, dtype=cuda.float32)
        for u in range(UNROLL_FACTOR):
            reg_array[u] = x[min(C - 1, i + u * cuda.blockDim.x)]
        for u in range(UNROLL_FACTOR):
            if (i + u * cuda.blockDim.x) < C:
                output = math.exp(reg_array[u] - offset)
                y[min(C - 1, i + u * cuda.blockDim.x)] = (
                    output  # compiler likes redundant min()?
                )
                sumval += output  # combined into the same loop unlike kernel3

    # okay now we calculated exp(x - max(x))
    # step 2: sum all the values and divide by the sum

    # within-warp reduction for sumval
    sumval = warp_sum(sumval)

    # write sumval to shared memory
    if laneId == 0:
        sumvals[warpId] = sumval
    cuda.syncthreads()

    if tid == 0:
        val = sumvals[tid]
        for i in range(1, warps_per_block):
            val += sumvals[i]
        sumvals[0] = val
    cuda.syncthreads()
    # broadcast the sum to all threads
    sum = sumvals[0]

    # divide the whole row by the sum
    for i in range(tid, C, cuda.blockDim.x * UNROLL_FACTOR):
        reg_array = cuda.local.array(shape=UNROLL_FACTOR, dtype=cuda.float32)
        for u in range(UNROLL_FACTOR):
            reg_array[u] = y[min(C - 1, i + u * cuda.blockDim.x)]
        for u in range(UNROLL_FACTOR):
            if (i + u * cuda.blockDim.x) < C:
                y[i + u * cuda.blockDim.x] = reg_array[u] / sum


def softmax_forward(out, inp, N, C):
    grid_size = N
    block_size = 512
    shared_mem_size = 2 * block_size // 32 * out.itemsize
    softmax_forward_kernel7[grid_size, block_size, 0, shared_mem_size](out, inp, N, C)


"""
------------------------------------------------------------
--------------------- BACKWARD KERNELS ---------------------
"""


@cuda.jit(
    "void(float32[:], float32[:], int32, int32, int32)",
    fastmath=True,
    link=cx_warp_files,
)
def matmul_backward_bias_kernel2(dbias, dout, B, T, OC):
    # dout is (B, T, OC), dbias is (OC)
    # e.g. if block_size = 128, then we have 4 warps per block, each in charge of one output channel
    warp_size = 32
    meta_group_size = cuda.blockDim.x // warp_size
    meta_group_rank = cuda.threadIdx.x // warp_size
    thread_rank = cuda.threadIdx.x % warp_size
    # meta_group_size is the number of warps in a block (e.g. 4), meta_group_rank is the warp index (0,1,2,3)
    idx = cuda.blockIdx.x * meta_group_size + meta_group_rank
    if idx >= OC:
        return

    BT = B * T  # number of elements to reduce in total, per channel
    sum = fp32(0.0)
    # first, thread coarsening to sum reduce the problem size from B*T to 32
    for i in range(thread_rank, BT, warp_size):
        sum += dout[i * OC + idx]
    # now do a warp-level reduce to get the sum across the 32 threads in this warp
    sum = warp_sum(sum)
    # write the result to output (global memory)
    if thread_rank == 0:
        dbias[idx] += sum


def matmul_backward(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC):
    one = numpy.array(1.0, dtype=numpy.float32)
    zero = numpy.array(0.0, dtype=numpy.float32)
    # backward to input, uses = in the backward pass (set the gradient)
    cublas.sgemm(
        CublasState.cublas_handle,
        cublas.Operation.N,
        cublas.Operation.N,
        C,
        B * T,
        OC,
        one.ctypes.data,
        weight.data.ptr,
        C,
        dout.data.ptr,
        OC,
        zero.ctypes.data,
        dinp.data.ptr,
        C,
    )
    # backward to weight, uses += in the backward pass (accumulate the gradient)
    cublas.sgemm(
        CublasState.cublas_handle,
        cublas.Operation.N,
        cublas.Operation.T,
        C,
        OC,
        B * T,
        one.ctypes.data,
        inp.data.ptr,
        C,
        dout.data.ptr,
        OC,
        one.ctypes.data,
        dweight.data.ptr,
        C,
    )
    # backward to bias, if given, does a +=
    if dbias is not None:
        block_size = 512
        grid_size = ceil_div(OC * 32, block_size)
        matmul_backward_bias_kernel2[grid_size, block_size](dbias, dout, B, T, OC)


@cuda.jit(
    "void(float32[:], float32[:], float32[:], float32[:], float32[:], float32[:], float32[:], float32[:], int32, int32, int32)",
    fastmath=True,
    link=cx_warp_files,
)
def layernorm_backward_kernel(
    dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C
):
    warp_size = 32
    meta_group_size = cuda.blockDim.x // warp_size
    meta_group_rank = cuda.threadIdx.x // warp_size
    thread_rank = cuda.threadIdx.x % warp_size
    idx = cuda.blockIdx.x * meta_group_size + meta_group_rank
    N = B * T
    if idx >= N:
        return
    b = idx // T
    t = idx % T
    dout_bt = dout[b * T * C + t * C :]
    dinp_bt = dinp[b * T * C + t * C :]
    inp_bt = inp[b * T * C + t * C :]
    mean_bt = mean[b * T + t]
    rstd_bt = rstd[b * T + t]

    # first: two reduce operations
    dnorm_mean = fp32(0.0)
    dnorm_norm_mean = fp32(0.0)
    for i in range(int32(thread_rank), int32(C), int32(warp_size)):
        norm_bti = fp32((inp_bt[i] - mean_bt) * rstd_bt)
        dnorm_i = fp32(weight[i] * dout_bt[i])
        dnorm_mean += fp32(dnorm_i)
        dnorm_norm_mean += fp32(dnorm_i * norm_bti)

    dnorm_mean = fp32(warp_sum(dnorm_mean))
    dnorm_norm_mean = fp32(warp_sum(dnorm_norm_mean))
    dnorm_mean = dnorm_mean / fp32(C)
    dnorm_norm_mean = dnorm_norm_mean / fp32(C)

    for i in range(int32(thread_rank), int32(C), int32(warp_size)):
        norm_bti = fp32(fp32(inp_bt[i] - mean_bt) * fp32(rstd_bt))
        dnorm_i = fp32(weight[i] * dout_bt[i])
        # gradient contribution to bias
        cuda.atomic.add(dbias, i, dout_bt[i])
        # gradient contribution to weight
        cuda.atomic.add(dweight, i, norm_bti * dout_bt[i])
        # gradient contribution to input
        dval = fp32(0.0)
        dval += dnorm_i  # term 1
        dval -= dnorm_mean  # term 2
        dval -= fp32(norm_bti * dnorm_norm_mean)  # term 3
        dval *= fp32(rstd_bt)  # final scale
        dinp_bt[i] += dval


def layernorm_backward(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C):
    block_size = 256
    N = B * T
    # one warp per token, so we need to divide by 32 here.
    grid_size = ceil_div(N, block_size // 32)
    layernorm_backward_kernel[grid_size, block_size](
        dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C
    )


@cuda.jit("void(float32[:], float32[:], float32[:], int32)", fastmath=True)
def gelu_backward_kernel(dinp, inp, dout, N):
    i = cuda.grid(1)
    if i < N:
        scaling_factor = math.sqrt(fp32(2.0) / fp32(math.pi))
        x = inp[i]
        cube = fp32(0.044715) * x * x * x
        tanh_arg = scaling_factor * (x + cube)
        tanh_out = cuda.libdevice.tanhf(tanh_arg)
        coshf_out = cuda.libdevice.coshf(tanh_arg)
        sech_out = fp32(1.0) / (coshf_out * coshf_out)
        local_grad = fp32(0.5) * (fp32(1.0) + tanh_out) + x * fp32(
            0.5
        ) * sech_out * scaling_factor * (fp32(1.0) + fp32(3.0) * fp32(0.044715) * x * x)
        dinp[i] = local_grad * dout[i]


def gelu_backward(dinp, inp, dout, N):
    block_size = 128
    grid_size = ceil_div(N, block_size)
    gelu_backward_kernel[grid_size, block_size](dinp, inp, dout, N)


@cuda.jit("void(float32[:], float32[:], int32, int32, int32, int32)", fastmath=True)
def unpermute_kernel_backward(dinp, dout, B, N, NH, d):
    idx = cuda.grid(1)
    if idx < (B * NH * N * d):
        b = idx // (NH * N * d)
        rest = idx % (NH * N * d)
        nh_ = rest // (N * d)
        rest = rest % (N * d)
        n = rest // d
        d_ = rest % d
        other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_
        dinp[idx] = dout[other_idx]


@cuda.jit(
    "void(float32[:], float32[:], float32[:], int32, int32, int32, float32)",
    fastmath=True,
    link=cx_warp_files,
)
def softmax_autoregressive_backward_kernel(dpreatt, datt, att, B, T, C, scale):
    BlockSize = 256
    T_per_block = 4

    warp_size = 32
    meta_group_rank = cuda.threadIdx.x // warp_size
    thread_rank = cuda.threadIdx.x % warp_size
    block_acc = cuda.shared.array(shape=32, dtype=cuda.float32)

    idx = cuda.blockIdx.y
    # go through blocks in reverse order, so the slowest block starts first
    t0 = T - 1 - T_per_block * cuda.blockIdx.x
    att = att[idx * T * T :]
    datt = datt[idx * T * T :]
    dpreatt = dpreatt[idx * T * T :]

    if meta_group_rank == 0:
        block_acc[thread_rank] = 0

    for to in range(int32(0), int32(T_per_block)):
        t = t0 - to
        if t < 0:
            return
        att_bth = att[t * T :]
        datt_bth = datt[t * T :]
        dpreatt_bth = dpreatt[t * T :]

        local_sum = fp32(0.0)
        for t2 in range(int32(cuda.threadIdx.x), int32(t + 1), int32(BlockSize)):
            local_sum += att_bth[t2] * datt_bth[t2]
            # if cuda.blockIdx.x == 0 and cuda.blockIdx.y == 0 and cuda.threadIdx.x == 0:
            #    print(t2 + t *T + idx * T * T, local_sum, att_bth[t2], datt_bth[t2])

        block_acc[meta_group_rank] = warp_sum(local_sum)
        cuda.syncthreads()
        local_sum = warp_sum(block_acc[thread_rank])
        for t3 in range(int32(cuda.threadIdx.x), int32(t + 1), int32(BlockSize)):
            acc = att_bth[t3] * (datt_bth[t3] - local_sum)
            dpreatt_bth[t3] = scale * acc


@cuda.jit(
    "void(float32[:], float32[:], float32[:], float32[:], int32, int32, int32, int32)",
    fastmath=True,
)
def permute_kernel_backward(dinp, dq, dk, dv, B, N, NH, d):
    idx = cuda.grid(1)
    if idx < B * NH * N * d:
        b = idx // (NH * N * d)
        rest = idx % (NH * N * d)
        nh_ = rest // (N * d)
        rest = rest % (N * d)
        n = rest // d
        d_ = rest % d

        inp_idx = (
            (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_
        )
        dinp[inp_idx] = dq[idx]
        dinp[inp_idx + NH * d] = dk[idx]
        dinp[inp_idx + 2 * (NH * d)] = dv[idx]


# the sequence of transformations in this compound op is:
# inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
def attention_backward(
    dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, att, B, T, C, NH
):
    block_size = 256
    HS = C // NH  # head size
    one = numpy.array(1.0, dtype=numpy.float32)
    zero = numpy.array(0.0, dtype=numpy.float32)
    # unpack convenience pointers into q, k, v
    q = qkvr[0 * B * T * C :]
    k = qkvr[1 * B * T * C :]
    v = qkvr[2 * B * T * C :]
    dq = dqkvr[0 * B * T * C :]
    dk = dqkvr[1 * B * T * C :]
    dv = dqkvr[2 * B * T * C :]
    # backward through the unpermute operation
    num_blocks = ceil_div(B * T * C, block_size)
    unpermute_kernel_backward[num_blocks, block_size](dvaccum, dout, B, T, NH, HS)
    # backward into datt
    cublas.sgemm_strided_batched(
        CublasState.cublas_handle,
        cublas.Operation.T,
        cublas.Operation.N,
        T,
        T,
        HS,
        one.ctypes.data,
        v.data.ptr,
        HS,
        T * HS,
        dvaccum.data.ptr,
        HS,
        T * HS,
        zero.ctypes.data,
        datt.data.ptr,
        T,
        T * T,
        B * NH,
    )
    # backward into dv
    cublas.sgemm_strided_batched(
        CublasState.cublas_handle,
        cublas.Operation.N,
        cublas.Operation.T,
        HS,
        T,
        T,
        one.ctypes.data,
        dvaccum.data.ptr,
        HS,
        T * HS,
        att.data.ptr,
        T,
        T * T,
        zero.ctypes.data,
        dv.data.ptr,
        HS,
        T * HS,
        B * NH,
    )
    # backward into preatt
    scale = numpy.array(1.0, dtype=numpy.float32) / numpy.sqrt(
        numpy.array(HS, dtype=numpy.int32)
    )
    softmax_autoregressive_backward_kernel[(T // 4, B * NH), 256](
        dpreatt, datt, att, B, T, C, scale
    )
    # backward into q
    cublas.sgemm_strided_batched(
        CublasState.cublas_handle,
        cublas.Operation.N,
        cublas.Operation.N,
        HS,
        T,
        T,
        one.ctypes.data,
        k.data.ptr,
        HS,
        T * HS,
        dpreatt.data.ptr,
        T,
        T * T,
        zero.ctypes.data,
        dq.data.ptr,
        HS,
        T * HS,
        B * NH,
    )
    # backward into k
    cublas.sgemm_strided_batched(
        CublasState.cublas_handle,
        cublas.Operation.N,
        cublas.Operation.T,
        HS,
        T,
        T,
        one.ctypes.data,
        q.data.ptr,
        HS,
        T * HS,
        dpreatt.data.ptr,
        T,
        T * T,
        zero.ctypes.data,
        dk.data.ptr,
        HS,
        T * HS,
        B * NH,
    )  # backward into inp
    num_blocks = ceil_div(B * NH * T * HS, block_size)
    permute_kernel_backward[num_blocks, block_size](dinp, dq, dk, dv, B, T, NH, HS)


@cuda.jit(
    "void(float32[:], float32[:], float32[:], int32[:], int32, int32, int32)",
    fastmath=True,
)
def encoder_backward_kernel(dwte, dwpe, dout, inp, B, T, C):
    idx = cuda.grid(1)
    N = B * T * C
    if idx < N:
        bt = idx // C
        b = bt // T
        t = bt % T
        c = idx % C

        ix = inp[b * T + t]

        dout_btc = dout[b * T * C + t * C + c :]
        dwte_ix = dwte[ix * C + c :]
        dwpe_tc = dwpe[t * C + c :]
        cuda.atomic.add(dwte_ix, 0, fp32(dout_btc[0]))
        cuda.atomic.add(dwpe_tc, 0, fp32(dout_btc[0]))


def encoder_backward(dwte, dwpe, dout, inp, B, T, C):
    N = B * T * C
    block_size = 256
    grid_size = ceil_div(N, block_size)
    encoder_backward_kernel[grid_size, block_size](dwte, dwpe, dout, inp, B, T, C)


"""
------------------------------------------------------------
"""


# Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
# Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
@cuda.jit("float32(float32, float32, float32)", device=True, fastmath=True)
def lerp_(start, end, weight):
    return cuda.libdevice.fma(weight, end, cuda.libdevice.fma(-weight, start, start))


@cuda.jit(
    "void(float32[:], float32[:], float32[:], float32[:], int32, float32, float32, float32, float32, float32, float32, float32)",
    fastmath=True,
)
def adamw_kernel2(
    params_memory,
    grads_memory,
    m_memory,
    v_memory,
    num_parameters,
    learning_rate,
    beta1,
    beta2,
    beta1_correction,
    beta2_correction,
    eps,
    weight_decay,
):
    i = cuda.grid(1)
    if i >= num_parameters:
        return
        # guard
    grad = fp32(grads_memory[i])
    m = fp32(m_memory[i])
    v = fp32(v_memory[i])
    # update the first moment (momentum)
    m = fp32(lerp_(grad, m, beta1))
    m_memory[i] = fp32(m)
    # update the second moment (RMSprop)
    v = fp32(lerp_(fp32(grad * grad), v, beta2))
    v_memory[i] = v
    m /= fp32(beta1_correction)  # m_hat
    v /= fp32(beta2_correction)  # v_hat
    params_memory[i] -= learning_rate * (
        fp32(m) / fp32(math.sqrt(fp32(v)) + fp32(eps))
        + fp32(weight_decay) * params_memory[i]
    )


"""
------------------------------------------------------------
"""


# TODO(ecastill) Use torch DataLoader instead?
class DataLoader:
    def __init__(self):
        self.B = 0
        self.T = 0
        self.tokens_file = None
        self.file_size = 0
        self.current_position = 0
        self.batch = None
        self.num_batches = 0

    def inputs(self):
        return self.batch[:-1]

    def targets(self):
        # targets are shifted by one
        return self.batch[1:]

    def init(self, filename, B, T):
        self.B = B
        self.T = T

        # open the input file for reading
        self.tokens_file = open(filename, "rb")
        self.tokens_file.seek(0, os.SEEK_END)
        self.file_size = self.tokens_file.tell()
        self.tokens_file.seek(0, os.SEEK_SET)
        if self.file_size < (B * T + 1) * 4:
            raise RuntimeError(
                "Error: file size is too small for the batch size and sequence length"
            )
        self.current_position = 0

        # allocate space for B*T + 1 integers to store the inputs and targets
        # Using CUDA CPU pinned memory for faster PCI Express transfers to GPU
        # See: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        self.batch = cupyx.empty_pinned(B * T + 1, dtype=cupy.int32)
        self.num_batches = self.file_size // (B * T * 4)

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        B = self.B
        T = self.T
        # if we are at the end of the file, loop back to the beginning
        if self.current_position + (B * T + 1) * 4 > self.file_size:
            self.current_position = 0

        # read the B*T+1 integers from the file into batch
        self.tokens_file.seek(self.current_position, os.SEEK_SET)
        batch_fmt = f"{B*T+1}i"
        batch_len = struct.calcsize(batch_fmt)
        batch_unpack = struct.Struct(batch_fmt).unpack_from
        batch = batch_unpack(self.tokens_file.read(batch_len))
        self.batch[:] = batch
        self.current_position += batch_len


class Tokenizer:
    def __init__(self):
        self.vocab_size = 0
        self.token_table = None
        self.init_ok = 0

    def init(self, filename):
        with open(filename, "rb") as f:
            header_fmt = "256i"
            header_len = struct.calcsize(header_fmt)
            header_unpack = struct.Struct(header_fmt).unpack_from
            header = header_unpack(f.read(header_len))
            assert header[0] == 20240328
            assert header[1] == 1
            self.vocab_size = header[2]
            self.token_table = []
            for _ in range(self.vocab_size):
                length = f.read(1)[0]
                token_bytes = f.read(length)
                self.token_table.append(token_bytes)
            self.init_ok = 1

    def decode(self, token_id):
        if self.init_ok == 0:
            return None
        if token_id < self.vocab_size:
            return self.token_table[token_id]
        else:
            raise RuntimeError(f"invalid token id {token_id}!\n")


NUM_PARAMETER_TENSORS = 16


class ParameterTensors:
    def __init__(self):
        self.wte = None  # (V, C)
        self.wpe = None  # (maxT, C)
        self.ln1w = None  # (L, C)
        self.ln1b = None  # (L, C)
        self.qkvw = None  # (L, 3*C, C)
        self.qkvb = None  # (L, 3*C)
        self.attprojw = None  # (L, C, C)
        self.attprojb = None  # (L, C)
        self.ln2w = None  # (L, C)
        self.ln2b = None  # (L, C)
        self.fcw = None  # (L, 4*C, C)
        self.fcb = None  # (L, 4*C)
        self.fcprojw = None  # (L, C, 4*C)
        self.fcprojb = None  # (L, C)
        self.lnfw = None  # (C)
        self.lnfb = None  # (C)

        # Used for iterate the tensors in order
        self.names = [
            "wte",
            "wpe",
            "ln1w",
            "ln1b",
            "qkvw",
            "qkvb",
            "attprojw",
            "attprojb",
            "ln2w",
            "ln2b",
            "fcw",
            "fcb",
            "fcprojw",
            "fcprojb",
            "lnfw",
            "lnfb",
        ]


class GPT2Config:
    def __init__(self):
        self.max_seq_len = None
        self.vocab_size = None
        self.num_layers = None
        self.num_heads = None
        self.channels = None

    def clone(self):
        new = GPT2Config()
        new.max_seq_len = self.max_seq_len
        new.vocab_size = self.vocab_size
        new.num_layers = self.num_layers
        new.num_heads = self.num_heads
        new.channels = self.channels
        return new


NUM_ACTIVATION_TENSORS = 25


class ActivationTensors:
    def __init__(self):
        self.encoded = None  # (B, T, C)
        self.ln1 = None  # (L, B, T, C)
        self.ln1_mean = None  # (L, B, T)
        self.ln1_rstd = None  # (L, B, T)
        self.qkv = None  # (L, B, T, 3*C)
        self.atty = None  # (L, B, T, C)
        self.preatt = None  # (L, B, NH, T, T)
        self.att = None  # (L, B, NH, T, T)
        self.attproj = None  # (L, B, T, C)
        self.residual2 = None  # (L, B, T, C)
        self.ln2 = None  # (L, B, T, C)
        self.ln2_mean = None  # (L, B, T)
        self.ln2_rstd = None  # (L, B, T)
        self.fch = None  # (L, B, T, 4*C)
        self.fch_gelu = None  # (L, B, T, 4*C)
        self.fcproj = None  # (L, B, T, C)
        self.residual3 = None  # (L, B, T, C)
        self.lnf = None  # (B, T, C)
        self.lnf_mean = None  # (B, T)
        self.lnf_rstd = None  # (B, T)
        # if we have targets, this will be the logit _gradients_.
        self.logits = None  # (B, T, V)
        self.probs = None  # (B, T, V)
        self.losses = None  # (B, T)
        # adding these two compared to the CPU .c code, needed for attention kernel as buffers
        self.qkvr = None  # (L, B, T, 3*C)
        self.v_accum = None  # (L, B, T, C)

        self.names = [
            "encoded",
            "ln1",
            "ln1_mean",
            "ln1_rstd",
            "qkv",
            "atty",
            "preatt",
            "att",
            "attproj",
            "residual2",
            "ln2",
            "ln2_mean",
            "ln2_rstd",
            "fch",
            "fch_gelu",
            "fcproj",
            "residual3",
            "lnf",
            "lnf_mean",
            "lnf_rstd",
            "logits",
            "probs",
            "losses",
            "qkvr",
            "v_accum",
        ]


# Used for fwd and bwd
def fill_in_activation_sizes(act_sizes, B, T, config):
    V = config.vocab_size
    L = config.num_layers
    NH = config.num_heads
    C = config.channels
    act_sizes[0] = B * T * C  # encoded
    act_sizes[1] = L * B * T * C  # ln1
    act_sizes[2] = L * B * T  # ln1_mean
    act_sizes[3] = L * B * T  # ln1_rstd
    act_sizes[4] = L * B * T * 3 * C  # qkv
    act_sizes[5] = L * B * T * C  # atty
    act_sizes[6] = B * NH * T * T  # preatt
    act_sizes[7] = L * B * NH * T * T  # att
    act_sizes[8] = L * B * T * C  # attproj
    act_sizes[9] = L * B * T * C  # residual2
    act_sizes[10] = L * B * T * C  # ln2
    act_sizes[11] = L * B * T  # ln2_mean
    act_sizes[12] = L * B * T  # ln2_rstd
    act_sizes[13] = L * B * T * 4 * C  # fch
    act_sizes[14] = L * B * T * 4 * C  # fch_gelu
    act_sizes[15] = L * B * T * C  # fcproj
    act_sizes[16] = L * B * T * C  # residual3
    act_sizes[17] = B * T * C  # lnf
    act_sizes[18] = B * T  # lnf_mean
    act_sizes[19] = B * T  # lnf_rstd
    act_sizes[20] = B * T * V  # logits
    act_sizes[21] = B * T * V  # probs
    act_sizes[22] = B * T  # losses
    act_sizes[23] = L * B * T * 3 * C  # qkvr
    act_sizes[24] = B * T * C  # v_accum


# TODO Unify with the params code, is exactly the same
def malloc_and_point(param_or_acts, sizes, num):
    # TODO(check): again, we are relying on cupy memory pool
    memory = cupy.empty(num, dtype=cupy.float32)
    current_size = 0
    for i, n in enumerate(param_or_acts.names):
        setattr(param_or_acts, n, memory[current_size : current_size + sizes[i]])
        current_size += sizes[i]
    return memory


class GPT2:
    def __init__(self):
        # We just replicate the c++ structure, we could use cupy tensors & views for this
        # Each of the parameters is just a pointer to a big memory allocation
        self.params = ParameterTensors()
        self.params_sizes = [0 for i in range(NUM_PARAMETER_TENSORS)]
        self.params_memory = None
        self.config = GPT2Config()
        self.acts = ActivationTensors()
        self.acts_memory = None
        self.act_sizes = [0 for i in range(NUM_ACTIVATION_TENSORS)]
        self.grads = ParameterTensors()
        self.grads_memory = None
        self.grads_acts = ActivationTensors()
        self.grads_acts_memory = None
        self.num_parameters = 0
        self.m_memory = None
        self.v_memory = None

    def fill_in_parameter_sizes(self):
        V = self.config.vocab_size
        C = self.config.channels
        maxT = self.config.max_seq_len
        L = self.config.num_layers
        self.params_sizes[0] = V * C
        self.params_sizes[1] = maxT * C
        self.params_sizes[2] = L * C
        self.params_sizes[3] = L * C
        self.params_sizes[4] = L * (3 * C) * C
        self.params_sizes[5] = L * (3 * C)
        self.params_sizes[6] = L * C * C
        self.params_sizes[7] = L * C
        self.params_sizes[8] = L * C
        self.params_sizes[9] = L * C
        self.params_sizes[10] = L * (4 * C) * C
        self.params_sizes[11] = L * (4 * C)
        self.params_sizes[12] = L * C * (4 * C)
        self.params_sizes[13] = L * C
        self.params_sizes[14] = C
        self.params_sizes[15] = C

    def build_from_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            header_fmt = "256i"
            header_len = struct.calcsize(header_fmt)
            header_unpack = struct.Struct(header_fmt).unpack_from
            model_header = header_unpack(f.read(header_len))
            if model_header[0] != 20240326:
                raise RuntimeError("Bad magic model file")
            if model_header[1] != 1:
                raise RuntimeError("Bad version in model file")

            self.config.max_seq_len = maxT = model_header[2]
            self.config.vocab_size = V = model_header[3]
            self.config.num_layers = L = model_header[4]
            self.config.num_heads = NH = model_header[5]
            self.config.channels = C = model_header[6]
            print("[GPT-2]")
            print(f"max_seq_len: {maxT}")
            print(f"vocab_size: {V}")
            print(f"num_layers: {L}")
            print(f"num_heads: {NH}")
            print(f"channels: {C}")
            self.fill_in_parameter_sizes()

            num_parameters = 0
            for i in range(NUM_PARAMETER_TENSORS):
                num_parameters += self.params_sizes[i]

            print(f"num_parameters: {num_parameters}")
            self.num_parameters = num_parameters

            # create memory for model parameters on the device
            self.params_memory = malloc_and_point(
                self.params, self.params_sizes, num_parameters
            )
            size_in_mb = int(
                round(num_parameters * cupy.dtype(cupy.float32).itemsize)
                / (1024 * 1024)
            )
            print(f"allocated {size_in_mb} MiB for model parameters")

            # Read the parameters and copy them to a numpy array
            params_fmt = f"{num_parameters}f"
            params_len = struct.calcsize(params_fmt)
            params_unpack = struct.Struct(params_fmt).unpack_from
            params_memory_cpu = numpy.array(
                params_unpack(f.read(params_len)), dtype=numpy.float32
            )
            cupy.cuda.runtime.memcpy(
                self.params_memory.data.ptr,
                params_memory_cpu.ctypes.data,
                params_memory_cpu.nbytes,
                cupy.cuda.runtime.memcpyHostToDevice,
            )
        # other inits
        self.batch_size = 0
        self.seq_len = 0
        self.mean_loss = -1.0  # -1.0f will designate no loss

    def _initialize_acts(self, B, T):
        if self.acts_memory is None:
            # record the current B,T as well
            self.batch_size = B
            self.seq_len = T
            # and now allocate the space
            fill_in_activation_sizes(self.act_sizes, B, T, self.config)
            num_activations = 0
            for i in range(NUM_ACTIVATION_TENSORS):
                num_activations += self.act_sizes[i]
            print("num_activations", num_activations)
            self.num_activations = num_activations
            self.acts_memory = malloc_and_point(
                self.acts, self.act_sizes, num_activations
            )
            acts_memory = int(
                round(
                    num_activations * cupy.dtype(cupy.float32).itemsize / (1024 * 1024)
                )
            )
            print(f"allocated {acts_memory} MiB for activations")
            self.cpu_losses = cupyx.empty_pinned(B * T, dtype=cupy.float32)
        else:
            # validate B,T is consistent with how we've allocated the memory before
            # in principle we could get more clever here in the future, for now this is safest
            if B != self.batch_size or T != self.seq_len:
                raise RuntimeError(
                    f"Model: B={self.batch_size} T={self.seq_len}, Desired: B={B} T={T}"
                )

    def forward(self, inputs, targets, B, T):
        # ensure the model was initialized or error out
        if self.params_memory is None:
            raise RuntimeError("Error: model was not initialized properly.\n")

        # convenience parameters
        V = self.config.vocab_size
        L = self.config.num_layers
        NH = self.config.num_heads
        C = self.config.channels

        # Validate inputs, all indices must be in the range [0, V)
        for i in range(B * T):
            assert 0 <= inputs[i] < V
            if targets is not None:
                assert 0 <= targets[i] < V

        # allocate space for all the activations if needed (done here, lazily)
        self._initialize_acts(B, T)

        # copy inputs/targets to the model
        # We are just creating a new cupy array time, memory is drawn from the pool and the numpy array
        # with the inputs is in pinned memory
        self.inputs = cupy.array(inputs)
        if targets is not None:
            self.targets = cupy.array(targets)

        # forward pass
        params = self.params
        acts = self.acts
        encoder_forward(acts.encoded, self.inputs, params.wte, params.wpe, B, T, C, V)
        if cupy.isnan(acts.encoded).any():
            print("NaN detected in encoder_forward output")
            print(
                f"  encoded stats: min={cupy.min(acts.encoded)}, max={cupy.max(acts.encoded)}, mean={cupy.mean(acts.encoded)}"
            )

        for l in range(L):

            residual = acts.encoded if l == 0 else acts.residual3[(l - 1) * B * T * C :]

            l_ln1w = params.ln1w[l * C :]
            l_ln1b = params.ln1b[l * C :]
            l_qkvw = params.qkvw[l * 3 * C * C :]
            l_qkvb = params.qkvb[l * 3 * C :]
            l_attprojw = params.attprojw[l * C * C :]
            l_attprojb = params.attprojb[l * C :]
            l_ln2w = params.ln2w[l * C :]
            l_ln2b = params.ln2b[l * C :]
            l_fcw = params.fcw[l * 4 * C * C :]
            l_fcb = params.fcb[l * 4 * C :]
            l_fcprojw = params.fcprojw[l * C * 4 * C :]
            l_fcprojb = params.fcprojb[l * C :]

            l_ln1 = acts.ln1[l * B * T * C :]
            l_ln1_mean = acts.ln1_mean[l * B * T :]
            l_ln1_rstd = acts.ln1_rstd[l * B * T :]
            l_qkv = acts.qkv[l * B * T * 3 * C :]
            l_qkvr = acts.qkvr[l * B * T * 3 * C :]
            l_atty = acts.atty[l * B * T * C :]
            l_att = acts.att[l * B * NH * T * T :]
            l_attproj = acts.attproj[l * B * T * C :]
            l_residual2 = acts.residual2[l * B * T * C :]
            l_ln2 = acts.ln2[l * B * T * C :]
            l_ln2_mean = acts.ln2_mean[l * B * T :]
            l_ln2_rstd = acts.ln2_rstd[l * B * T :]
            l_fch = acts.fch[l * B * T * 4 * C :]
            l_fch_gelu = acts.fch_gelu[l * B * T * 4 * C :]
            l_fcproj = acts.fcproj[l * B * T * C :]
            l_residual3 = acts.residual3[l * B * T * C :]
            # these are only needed as scratchpads for the forward pass, but
            # need not be stored for backward
            l_preatt = acts.preatt
            l_v_accum = acts.v_accum

            layernorm_forward(
                l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C
            )
            if cupy.isnan(l_ln1).any():
                print(f"NaN detected in layer {l} after ln1")
                print(
                    f"  residual stats: min={cupy.min(residual)}, max={cupy.max(residual)}"
                )
                print(f"  ln1w stats: min={cupy.min(l_ln1w)}, max={cupy.max(l_ln1w)}")
                print(f"  ln1b stats: min={cupy.min(l_ln1b)}, max={cupy.max(l_ln1b)}")

            matmul_forward_cublaslt(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C)
            if cupy.isnan(l_qkv).any():
                print(f"NaN detected in layer {l} after qkv matmul")
                print(f"  l_ln1 stats: min={cupy.min(l_ln1)}, max={cupy.max(l_ln1)}")
                print(f"  l_qkvw stats: min={cupy.min(l_qkvw)}, max={cupy.max(l_qkvw)}")

            attention_forward(
                l_atty, l_v_accum, l_qkvr, l_preatt, l_att, l_qkv, B, T, C, NH
            )
            if cupy.isnan(l_atty).any():
                print(f"NaN detected in layer {l} after attention")
                print(f"  l_qkv stats: min={cupy.min(l_qkv)}, max={cupy.max(l_qkv)}")
                print(f"  l_att stats: min={cupy.min(l_att)}, max={cupy.max(l_att)}")
                print(
                    f"  l_preatt stats: min={cupy.min(l_preatt)}, max={cupy.max(l_preatt)}"
                )

            matmul_forward_cublaslt(
                l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C
            )
            if cupy.isnan(l_attproj).any():
                print(f"NaN detected in layer {l} after attproj matmul")

            residual_forward(l_residual2, residual, l_attproj, B * T * C)
            if cupy.isnan(l_residual2).any():
                print(f"NaN detected in layer {l} after residual2")

            layernorm_forward(
                l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C
            )
            if cupy.isnan(l_ln2).any():
                print(f"NaN detected in layer {l} after ln2")

            matmul_forward_cublaslt(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C)
            if cupy.isnan(l_fch).any():
                print(f"NaN detected in layer {l} after fch matmul")

            gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C)
            if cupy.isnan(l_fch_gelu).any():
                print(f"NaN detected in layer {l} after gelu")

            matmul_forward_cublaslt(
                l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C
            )
            if cupy.isnan(l_fcproj).any():
                print(f"NaN detected in layer {l} after fcproj matmul")

            residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C)
            if cupy.isnan(l_residual3).any():
                print(f"NaN detected in layer {l} after residual3")

        residual = acts.residual3[
            (L - 1) * B * T * C :
        ]  # last residual is in residual3
        layernorm_forward(
            acts.lnf,
            acts.lnf_mean,
            acts.lnf_rstd,
            residual,
            params.lnfw,
            params.lnfb,
            B,
            T,
            C,
        )
        if cupy.isnan(acts.lnf).any():
            print("NaN detected in final layernorm")
            print(
                f"  residual stats: min={cupy.min(residual)}, max={cupy.max(residual)}"
            )
            print(
                f"  lnfw stats: min={cupy.min(params.lnfw)}, max={cupy.max(params.lnfw)}"
            )
            print(
                f"  lnfb stats: min={cupy.min(params.lnfb)}, max={cupy.max(params.lnfb)}"
            )

        matmul_forward_cublas(acts.logits, acts.lnf, params.wte, None, B, T, C, V)
        if cupy.isnan(acts.logits).any():
            print("NaN detected in logits after matmul")
            print(f"  lnf stats: min={cupy.min(acts.lnf)}, max={cupy.max(acts.lnf)}")
            print(
                f"  wte stats: min={cupy.min(params.wte)}, max={cupy.max(params.wte)}"
            )

        if targets is not None:
            fused_classifier3(
                acts.logits, acts.losses, self.targets.ravel(), B, T, V, V
            )
            if cupy.isnan(acts.losses).any():
                print("NaN detected in losses after fused_classifier")
                print(
                    f"  logits stats: min={cupy.min(acts.logits)}, max={cupy.max(acts.logits)}"
                )
            self.cpu_losses = cupy.asnumpy(acts.losses[: B * T])
            mean_loss = numpy.mean(self.cpu_losses)
            self.mean_loss = mean_loss
        else:
            softmax_forward(acts.probs, acts.logits, B * T, V)
            if cupy.isnan(acts.probs).any():
                print("NaN detected in probs after softmax")
            self.mean_loss = -1

    def zero_grad(self):
        if self.grads_memory is not None:
            self.grads_acts_memory.fill(0.0)
            self.grads_memory.fill(0.0)

    def _initialize_grads(self):
        if self.grads_memory is None:
            self.grads_memory = malloc_and_point(
                self.grads, self.params_sizes, self.num_parameters
            )
            size_in_mb = int(
                round(self.num_parameters * cupy.dtype(cupy.float32).itemsize)
                / (1024 * 1024)
            )
            print(f"allocated {size_in_mb} MiB for parameter gradients")
            # we're going to be clever for the activations backward pass. we don't need to exactly
            # mirror the forward pass acrtivations and we will save memory.
            bw_act_sizes = [0] * NUM_ACTIVATION_TENSORS
            cfg = self.config.clone()
            cfg.num_layers = 1
            # copy the configuration but override number of layers to 1
            fill_in_activation_sizes(bw_act_sizes, self.batch_size, self.seq_len, cfg)
            # on top of that, some buffers are not needed at all, set their sizes to zero
            bw_act_sizes[0] = 0  # encoded
            bw_act_sizes[2] = 0  # ln1_mean
            bw_act_sizes[3] = 0  # ln1_rstd
            bw_act_sizes[8] = 0  # attproj
            bw_act_sizes[9] = 0  # residual2
            bw_act_sizes[11] = 0  # ln2_mean
            bw_act_sizes[12] = 0  # ln2_rstd
            bw_act_sizes[18] = 0  # lnf_mean
            bw_act_sizes[19] = 0  # lnf_rstd
            bw_act_sizes[21] = 0  # probs
            # count up and allocate the space
            self.num_grad_acts = 0
            for i in range(NUM_ACTIVATION_TENSORS):
                self.num_grad_acts += bw_act_sizes[i]
            self.grads_acts_memory = malloc_and_point(
                self.grads_acts, bw_act_sizes, self.num_grad_acts
            )

            size_in_mb = int(
                round(self.num_grad_acts * cupy.dtype(cupy.float32).itemsize)
                / (1024 * 1024)
            )
            print(f"allocated {size_in_mb} MiB for activation gradients")
            # init gradients of parameters and activations to zero
            self.zero_grad()

    def backward(self):
        if self.mean_loss == -1.0:
            raise RuntimeError("Error: must forward with targets before backward")

        self._initialize_grads()
        # convenience shortcuts
        B = self.batch_size
        T = self.seq_len
        V = self.config.vocab_size
        L = self.config.num_layers
        NH = self.config.num_heads
        C = self.config.channels

        # backward pass: go in the reverse order of the forward pass, and call backward() functions
        params = self.params  # for brevity
        grads = self.grads
        acts = self.acts
        grads_acts = self.grads_acts

        # we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        # this was done in the fused classifier kernel as last step of forward pass
        # technically that is a small, inline backward() pass of calculating
        # total, final loss as the mean over all losses over all (B,T) positions in the batch
        # next: backward the classifier matmul
        if cupy.isnan(acts.logits).any():
            print("NaN in logits at start of backward")
        matmul_backward(
            grads_acts.lnf,
            grads.wte,
            None,
            acts.logits,
            acts.lnf,
            params.wte,
            B,
            T,
            C,
            V,
        )
        if cupy.isnan(grads_acts.lnf).any():
            print("NaN detected in backward after matmul_backward (lnf)")
        residual = acts.residual3[
            (L - 1) * B * T * C :
        ]  # last residual is in residual3
        dresidual = (
            grads_acts.residual3
        )  # the main buffer holding the gradient in the backward pass
        layernorm_backward(
            dresidual,
            grads.lnfw,
            grads.lnfb,
            grads_acts.lnf,
            residual,
            params.lnfw,
            acts.lnf_mean,
            acts.lnf_rstd,
            B,
            T,
            C,
        )
        if cupy.isnan(dresidual).any():
            print("NaN detected in backward after final layernorm_backward")
        for l in range(L - 1, -1, -1):
            residual = acts.encoded if l == 0 else acts.residual3[(l - 1) * B * T * C :]
            # get the pointers of the weights for this layer
            l_ln1w = params.ln1w[l * C :]
            l_qkvw = params.qkvw[l * 3 * C * C :]
            l_attprojw = params.attprojw[l * C * C :]
            l_ln2w = params.ln2w[l * C :]
            l_fcw = params.fcw[l * 4 * C * C :]
            l_fcprojw = params.fcprojw[l * C * 4 * C :]
            # get the pointers of the gradients of the weights for this layer
            dl_ln1w = grads.ln1w[l * C :]
            dl_ln1b = grads.ln1b[l * C :]
            dl_qkvw = grads.qkvw[l * 3 * C * C :]
            dl_qkvb = grads.qkvb[l * 3 * C :]
            dl_attprojw = grads.attprojw[l * C * C :]
            dl_attprojb = grads.attprojb[l * C :]
            dl_ln2w = grads.ln2w[l * C :]
            dl_ln2b = grads.ln2b[l * C :]
            dl_fcw = grads.fcw[l * 4 * C * C :]
            dl_fcb = grads.fcb[l * 4 * C :]
            dl_fcprojw = grads.fcprojw[l * C * 4 * C :]
            dl_fcprojb = grads.fcprojb[l * C :]
            # get the pointers of the activations for this layer
            l_ln1 = acts.ln1[l * B * T * C :]
            l_ln1_mean = acts.ln1_mean[l * B * T :]
            l_ln1_rstd = acts.ln1_rstd[l * B * T :]
            l_qkv = acts.qkv[l * B * T * 3 * C :]
            l_qkvr = acts.qkvr[l * B * T * 3 * C :]
            l_atty = acts.atty[l * B * T * C :]
            l_att = acts.att[l * B * NH * T * T :]
            l_residual2 = acts.residual2[l * B * T * C :]
            l_ln2 = acts.ln2[l * B * T * C :]
            l_ln2_mean = acts.ln2_mean[l * B * T :]
            l_ln2_rstd = acts.ln2_rstd[l * B * T :]
            l_fch = acts.fch[l * B * T * 4 * C :]
            l_fch_gelu = acts.fch_gelu[l * B * T * 4 * C :]
            # get the pointers of the gradients of the activations for this layer
            # notice that there is no l *, because we just have a single copy, and keep
            # re-using this memory in every Transformer block as we calculate backward pass
            dl_ln1 = grads_acts.ln1
            dl_qkv = grads_acts.qkv
            dl_qkvr = grads_acts.qkvr
            dl_atty = grads_acts.atty
            dl_preatt = grads_acts.preatt
            dl_att = grads_acts.att
            dl_v_accum = grads_acts.v_accum
            dl_ln2 = grads_acts.ln2
            dl_fch = grads_acts.fch
            dl_fch_gelu = grads_acts.fch_gelu

            matmul_backward(
                dl_fch_gelu,
                dl_fcprojw,
                dl_fcprojb,
                dresidual,
                l_fch_gelu,
                l_fcprojw,
                B,
                T,
                4 * C,
                C,
            )
            if cupy.isnan(dl_fch_gelu).any():
                print(
                    f"NaN detected in backward layer {l} after fcproj matmul_backward"
                )
            gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C)
            if cupy.isnan(dl_fch).any():
                print(f"NaN detected in backward layer {l} after gelu_backward")
            matmul_backward(
                dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C
            )
            if cupy.isnan(dl_ln2).any():
                print(f"NaN detected in backward layer {l} after fch matmul_backward")
            # layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
            layernorm_backward(
                dresidual,
                dl_ln2w,
                dl_ln2b,
                dl_ln2,
                l_residual2,
                l_ln2w,
                l_ln2_mean,
                l_ln2_rstd,
                B,
                T,
                C,
            )
            if cupy.isnan(dresidual).any():
                print(
                    f"NaN detected in backward layer {l} after ln2 layernorm_backward"
                )
            matmul_backward(
                dl_atty,
                dl_attprojw,
                dl_attprojb,
                dresidual,
                l_atty,
                l_attprojw,
                B,
                T,
                C,
                C,
            )
            if cupy.isnan(dl_atty).any():
                print(
                    f"NaN detected in backward layer {l} after attproj matmul_backward"
                )
            attention_backward(
                dl_qkv,
                dl_qkvr,
                dl_preatt,
                dl_att,
                dl_v_accum,
                dl_atty,
                l_qkv,
                l_qkvr,
                l_att,
                B,
                T,
                C,
                NH,
            )
            if cupy.isnan(dl_qkv).any():
                print(f"NaN detected in backward layer {l} after attention_backward")
            matmul_backward(
                dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C
            )
            if cupy.isnan(dl_ln1).any():
                print(f"NaN detected in backward layer {l} after qkv matmul_backward")
            # layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
            layernorm_backward(
                dresidual,
                dl_ln1w,
                dl_ln1b,
                dl_ln1,
                residual,
                l_ln1w,
                l_ln1_mean,
                l_ln1_rstd,
                B,
                T,
                C,
            )
            if cupy.isnan(dresidual).any():
                print(
                    f"NaN detected in backward layer {l} after ln1 layernorm_backward"
                )

        encoder_backward(grads.wte, grads.wpe, dresidual, self.inputs, B, T, C)
        if cupy.isnan(grads.wte).any():
            print("NaN detected in grads.wte after encoder_backward")
        if cupy.isnan(grads.wpe).any():
            print("NaN detected in grads.wpe after encoder_backward")

    def update(self, learning_rate, beta1, beta2, eps, weight_decay, t):
        # reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

        # lazily allocate the memory for m_memory and v_memory
        if self.m_memory is None:
            self.m_memory = cupy.zeros(self.num_parameters, dtype=cupy.float32)
            self.v_memory = cupy.zeros(self.num_parameters, dtype=cupy.float32)
            size_in_mb = int(
                round(self.num_parameters * cupy.dtype(cupy.float32).itemsize)
                / (1024 * 1024)
            )
            print(f"allocated {size_in_mb} MiB for AdamW optimizer state m")
            print(f"allocated {size_in_mb} MiB for AdamW optimizer state v")

        block_size = 512
        num_blocks = ceil_div(self.num_parameters, block_size)
        beta1_correction = numpy.float32(1.0) - numpy.float_power(
            numpy.float32(beta1), t
        )
        beta2_correction = numpy.float32(1.0) - numpy.float_power(
            numpy.float32(beta2), t
        )

        if cupy.isnan(self.grads_memory).any():
            print("NaN detected in gradients before update")
            print(
                f"  grads stats: min={cupy.min(self.grads_memory)}, max={cupy.max(self.grads_memory)}"
            )

        adamw_kernel2[num_blocks, block_size](
            self.params_memory,
            self.grads_memory,
            self.m_memory,
            self.v_memory,
            self.num_parameters,
            fp32(learning_rate),
            fp32(beta1),
            fp32(beta2),
            fp32(beta1_correction),
            fp32(beta2_correction),
            fp32(eps),
            fp32(weight_decay),
        )

        if cupy.isnan(self.params_memory).any():
            print("NaN detected in parameters after update")
            print(
                f"  params stats: min={cupy.min(self.params_memory)}, max={cupy.max(self.params_memory)}"
            )


def sample_mult(probabilities, n, coin):
    # sample index from probabilities (they must sum to 1!)
    # coin is a random number in [0, 1), usually from random_f32()
    cdf = numpy.cumsum(probabilities)
    index = numpy.searchsorted(cdf, coin, side="right")
    return min(index, n - 1)


class Logger:
    def __init__(self, filename):
        self.logfile = open(filename, "w") if filename is not None else None
        self.flush_every = 20
        self.buffer = []

    def __del__(self):
        if self.logfile is not None:
            self.logfile.close()

    def log_val(self, step, val_loss):
        if self.logfile is not None:
            self.buffer.append(f"s:{step} tel:{val_loss:.4f}\n")

    def log_train(self, step, train_loss):
        if self.logfile is not None:
            self.buffer.append(f"s:{step} trl:{train_loss:.4f}\n")
            if step % self.flush_every == 0:
                self.logfile.writelines(self.buffer)
                self.buffer = []


def validation(model, val_loader, val_num_batches, B, T):
    val_loss = 0.0
    val_loader.reset()
    for _ in range(val_num_batches):
        val_loader.next_batch()
        model.forward(val_loader.inputs(), val_loader.targets(), B, T)
        val_loss += model.mean_loss
    val_loss /= val_num_batches
    return val_loss


def inference(model, tokenizer, genT, B, T):
    gen_tokens = numpy.empty(B * T, dtype=numpy.int32)
    cpu_probs = numpy.empty(model.config.vocab_size, dtype=numpy.float32)
    GPT2_EOT = 50256

    # fill up gen_tokens with the GPT2_EOT, which kicks off the generation
    for i in range(B * T):
        gen_tokens[i] = GPT2_EOT

    # now sample from the model autoregressively
    print("generating:\n---")
    token_str = bytearray(b"")
    for t in range(1, genT):
        # note that inference is very wasteful here because for each token
        # we re-calculate the forward pass for all of (B,T) positions from scratch
        # but the inference here is just for sanity checking anyway
        # and we can maybe optimize a bit more later, with careful tests
        model.forward(gen_tokens, None, B, T)
        # furthermore, below we're only using b=0 (i.e. the first row) of all B rows
        # we're in principle running B "inference streams" in parallel here
        # only using position 0 because it's a bit faster (copy less probs from GPU -> CPU)
        # get the V-dimensional vector probs[0, t-1, :]
        probs = model.acts.probs[(t - 1) * model.config.vocab_size :]
        # move probs back to CPU and sample
        cupy.cuda.runtime.memcpy(
            cpu_probs.ctypes.data,
            probs.data.ptr,
            model.config.vocab_size * probs.itemsize,
            cupy.cuda.runtime.memcpyDeviceToHost,
        )
        coin = numpy.random.random_sample()
        next_token = sample_mult(cpu_probs, model.config.vocab_size, coin)
        gen_tokens[t] = next_token
        # print the generated token, either using the Tokenizer or a fallback
        if tokenizer.init_ok:
            token_str.extend(tokenizer.decode(next_token))
    print(token_str.decode("utf-8"))


def train_loop(
    model,
    tokenizer,
    train_loader,
    val_loader,
    logger,
    learning_rate,
    genT,
    val_loss_every,
    val_max_batches,
    sample_every,
    B,
    T,
    max_steps=None,
):
    train_num_batches = train_loader.num_batches  # let's do 1 epoch by default
    if max_steps is not None:
        train_num_batches = min(train_num_batches, max_steps)
    val_num_batches = (
        train_loader.num_batches
        if train_loader.num_batches < val_max_batches
        else val_max_batches
    )

    train_losses = []
    val_losses = []

    for step in range(train_num_batches + 1):
        last_step = step == train_num_batches
        # once in a while estimate the validation loss
        if step % val_loss_every == 0 or last_step:
            val_loss = validation(model, val_loader, val_num_batches, B, T)
            print(f"val loss {val_loss}")
            logger.log_val(step, val_loss)
            val_losses.append(val_loss)

        # once in a while do model inference to print generated text
        if step > 0 and step % sample_every == 0 or last_step:
            inference(model, tokenizer, genT, B, T)

        if last_step:
            break

        start = time.time()
        train_loader.next_batch()

        model.forward(train_loader.inputs(), train_loader.targets(), B, T)
        model.zero_grad()
        model.backward()
        model.update(learning_rate, 0.9, 0.999, 1e-8, 0.0, step + 1)

        train_losses.append(model.mean_loss)
        print(
            f"step {step+1}/{train_num_batches}: train loss {model.mean_loss} ({int((time.time() - start)*1000)} ms)"
        )
        logger.log_train(step, model.mean_loss)

    return train_losses, val_losses


def setup_cuda():
    deviceIdx = 0
    # checkCudaErrors(cudart.cudaSetDevice(deviceIdx))
    # deviceProp = checkCudaErrors(cudart.cudaGetDeviceProperties(deviceIdx))
    print("[System]")
    # print(f"Device {deviceIdx}: {deviceProp.name.decode('utf-8')}")
    # For nvmath errors are thrown in a pythonic way
    enable_tf32 = True  # deviceProp.major >= 8
    print(f"enable_tf32: {enable_tf32}")
    setup_cublas(enable_tf32)


def setup_cublas(enable_tf32):
    CublasState.cublas_handle = cublas.create()
    CublasState.cublaslt_handle = cublaslt.create()
    if enable_tf32:
        CublasState.cublas_compute_type = cublas.ComputeType.COMPUTE_32F_FAST_TF32
        cublas_math_mode = cublas.Math.TF32_TENSOR_OP_MATH
    else:
        CublasState.cublas_compute_type = cublas.ComputeType.COMPUTE_32F
        cublas_math_mode = cublas.Math.DEFAULT_MATH
    cublas.set_math_mode(CublasState.cublas_handle, cublas_math_mode)
    # setup the (global) cuBLASLt workspace
    CublasState.cublaslt_workspace = cupy.empty(
        CublasState.cublaslt_workspace_size, dtype=cupy.uint8
    )


def run_training(
    input_dataset_prefix="data/tiny_shakespeare",
    output_log_file=None,
    B=4,
    T=1024,
    learning_rate=1e-4,
    val_loss_every=20,
    val_max_batches=20,
    sample_every=20,
    genT=64,
    max_steps=None,
    working_dir=None,
):
    import os

    if working_dir is not None:
        os.chdir(working_dir)

    print(f"input dataset prefix: {input_dataset_prefix}")
    print(f"output log file: {'NULL' if output_log_file is None else output_log_file}")
    print(f"batch size B: {B}")
    print(f"sequence length T: {T}")
    print(f"learning rate: {learning_rate}")
    print(f"val_loss_every: {val_loss_every}")
    print(f"val_max_batches: {val_max_batches}")
    print(f"sample_every: {sample_every}")
    print(f"genT: {genT}")
    print(f"max_steps: {max_steps}")

    config.CUDA_ARRAY_INTERFACE_SYNC = False

    numpy.random.seed(1336)

    setup_cuda()

    model = GPT2()
    model.build_from_checkpoint("gpt2_124M.bin")

    train_tokens_filename = f"{input_dataset_prefix}_train.bin"
    val_tokens_filename = f"{input_dataset_prefix}_val.bin"

    train_loader = DataLoader()
    train_loader.init(train_tokens_filename, B, T)
    val_loader = DataLoader()
    val_loader.init(val_tokens_filename, B, T)

    print(
        f"train dataset num_batches: {train_loader.num_batches}",
        train_loader.num_batches,
    )
    print(f"val dataset num_batches: {val_loader.num_batches}", val_loader.num_batches)

    tokenizer = Tokenizer()
    tokenizer.init("gpt2_tokenizer.bin")
    logger = Logger(output_log_file)

    return train_loop(
        model,
        tokenizer,
        train_loader,
        val_loader,
        logger,
        learning_rate,
        genT,
        val_loss_every,
        val_max_batches,
        sample_every,
        B,
        T,
        max_steps=max_steps,
    )


def main(args):
    run_training(
        input_dataset_prefix=args.input,
        output_log_file=args.output,
        B=args.batch_size,
        T=args.sequence_length,
        learning_rate=args.learning_rate,
        val_loss_every=args.val_loss_every,
        val_max_batches=args.val_max_batches,
        sample_every=args.sample_every,
        genT=args.genT,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="llm.py", description="Train an LLM in Python using CUDA primitives"
    )
    parser.add_argument("-i", "--input", default="data/tiny_shakespeare")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument("-t", "--sequence-length", type=int, default=1024)
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("-v", "--val-loss-every", type=int, default=20)
    parser.add_argument("-m", "--val-max-batches", type=int, default=20)
    parser.add_argument("-s", "--sample-every", type=int, default=20)
    parser.add_argument("-g", "--genT", type=int, default=64)
    main(parser.parse_args())
