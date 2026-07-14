/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kernel.h"

#include "check.h"
#include "cuda_loader.h"
#include "cuda_helper.h"
#include "ref_ptr.h"

#include <cstdint>
#include "cuda_shim.h"
#include <dlpack.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <optional>
#include <string>
#include <utility>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace {

PyObject* g___cuda_array_interface___pyunicode;
PyObject* g___array_interface___pyunicode;
PyObject* g_typestr_pyunicode;
PyObject* g_shape_pyunicode;
PyObject* g_data_pyunicode;
PyObject* g_strides_pyunicode;
PyObject* g___dlpack___pyunicode;
PyObject* g_int64_pyunicode;

PyTypeObject* g_torch_Tensor_type;
PyTypeObject* g_torch_cuda_Stream_type;
PyObject* g_torch_to_dlpack_func;
PyObject* g_torch_cuda_getCurrentRawStream;

PyTypeObject* g_cupy_ndarray_type;
PyTypeObject* g_cupy_cuda_Stream_type;
PyObject* g_cupy_cuda_get_current_stream;

PyTypeObject* g_numba_cuda_Stream_type;

PyTypeObject* g_cuda_core_Stream_type;

PyTypeObject* g_enum_Enum_type;

constexpr uint8_t BYTE_BITWIDTH = 8;

constexpr const char* ERROR_CODE_GLOBAL_NAME = "__numba_cuda_mlir_error_code";

void* allocate_aligned_storage(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return nullptr;
    return ptr;
#endif
}

void free_aligned_storage(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Forward declarations
class CudaLibrary;
Status check_kernel_error_code(const CudaLibrary& lib);


// RAII wrapper around CUlibrary
class CudaLibrary {
public:
    explicit CudaLibrary(CUlibrary lib, CUcontext ctx) : lib_(lib), ctx_(ctx) {}

    CudaLibrary(CudaLibrary&& other) : lib_(other.lib_), ctx_(other.ctx_) {
        other.lib_ = nullptr;
        other.ctx_ = nullptr;
    }

    CudaLibrary(const CudaLibrary&) = delete;
    void operator=(const CudaLibrary&) = delete;

    ~CudaLibrary() {
        if (lib_) {
            CUresult res = g_cuLibraryUnload(lib_);
            CHECK(res == CUDA_SUCCESS);
        }
    }

    const CUlibrary& get() const {
        return lib_;
    }

    CUcontext context() const {
        return ctx_;
    }

private:
    CUlibrary lib_;
    CUcontext ctx_;
};

Result<CudaLibrary> load_cuda_library(const void* cubin) {
    // Get current context to associate with library.
    // We require that CUDA is already initialized and a context exists.
    // This is typically done by numba's driver or by the user.
    CUcontext ctx;
    CUresult res = g_cuCtxGetCurrent(&ctx);
    if (res != CUDA_SUCCESS || ctx == nullptr) {
        return raise(PyExc_RuntimeError,
            "No CUDA context available. Ensure CUDA is initialized before launching kernels. "
            "This typically happens automatically when using numba_cuda_mlir operations.");
    }

    CUlibrary lib;
    res = g_cuLibraryLoadData(&lib, cubin, nullptr, nullptr, 0,
                              nullptr, nullptr, 0);
    if (res == CUDA_SUCCESS)
        return CudaLibrary(lib, ctx);

    return raise(PyExc_RuntimeError, "Failed to load CUDA library from binary data: %s",
                 get_cuda_error(res));
}

struct CudaKernel {
    CudaLibrary lib;
    CUkernel kernel;
};

Result<CudaKernel> load_cuda_kernel(const void* cubin, const char* func_name) {
    Result<CudaLibrary> lib = load_cuda_library(cubin);
    if (!lib.is_ok()) return ErrorRaised;

    CUkernel kernel;
    CUresult res = g_cuLibraryGetKernel(&kernel, lib->get(), func_name);
    if (res == CUDA_SUCCESS)
        return CudaKernel{std::move(*lib), kernel};

    return raise(PyExc_RuntimeError, "Failed to get kernel '%s' from library: %s",
                 func_name, get_cuda_error(res));
}

Status raise_kernel_error(int32_t code) {
    switch (code) {
        case 1: return raise(PyExc_AssertionError, "Kernel assertion failed");
        case 2: return raise(PyExc_IndexError, "Index out of bounds in kernel");
        case 3: return raise(PyExc_ValueError, "Invalid value in kernel");
        case 4: return raise(PyExc_RuntimeError, "Runtime error in kernel");
        case 5: return raise(PyExc_ZeroDivisionError, "Division by zero in kernel");
        default: return raise(PyExc_RuntimeError, "Kernel error (code %d)", code);
    }
}

Status check_kernel_error_code(const CudaLibrary& lib) {
    // Get error global from module
    CUmodule mod;
    CUresult err_res = g_cuLibraryGetModule(&mod, lib.get());
    if (err_res != CUDA_SUCCESS)
        return OK;

    CUdeviceptr error_ptr;
    size_t error_size;
    err_res = g_cuModuleGetGlobal(&error_ptr, &error_size, mod, ERROR_CODE_GLOBAL_NAME);
    if (err_res != CUDA_SUCCESS || error_size < sizeof(int32_t))
        return OK;

    // Ensure we're in the correct context
    CUcontext current_ctx;
    err_res = g_cuCtxGetCurrent(&current_ctx);
    if (err_res != CUDA_SUCCESS)
        return OK;

    CUcontext lib_ctx = lib.context();
    bool need_ctx_switch = (current_ctx != lib_ctx);
    CUcontext old_ctx = nullptr;

    if (need_ctx_switch) {
        err_res = g_cuCtxPushCurrent(lib_ctx);
        if (err_res != CUDA_SUCCESS)
            return OK;
    }

    // Synchronize to ensure kernel has completed
    err_res = g_cuCtxSynchronize();
    if (err_res != CUDA_SUCCESS) {
        if (need_ctx_switch) g_cuCtxPopCurrent(&old_ctx);
        return raise(PyExc_RuntimeError, "Failed to synchronize CUDA context: %s",
                     get_cuda_error(err_res));
    }

    // Read error code from device
    int32_t error_code = 0;
    err_res = g_cuMemcpyDtoH(&error_code, error_ptr, sizeof(int32_t));

    if (need_ctx_switch) {
        g_cuCtxPopCurrent(&old_ctx);
    }

    if (err_res != CUDA_SUCCESS)
        return OK;

    if (error_code != 0) {
        // Reset error code for next launch
        if (need_ctx_switch)
            g_cuCtxPushCurrent(lib_ctx);
        int32_t zero = 0;
        g_cuMemcpyHtoD(error_ptr, &zero, sizeof(int32_t));
        if (need_ctx_switch)
            g_cuCtxPopCurrent(&old_ctx);
        return raise_kernel_error(error_code);
    }

    return OK;
}

enum class ConstantArgType {
    INT64,
    FLOAT64,
    STRING,
};

struct ConstantArg {
    ConstantArgType type;
    union {
        int64_t i64;
        double f64;
    } value;
    std::string str;

    template <typename T>
    ConstantArg(T val) :
        type(std::is_floating_point_v<T> ? ConstantArgType::FLOAT64 : ConstantArgType::INT64),
        value{}
    {
        if constexpr (std::is_floating_point_v<T>) {
            value.f64 = static_cast<double>(val);
        } else {
            value.i64 = static_cast<int64_t>(val);
        }
    }

    explicit ConstantArg(std::string val) :
        type(ConstantArgType::STRING),
        value{},
        str(std::move(val))
    {}

    bool operator==(const ConstantArg& other) const {
        if (type != other.type) return false;
        switch (type) {
        case ConstantArgType::INT64:
            return value.i64 == other.value.i64;
        case ConstantArgType::FLOAT64:
            // compare bits of two floats directly
            uint64_t this_float_bits, other_float_bits;
            std::memcpy(&this_float_bits, &value.f64, sizeof(value.f64));
            std::memcpy(&other_float_bits, &other.value.f64, sizeof(other.value.f64));
            return this_float_bits == other_float_bits;
        case ConstantArgType::STRING:
            return str == other.str;
        default:
            return false;
        }
    }
};

[[maybe_unused]]
inline void hash_combine(size_t& h, size_t other) {
    h ^= other + 0x9e3779b9 + (h << 6) + (h >> 2);
}

template <typename T>
struct HashVector {
    size_t operator() (const std::vector<T>& v) const {
        size_t ret = 0;
        const std::hash<T> elem_hash;
        for (const T& x : v)
            hash_combine(ret, elem_hash(x));
        return ret;
    }
};

struct CudaKernelHandle {
    CudaKernel cukernel;
    PyPtr post_load_callback;
    bool cooperative = false;
    // When false, the compiler proved the kernel contains no use of the
    // error-code global, so the post-launch device error check (which
    // synchronizes the context) is skipped and launches stay
    // asynchronous.
    bool check_error_code = true;
};

// This should compile to a no-op
inline uint32_t dtype_as_uint(DLDataType dtype) {
    return static_cast<uint32_t>(dtype.code)
        | (static_cast<uint32_t>(dtype.bits) << 8)
        | (static_cast<uint32_t>(dtype.lanes) << 16);
}

// Pack data type and array rank in a single int64_t so it could be used
// as a single constant for looking up the kernel in a family
int64_t pack_dtype_and_ndim(DLDataType dtype, size_t ndim) {
    uint64_t dtype_u = static_cast<uint64_t>(dtype_as_uint(dtype));
    return static_cast<int64_t>(dtype_u | (static_cast<uint64_t>(ndim) << 32));
}

struct KernelFamily : SimpleRefcount<KernelFamily> {
    using KernelMap = std::unordered_map<
            std::vector<ConstantArg>, CudaKernelHandle, HashVector<ConstantArg>>;

    KernelMap kernels_by_constants;
};

union CudaArg {
    void* device_ptr;
    uint16_t u16;
    int32_t i32;
    int64_t i64;
    float f32;
    double f64;
    struct {
        float real32;
        float imag32;
    } complex64;  // For complex64 (packed into 8 bytes)
};

CudaArg cuda_arg_device_ptr(void* ptr) {
    CudaArg arg = {};
    arg.device_ptr = ptr;
    return arg;
}

CudaArg cuda_arg_u16(uint16_t value) {
    CudaArg arg = {};
    arg.u16 = value;
    return arg;
}

CudaArg cuda_arg_i64(int64_t value) {
    CudaArg arg = {};
    arg.i64 = value;
    return arg;
}

CudaArg cuda_arg_f32(float value) {
    CudaArg arg = {};
    arg.f32 = value;
    return arg;
}

CudaArg cuda_arg_f64(double value) {
    CudaArg arg = {};
    arg.f64 = value;
    return arg;
}

// Metadata describing where each argument's data is stored in the flat cuargs vector
struct ArgMetadata {
    enum class Kind { Scalar, Array, TMADescriptor };
    Kind kind;
    size_t start_idx;  // Start index in LaunchHelper.cuargs
    size_t ndim;       // For arrays: number of dimensions; for scalars: 0

    bool is_array() const { return kind == Kind::Array; }
    bool is_tma_descriptor() const { return kind == Kind::TMADescriptor; }
};

// Info needed to copy scalar records back from device to host after kernel
struct RecordCopyInfo {
    void* host_ptr;
    CUdeviceptr device_ptr;
    size_t size;
};

struct LaunchHelper {
    std::vector<PyTypeObject*> pyarg_types;
    std::vector<ArgMetadata> arg_metadata;  // Metadata for each argument
    std::vector<CudaArg> cuargs;  // Flat storage: all ptrs, offsets, shapes, strides, scalars
    std::vector<void*> cuarg_pointers;  // Built just before launch
    std::vector<ConstantArg> constants;
    CUcontext cuda_context;
    LaunchHelper* next_free;
    std::vector<void*> aligned_tma_descriptors; // 128-byte aligned storage for TMA descriptors
    std::vector<RecordCopyInfo> record_copies; // Info for copying scalar records back to host

    ~LaunchHelper() {
        for (void* ptr : aligned_tma_descriptors) {
            if (ptr)
                free_aligned_storage(ptr);
        }
        for (const auto& info : record_copies) {
            if (info.device_ptr)
                g_cuMemFree(info.device_ptr);
        }
    }
};

LaunchHelper* g_helper_freelist;

#ifdef Py_GIL_DISABLED
std::mutex& helper_freelist_mutex() {
    static auto* mutex = new std::mutex;
    return *mutex;
}
#endif

struct LaunchHelperDeleter {
    void operator() (LaunchHelper* helper) const {
#ifdef Py_GIL_DISABLED
        std::lock_guard<std::mutex> guard(helper_freelist_mutex());
#endif
        helper->next_free = g_helper_freelist;
        g_helper_freelist = helper;
    }
};

using LaunchHelperPtr = std::unique_ptr<LaunchHelper, LaunchHelperDeleter>;


LaunchHelperPtr launch_helper_get() {
#ifdef Py_GIL_DISABLED
    std::lock_guard<std::mutex> guard(helper_freelist_mutex());
#endif
    if (g_helper_freelist) {
        LaunchHelper* ret = g_helper_freelist;
        g_helper_freelist = ret->next_free;
        return LaunchHelperPtr(ret);
    } else {
        return LaunchHelperPtr(new LaunchHelper());
    }
}

enum class PythonArgKind {
    // A torch.Tensor that we can access via torch._C._to_dlpack
    TorchTensorDlpack,
    // An object with __dlpack__ method
    DlpackArray,
    // An object with __cuda_array_interface__
    CudaArray,
    // Python `int`,
    PyLong,
    // Python `float`
    PyFloat,
    // numpy.float16 (half precision float)
    NpFloat16,
    // numpy.float32 (single precision float)
    NpFloat32,
    // Python `complex` or numpy.complex128 (both are complex128)
    PyComplex,
    // numpy.complex64 (single precision complex)
    NpComplex64,
    // A TMA descriptor
    TMADescriptor,
    // numpy.void (scalar record/structured dtype, host memory)
    NumpyVoid,
    // DeviceRecord (scalar record already on device)
    DeviceRecord,
    // Python Enum with int or float value (IntEnum is handled directly as PyLong)
    PyEnum,
    // numpy.datetime64 or numpy.timedelta64 scalar (stored as int64)
    NpDatetime,
};

enum class ParameterKind {
    Array,
    Integer,
    Float,
    Complex,
    Record
};

Result<std::pair<PythonArgKind, ParameterKind>> classify_arg(PyObject* arg) {
    if (PyLong_Check(arg))
        return {{PythonArgKind::PyLong, ParameterKind::Integer}};

    if (PyFloat_Check(arg))
        return {{PythonArgKind::PyFloat, ParameterKind::Float}};

    // Check for Python Enum types. IntEnum members pass PyLong_Check above,
    // but regular Enum members need special handling to extract their value.
    if (g_enum_Enum_type) {
        int is_enum = PyObject_IsInstance(arg, (PyObject*)g_enum_Enum_type);
        if (is_enum < 0)
            return ErrorRaised;
        if (is_enum) {
            PyObject* value = PyObject_GetAttrString(arg, "value");
            if (!value)
                return ErrorRaised;
            bool is_int = PyLong_Check(value);
            bool is_float = PyFloat_Check(value);
            const char* val_type_name = Py_TYPE(value)->tp_name;
            Py_DECREF(value);
            if (is_int)
                return {{PythonArgKind::PyEnum, ParameterKind::Integer}};
            if (is_float)
                return {{PythonArgKind::PyEnum, ParameterKind::Float}};
            // Enum with unsupported value type (tuple, str, etc.)
            return raise(PyExc_TypeError,
                "Enum members with %s values are not supported. "
                "Only integer and float valued enums are supported.",
                val_type_name);
        }
    }

    // Check for numpy scalar types by name
    const char* type_name = Py_TYPE(arg)->tp_name;

    // numpy.float16 (half precision float)
    if (type_name && strcmp(type_name, "numpy.float16") == 0)
        return {{PythonArgKind::NpFloat16, ParameterKind::Float}};

    // numpy.float32 (single precision float)
    if (type_name && strcmp(type_name, "numpy.float32") == 0)
        return {{PythonArgKind::NpFloat32, ParameterKind::Float}};

    // Check for numpy.complex64 before PyComplex_Check
    // numpy.complex64 is NOT a subclass of Python complex, but numpy.complex128 IS
    if (type_name && strcmp(type_name, "numpy.complex64") == 0)
        return {{PythonArgKind::NpComplex64, ParameterKind::Complex}};

    if (PyComplex_Check(arg))
        return {{PythonArgKind::PyComplex, ParameterKind::Complex}};

    if (g_torch_Tensor_type && PyObject_TypeCheck(arg, g_torch_Tensor_type)) {
        // Calling torch._C._to_dlpack(arg) is much faster than calling arg.__dlpack__()
        // because it goes straight into C++ code, with no Python in between.
        if (g_torch_to_dlpack_func)
            return {{PythonArgKind::TorchTensorDlpack, ParameterKind::Array}};
    }

    // Recognize TMA descriptors
    // Custom TMA descriptor wrappers have __tma_descriptor_c_api__()
    if (PyObject_HasAttrString(arg, "__tma_descriptor_c_api__"))
        return {{PythonArgKind::TMADescriptor, ParameterKind::Integer}};

    // cuda.bindings.driver.CUtensorMap from cuda-python
    if (type_name && strcmp(type_name, "cuda.bindings.driver.CUtensorMap") == 0) {
        if (PyObject_HasAttrString(arg, "getPtr") && PyObject_HasAttrString(arg, "opaque"))
            return {{PythonArgKind::TMADescriptor, ParameterKind::Integer}};
    }

    // numpy.datetime64 and numpy.timedelta64 scalars (stored as int64)
    if (type_name && (strcmp(type_name, "numpy.datetime64") == 0
                   || strcmp(type_name, "numpy.timedelta64") == 0))
        return {{PythonArgKind::NpDatetime, ParameterKind::Integer}};

    // Check for numpy.void or record (scalar record/structured dtype) before __cuda_array_interface__
    // numpy.void has __array_interface__ but not __cuda_array_interface__
    // Numba wraps numpy.void scalars in a type called "record"
    if (type_name && (strcmp(type_name, "numpy.void") == 0 || strcmp(type_name, "record") == 0))
        return {{PythonArgKind::NumpyVoid, ParameterKind::Record}};

    // DeviceRecord: a scalar record already resident in device memory.
    // It has __cuda_array_interface__ (inherited) but must be passed as a single pointer,
    // not as a memref descriptor — check before the generic __cuda_array_interface__ path.
    if (PyObject_HasAttrString(arg, "__device_record__"))
        return {{PythonArgKind::DeviceRecord, ParameterKind::Record}};

    // Check __cuda_array_interface__ first - CUDA arrays (including managed arrays)
    // may also have __dlpack__ from numpy inheritance, but we should use the CUDA
    // interface for better compatibility
    if (PyObject_HasAttr(arg, g___cuda_array_interface___pyunicode))
        return {{PythonArgKind::CudaArray, ParameterKind::Array}};

    if (PyObject_HasAttr(arg, g___dlpack___pyunicode))
        return {{PythonArgKind::DlpackArray, ParameterKind::Array}};

    PyErr_Format(PyExc_TypeError, "Unsupported argument type %s", Py_TYPE(arg)->tp_name);
    return ErrorRaised;
}


using ScalarExtractor = void(*)(PyObject*, CudaArg*);

inline void fast_extract_np_float32(PyObject* obj, CudaArg* out) {
    out->f32 = (float)PyFloat_AsDouble(obj);
}

inline void fast_extract_py_float(PyObject* obj, CudaArg* out) {
    out->f64 = PyFloat_AS_DOUBLE(obj);
}

inline void fast_extract_py_long(PyObject* obj, CudaArg* out) {
    out->i64 = PyLong_AsLongLong(obj);
}

ScalarExtractor get_scalar_extractor(PythonArgKind kind) {
    switch (kind) {
    case PythonArgKind::NpFloat32:  return fast_extract_np_float32;
    case PythonArgKind::PyFloat:    return fast_extract_py_float;
    case PythonArgKind::PyLong:     return fast_extract_py_long;
    default:                        return nullptr;
    }
}

struct PythonArgProfile {
    RefPtr<KernelFamily> family;
    std::vector<PythonArgKind> arg_kinds;

    // Populated on first use: per-arg fast extractors for all-scalar, no-constant profiles.
    // Empty if the profile contains arrays, TMA descriptors, or unsupported scalar types.
    std::vector<ScalarExtractor> fast_extractors;
    bool fast_path_checked = false;

    void maybe_init_fast_path(const std::vector<bool>& constant_flags) {
        if (fast_path_checked) return;
        fast_path_checked = true;

        // Only enable if no arg is constant and all args have a fast extractor
        std::vector<ScalarExtractor> extractors;
        extractors.reserve(arg_kinds.size());
        for (size_t i = 0; i < arg_kinds.size(); ++i) {
            if (i < constant_flags.size() && constant_flags[i])
                return;  // constant args need the full path
            ScalarExtractor ex = get_scalar_extractor(arg_kinds[i]);
            if (!ex) return;  // unsupported type (array, complex, etc.)
            extractors.push_back(ex);
        }
        fast_extractors = std::move(extractors);
    }
};

// Concatenate values of two chars in a single unsigned integer
#define CHAR_PAIR(x, y) \
    (\
        (static_cast<unsigned>(static_cast<unsigned char>((x))) << 16) \
        | (static_cast<unsigned>(static_cast<unsigned char>((y)))) \
    )


Result<DLDataType> parse_typestr(PyObject* typestr) {
    if (!PyUnicode_Check(typestr)) {
        PyErr_SetString(PyExc_TypeError, "__cuda_array_interface__['typestr'] is not a string");
        return ErrorRaised;
    }

    Py_ssize_t len;
    const char* str = PyUnicode_AsUTF8AndSize(typestr, &len);
    if (!str) return ErrorRaised;

    if (len < 3) {
        PyErr_Format(PyExc_TypeError, "__cuda_array_interface__['typestr'] has invalid value %S",
                     typestr);
        return ErrorRaised;
    }

    // TODO: support big endian one day?
    if (str[0] != '<' && str[0] != '|') {
        PyErr_SetString(PyExc_TypeError, "Only little-endian types are supported");
        return ErrorRaised;
    }

    DLDataType ret;
    ret.lanes = 1;

    switch (str[1]) {
    case 'b': ret.code = kDLBool; break;
    case 'i': ret.code = kDLInt; break;
    case 'u': ret.code = kDLUInt; break;
    case 'f': ret.code = kDLFloat; break;
    case 'V': ret.code = kDLOpaqueHandle; break;  // void/record type
    case 'c': ret.code = kDLComplex; break;
    case 'M': ret.code = kDLInt; break;  // datetime64 - stored as int64
    case 'm': ret.code = kDLInt; break;  // timedelta64 - stored as int64
    case 'S': ret.code = kDLOpaqueHandle; break;  // byte string (CharSeq) - opaque bytes
    case 'U': ret.code = kDLOpaqueHandle; break;  // unicode string (UnicodeCharSeq) - opaque bytes
    default:
        PyErr_Format(PyExc_TypeError, "Unsupported type code %c", str[1]);
        return ErrorRaised;
    }

    // For void/record ('V'), byte string ('S'), and unicode string ('U') types,
    // parse arbitrary byte sizes. The array is treated as raw bytes.
    if (str[1] == 'V' || str[1] == 'S' || str[1] == 'U') {
        char* endptr;
        long size = strtol(str + 2, &endptr, 10);
        if (endptr == str + 2 || *endptr != '\0' || size <= 0) {
            PyErr_Format(PyExc_TypeError, "Invalid byte size in typestr: %s", str + 2);
            return ErrorRaised;
        }
        // Opaque byte types: lanes=0 marks this as an opaque type (stride
        // computation uses dtype_bytewidth=1, treating the memref as i8).
        // bits+lanes encode the element byte size for potential future use:
        //   bits  = size & 0xFF   (low 8 bits)
        //   lanes = size >> 8     (high bits, up to 65535*256 + 255 = ~16M)
        ret.bits = static_cast<uint8_t>(size & 0xFF);
        ret.lanes = static_cast<uint16_t>(size >> 8);
        return ret;
    }

    // datetime64/timedelta64 typestr has a unit suffix like "<M8[D]" or "<m8[ns]".
    // Parse only the byte count and ignore the unit.
    if (str[1] == 'M' || str[1] == 'm') {
        char* endptr;
        long size = strtol(str + 2, &endptr, 10);
        if (endptr == str + 2 || size <= 0) {
            PyErr_Format(PyExc_TypeError, "Invalid byte size in typestr: %s", str + 2);
            return ErrorRaised;
        }
        ret.bits = static_cast<uint8_t>(size * 8);
        return ret;
    }

    // str[3] is safe to index because there is always a NUL byte at the end
    switch (CHAR_PAIR(str[2], str[3])) {
    case CHAR_PAIR('1', '\0'): ret.bits = 8; break;
    case CHAR_PAIR('2', '\0'): ret.bits = 16; break;
    case CHAR_PAIR('4', '\0'): ret.bits = 32; break;
    case CHAR_PAIR('8', '\0'): ret.bits = 64; break;
    case CHAR_PAIR('1', '6'):
        if (!str[4]) {
            ret.bits = 128;
            break;
        }
        [[fallthrough]];
    default:
        PyErr_Format(PyExc_TypeError, "Unsupported byte size in typestr: %s", str + 2);
        return ErrorRaised;
    }

    // Bits should be per component for complex types (half the total bits)
    if (ret.code == kDLComplex) {
        ret.bits /= 2;
    }

    return ret;
}


Result<std::string> array_dtype_identity_constant(PyObject* pyobj, PyObject* typestr) {
    Py_ssize_t len;
    const char* str = PyUnicode_AsUTF8AndSize(typestr, &len);
    if (!str) return ErrorRaised;

    if (len < 2) return std::string();

    // DLDataType loses schema/unit information for opaque records and
    // datetime-like arrays, so include the Python dtype in the kernel key.
    switch (str[1]) {
    case 'V':
    case 'S':
    case 'U':
    case 'M':
    case 'm':
        break;
    default:
        return std::string();
    }

    PyPtr dtype = steal(PyObject_GetAttrString(pyobj, "dtype"));
    if (dtype) {
        PyPtr repr = steal(PyObject_Repr(dtype.get()));
        if (!repr) return ErrorRaised;

        Py_ssize_t repr_len;
        const char* repr_str = PyUnicode_AsUTF8AndSize(repr.get(), &repr_len);
        if (!repr_str) return ErrorRaised;
        return std::string(repr_str, repr_len);
    }

    PyErr_Clear();
    return std::string(str, len);
}


Status compute_compact_row_major_strides(const std::vector<int64_t>& shapes,
                                          std::vector<int64_t>& strides) {
    size_t ndim = shapes.size();
    if (ndim == 0) return OK;

    strides.resize(ndim);
    strides[ndim - 1] = 1;

    for (size_t i = 0; i < ndim - 1; ++i) {
        size_t idx = ndim - 2 - i;
        strides[idx] = strides[idx + 1] * shapes[idx + 1];
        if (strides[idx] < INT64_MIN || strides[idx] > INT64_MAX)
            return raise(PyExc_OverflowError, "stride is too big");
    }

    return OK;
}


#define UNPACK_ARRAY_INTERFACE(dict, key) \
    PyObject* key = PyDict_GetItemWithError((dict).get(), g_##key##_pyunicode); \
    if (!key) { \
        if (!PyErr_Occurred()) \
            PyErr_SetString(PyExc_TypeError, \
                            "__cuda_array_interface__ is missing the '" #key "' key"); \
        return ErrorRaised; \
    }


#define ASSERT_NDIM(ndim) \
    if (static_cast<uintmax_t>(ndim) > UINT32_MAX) \
        return raise(PyExc_TypeError, "Input array exceeds max supported dimensions: %ld > %u", \
                     ndim, UINT32_MAX);


Status extract_cuda_array(PyObject* pyobj, LaunchHelper& helper) {
    PyPtr dict = steal(PyObject_GetAttr(pyobj, g___cuda_array_interface___pyunicode));
    if (!PyDict_Check(dict.get())) {
        PyErr_SetString(PyExc_TypeError,
                        "__cuda_array_interface__ returned a non-dictionary object");
        return ErrorRaised;
    }

    UNPACK_ARRAY_INTERFACE(dict, typestr);
    UNPACK_ARRAY_INTERFACE(dict, shape);
    UNPACK_ARRAY_INTERFACE(dict, data);

    // Parse the dtype
    Result<DLDataType> dtype = parse_typestr(typestr);
    if (!dtype.is_ok()) return ErrorRaised;

    // Parse the data pointer
    if (!PyTuple_Check(data) || PyTuple_GET_SIZE(data) != 2) {
        PyErr_SetString(PyExc_TypeError,
                        "__cuda_array_interface['data'] is not a tuple of length 2");
        return ErrorRaised;
    }

    PyObject* data_ptr_pylong = PyTuple_GET_ITEM(data, 0);
    if (!PyLong_Check(data_ptr_pylong)) {
        PyErr_SetString(PyExc_TypeError, "__cuda_array_interface['data'][0] is not an integer");
        return ErrorRaised;
    }

    intptr_t data_ptr_int = pylong_as<intptr_t>(data_ptr_pylong);
    if (PyErr_Occurred()) return ErrorRaised;

    void* data_ptr = reinterpret_cast<void*>(data_ptr_int);

    if (!helper.cuda_context)
        g_cuPointerGetAttribute(&helper.cuda_context, CU_POINTER_ATTRIBUTE_CONTEXT,
                                reinterpret_cast<CUdeviceptr>(data_ptr));

    Py_ssize_t ndim = PyTuple_GET_SIZE(shape);
    ASSERT_NDIM(ndim);

    // Parse the shape
    if (!PyTuple_Check(shape))
        return raise(PyExc_TypeError, "__cuda_array_interface['shape'] is not a tuple");

    std::vector<int64_t> shapes;
    shapes.reserve(ndim);
    for (Py_ssize_t i = 0; i < ndim; ++i) {
        int64_t size = pylong_as<int64_t>(PyTuple_GET_ITEM(shape, i));
        if (PyErr_Occurred()) return ErrorRaised;
        shapes.push_back(size);
    }

    // Parse the strides
    PyObject* strides = PyDict_GetItem(dict.get(), g_strides_pyunicode);
    if (PyErr_Occurred()) return ErrorRaised;

    std::vector<int64_t> strides_vec;
    strides_vec.reserve(ndim);
    if (!strides || strides == Py_None) {
        if(!compute_compact_row_major_strides(shapes, strides_vec))
            return ErrorRaised;
    } else if (PyTuple_Check(strides)) {
        // Opaque types (V/S/U) use memref<?xi8> so strides stay in bytes
        bool is_opaque_type = (dtype->code == kDLOpaqueHandle);
        uint8_t dtype_bytewidth = is_opaque_type ? 1 : (dtype->bits / BYTE_BITWIDTH);
        for (Py_ssize_t i = 0; i < ndim; ++i) {
            int64_t stride_bytes = pylong_as<int64_t>(PyTuple_GET_ITEM(strides, i));
            if (PyErr_Occurred()) return ErrorRaised;
            strides_vec.push_back(stride_bytes / dtype_bytewidth);
        }
    } else {
        return raise(PyExc_TypeError, "__cuda_array_interface['strides'] can only be"
                                      " absent, None, or a tuple");
    }

    // Store in flat cuargs vector: ptr, ptr, offset, shapes..., strides...
    size_t start_idx = helper.cuargs.size();
    helper.cuargs.push_back(cuda_arg_device_ptr(data_ptr));  // allocated ptr
    helper.cuargs.push_back(cuda_arg_device_ptr(data_ptr));  // aligned ptr
    helper.cuargs.push_back(cuda_arg_i64(0));  // offset
    for (int64_t dim : shapes) {
        helper.cuargs.push_back(cuda_arg_i64(dim));
    }
    for (int64_t stride : strides_vec) {
        helper.cuargs.push_back(cuda_arg_i64(stride));
    }

    helper.arg_metadata.push_back({ArgMetadata::Kind::Array, start_idx, static_cast<size_t>(ndim)});
    helper.constants.push_back(ConstantArg(pack_dtype_and_ndim(*dtype, ndim)));
    Result<std::string> dtype_identity = array_dtype_identity_constant(pyobj, typestr);
    if (!dtype_identity.is_ok()) return ErrorRaised;
    if (!dtype_identity->empty())
        helper.constants.push_back(ConstantArg(std::move(*dtype_identity)));
    return OK;
}

Status extract_dlpack_common(PyObject* dlpack_capsule, LaunchHelper& helper) {
    void* ptr = PyCapsule_GetPointer(dlpack_capsule, "dltensor");
    if (!ptr) return ErrorRaised;
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr);

    if (tensor->dl_tensor.device.device_type != kDLCUDA)
        return raise(PyExc_ValueError, "Input array is not on a CUDA device");

    // TODO: check device ID

    void* data_ptr = static_cast<char*>(tensor->dl_tensor.data) + tensor->dl_tensor.byte_offset;

    if (!helper.cuda_context)
        g_cuPointerGetAttribute(&helper.cuda_context, CU_POINTER_ATTRIBUTE_CONTEXT,
                                reinterpret_cast<CUdeviceptr>(data_ptr));

    int32_t ndim = tensor->dl_tensor.ndim;
    ASSERT_NDIM(ndim);

    std::vector<int64_t> shapes;
    shapes.reserve(ndim);
    for (int32_t i = 0; i < ndim; ++i) {
        shapes.push_back(tensor->dl_tensor.shape[i]);
    }

    std::vector<int64_t> strides_vec;
    strides_vec.reserve(ndim);
    if (!tensor->dl_tensor.strides) {
        if (!compute_compact_row_major_strides(shapes, strides_vec))
            return ErrorRaised;
    } else {
        for (int32_t i = 0; i < ndim; ++i) {
            strides_vec.push_back(tensor->dl_tensor.strides[i]);
        }
    }

    // Store in flat cuargs vector: ptr, ptr, offset, shapes..., strides...
    size_t start_idx = helper.cuargs.size();
    helper.cuargs.push_back(cuda_arg_device_ptr(data_ptr));  // allocated ptr
    helper.cuargs.push_back(cuda_arg_device_ptr(data_ptr));  // aligned ptr
    helper.cuargs.push_back(cuda_arg_i64(0));  // offset
    for (int64_t dim : shapes) {
        helper.cuargs.push_back(cuda_arg_i64(dim));
    }
    for (int64_t stride : strides_vec) {
        helper.cuargs.push_back(cuda_arg_i64(stride));
    }

    helper.arg_metadata.push_back({ArgMetadata::Kind::Array, start_idx, static_cast<size_t>(ndim)});
    helper.constants.push_back(ConstantArg(pack_dtype_and_ndim(tensor->dl_tensor.dtype, ndim)));

    PyCapsule_SetName(dlpack_capsule, "used_dltensor");

    // We assume that __dlpack__ returns a view of the tensor,
    // so we release the capsule immediately. This should be OK for using with PyTorch
    // since it always returns a view.
    //
    // This is technically an incorrect implementation. To do it correctly, we would
    // need to implement a mechanism similar to the one found in Torch's CUDACachingAllocator:
    // instead of calling the deleter immediately, we would push a cudaEvent to the stream
    // after we launch the kernel, and only call the deleter once the event is ready.
    tensor->deleter(tensor);
    return OK;
}

Status extract_torch_tensor_dlpack(PyObject* pyobj, LaunchHelper& helper) {
    PyPtr dlpack_capsule = steal(PyObject_CallFunctionObjArgs(
                g_torch_to_dlpack_func, pyobj, nullptr));
    if (!dlpack_capsule) return ErrorRaised;

    return extract_dlpack_common(dlpack_capsule.get(), helper);
}

Status extract_dlpack(PyObject* pyobj, LaunchHelper& helper) {
    PyPtr dlpack_method = steal(PyObject_GetAttr(pyobj, g___dlpack___pyunicode));
    if (!dlpack_method) return ErrorRaised;

    PyPtr empty_args = steal(PyTuple_New(0));
    if (!empty_args) return ErrorRaised;

    PyPtr kwargs = steal(PyDict_New());
    if (!kwargs) return ErrorRaised;

    // stream -1 signals "producer must not perform any synchronization"
    PyPtr stream_value = steal(PyLong_FromLong(-1));
    if (!stream_value) return ErrorRaised;
    PyDict_SetItemString(kwargs.get(), "stream", stream_value.get());

    PyPtr dlpack_capsule = steal(PyObject_Call(
                dlpack_method.get(), empty_args.get(), kwargs.get()));
    if (!dlpack_capsule) return ErrorRaised;

    return extract_dlpack_common(dlpack_capsule.get(), helper);
}

inline Status extract_np_datetime(PyObject* pyobj, bool is_constant, LaunchHelper& helper) {
    // numpy.datetime64/timedelta64 are stored as int64 internally.
    // Call view() with the cached "int64" PyUnicode to get the numpy.int64
    // scalar, then extract.
    PyPtr int_view = steal(PyObject_CallMethod(pyobj, "view", "O", g_int64_pyunicode));
    if (!int_view) return ErrorRaised;
    int64_t value = pylong_as<int64_t>(int_view.get());
    if (PyErr_Occurred()) return ErrorRaised;

    size_t start_idx = helper.cuargs.size();
    helper.cuargs.push_back(cuda_arg_i64(value));
    helper.arg_metadata.push_back({ArgMetadata::Kind::Scalar, start_idx, 0});

    if (is_constant)
        helper.constants.push_back(ConstantArg(value));

    return OK;
}

inline Status extract_py_long(PyObject* pyobj, bool is_constant, LaunchHelper& helper) {
    int64_t value = pylong_as<int64_t>(pyobj);
    if (PyErr_Occurred()) return ErrorRaised;

    size_t start_idx = helper.cuargs.size();
    helper.cuargs.push_back(cuda_arg_i64(value));
    helper.arg_metadata.push_back({ArgMetadata::Kind::Scalar, start_idx, 0});

    if (is_constant)
        helper.constants.push_back(ConstantArg(value));

    return OK;
}

inline Status extract_py_enum(PyObject* pyobj, bool is_constant, LaunchHelper& helper) {
    PyObject* value_attr = PyObject_GetAttrString(pyobj, "value");
    if (!value_attr) return ErrorRaised;

    size_t start_idx = helper.cuargs.size();
    PyObject* enum_type = (PyObject*)Py_TYPE(pyobj);
    if (is_constant)
        helper.constants.push_back(ConstantArg(reinterpret_cast<int64_t>(enum_type)));

    if (PyLong_Check(value_attr)) {
        int64_t value = pylong_as<int64_t>(value_attr);
        Py_DECREF(value_attr);
        if (PyErr_Occurred()) return ErrorRaised;
        helper.cuargs.push_back(cuda_arg_i64(value));
        if (is_constant)
            helper.constants.push_back(ConstantArg(value));
    } else {
        double value = PyFloat_AsDouble(value_attr);
        Py_DECREF(value_attr);
        if (PyErr_Occurred()) return ErrorRaised;
        helper.cuargs.push_back(cuda_arg_f64(value));
        if (is_constant)
            helper.constants.push_back(ConstantArg(value));
    }

    helper.arg_metadata.push_back({ArgMetadata::Kind::Scalar, start_idx, 0});
    return OK;
}

Status extract_tma_descriptor(PyObject* pyobj, LaunchHelper& helper) {
  PyObject *res = nullptr;

  // Try __tma_descriptor_c_api__() first (custom wrapper objects)
  if (PyObject_HasAttrString(pyobj, "__tma_descriptor_c_api__")) {
    res = PyObject_CallMethod(pyobj, "__tma_descriptor_c_api__", nullptr);
    if (PyErr_Occurred())
      return ErrorRaised;
  }
  // Fall back to getPtr() for cuda.bindings.driver.CUtensorMap
  else if (PyObject_HasAttrString(pyobj, "getPtr")) {
    res = PyObject_CallMethod(pyobj, "getPtr", nullptr);
    if (PyErr_Occurred())
      return ErrorRaised;
  }
  else {
    return raise(PyExc_TypeError,
                 "TMA descriptor must have either __tma_descriptor_c_api__() or getPtr() method");
  }

  if (!PyLong_Check(res))
    return raise(PyExc_TypeError,
                 "TMA descriptor method returned a non-integer");

  intptr_t source_ptr = pylong_as<intptr_t>(res);

  // TMA descriptors are 128 bytes and MUST be 128-byte aligned
  const size_t TMA_DESCRIPTOR_SIZE = 128;

  // Allocate 128-byte aligned storage
  void* aligned_ptr = allocate_aligned_storage(128, TMA_DESCRIPTOR_SIZE);
  if (!aligned_ptr) {
    return raise(PyExc_MemoryError, "Failed to allocate aligned memory for TMA descriptor");
  }

  // Copy the 128-byte descriptor to aligned storage
  std::memcpy(aligned_ptr, reinterpret_cast<void*>(source_ptr), TMA_DESCRIPTOR_SIZE);

  // Store the pointer so we can free it later
  helper.aligned_tma_descriptors.push_back(aligned_ptr);

  // Store the aligned pointer
  helper.cuargs.push_back(cuda_arg_device_ptr(aligned_ptr));
  helper.arg_metadata.push_back(
      {ArgMetadata::Kind::TMADescriptor, helper.cuargs.size() - 1, 0});

  return OK;
}

void extract_py_float(PyObject* pyobj, bool is_constant, LaunchHelper& helper) {
    double value = PyFloat_AS_DOUBLE(pyobj);

    size_t start_idx = helper.cuargs.size();
    helper.cuargs.push_back(cuda_arg_f64(value));
    helper.arg_metadata.push_back({ArgMetadata::Kind::Scalar, start_idx, 0});

    if (is_constant)
        helper.constants.push_back(ConstantArg(value));
}

static uint16_t double_to_f16_bits(double d) {
    float f = (float)d;
    uint32_t fbits;
    memcpy(&fbits, &f, sizeof(fbits));

    uint16_t sign = (fbits >> 16) & 0x8000;
    int32_t exp = ((fbits >> 23) & 0xFF) - 127;
    uint32_t mant = fbits & 0x7FFFFF;

    if (exp == 128) {
        return static_cast<uint16_t>(sign | 0x7C00 | (mant ? ((mant >> 13) | 1) : 0));
    } else if (exp > 15) {
        return static_cast<uint16_t>(sign | 0x7C00);
    } else if (exp > -15) {
        return static_cast<uint16_t>(sign | ((uint16_t)(exp + 15) << 10) | (uint16_t)(mant >> 13));
    } else if (exp >= -24) {
        mant |= 0x800000;
        return static_cast<uint16_t>(sign | (uint16_t)(mant >> (-exp - 14 + 23)));
    }
    return sign;
}

Status extract_np_float16(PyObject* pyobj, bool is_constant, LaunchHelper& helper) {
    PyPtr py_float = steal(PyNumber_Float(pyobj));
    if (!py_float) return ErrorRaised;
    double value = PyFloat_AS_DOUBLE(py_float.get());

    size_t start_idx = helper.cuargs.size();
    helper.cuargs.push_back(cuda_arg_u16(double_to_f16_bits(value)));
    helper.arg_metadata.push_back({ArgMetadata::Kind::Scalar, start_idx, 0});

    if (is_constant)
        helper.constants.push_back(ConstantArg(value));
    return OK;
}

void extract_np_float32(PyObject* pyobj, bool is_constant, LaunchHelper& helper) {
    // numpy.float32 scalar - extract as 32-bit float
    // Use PyFloat_AsDouble to convert numpy scalar to C double, then cast to float
    float value = (float)PyFloat_AsDouble(pyobj);

    size_t start_idx = helper.cuargs.size();
    helper.cuargs.push_back(cuda_arg_f32(value));
    // Use ndim=3 to indicate this is a 32-bit float (distinguish from 64-bit which uses ndim=0)
    helper.arg_metadata.push_back({ArgMetadata::Kind::Scalar, start_idx, 3});

    if (is_constant)
        helper.constants.push_back(ConstantArg((double)value));
}

void extract_py_complex(PyObject* pyobj, bool is_constant, LaunchHelper& helper) {
    // Python complex uses two doubles internally (complex128)
    // Push real and imaginary parts as consecutive f64 values
    double real = PyComplex_RealAsDouble(pyobj);
    double imag = PyComplex_ImagAsDouble(pyobj);

    size_t start_idx = helper.cuargs.size();
    helper.cuargs.push_back(cuda_arg_f64(real));
    helper.cuargs.push_back(cuda_arg_f64(imag));
    // Complex128 is 2 f64 scalars, use ndim=2 to indicate complex128 (2 doubles)
    helper.arg_metadata.push_back({ArgMetadata::Kind::Scalar, start_idx, 2});

    // Note: constants for complex not yet supported
    (void)is_constant;
}

void extract_np_complex64(PyObject* pyobj, bool is_constant, LaunchHelper& helper) {
    // numpy.complex64 stores two float32 values packed into 8 bytes
    // Extract via Python's __complex__ protocol, then convert to float
    float real, imag;
    PyObject* py_complex = PyObject_CallMethod(pyobj, "__complex__", nullptr);
    if (!py_complex) {
        PyErr_Clear();
        // Fallback: try to get real and imag attributes
        PyObject* real_obj = PyObject_GetAttrString(pyobj, "real");
        PyObject* imag_obj = PyObject_GetAttrString(pyobj, "imag");
        real = real_obj ? (float)PyFloat_AsDouble(real_obj) : 0.0f;
        imag = imag_obj ? (float)PyFloat_AsDouble(imag_obj) : 0.0f;
        Py_XDECREF(real_obj);
        Py_XDECREF(imag_obj);
    } else {
        real = (float)PyComplex_RealAsDouble(py_complex);
        imag = (float)PyComplex_ImagAsDouble(py_complex);
        Py_DECREF(py_complex);
    }

    // Pack both f32 values into a single 8-byte CudaArg entry
    // PTX expects one parameter pointing to 8 contiguous bytes
    size_t start_idx = helper.cuargs.size();
    CudaArg arg = {};
    arg.complex64.real32 = real;
    arg.complex64.imag32 = imag;
    helper.cuargs.push_back(arg);
    // ndim=1 indicates complex64 (one 8-byte entry containing two f32)
    helper.arg_metadata.push_back({ArgMetadata::Kind::Scalar, start_idx, 1});

    (void)is_constant;
}

Status extract_numpy_void(PyObject* pyobj, LaunchHelper& helper) {
    // numpy.void is a scalar record (structured dtype) on the HOST.
    // We need to allocate device memory, copy data there, and pass the device pointer.

    PyPtr dict = steal(PyObject_GetAttr(pyobj, g___array_interface___pyunicode));
    if (!dict) return ErrorRaised;

    if (!PyDict_Check(dict.get())) {
        PyErr_SetString(PyExc_TypeError, "__array_interface__ is not a dict");
        return ErrorRaised;
    }

    // Get the data pointer (host memory)
    PyObject* data = PyDict_GetItemWithError(dict.get(), g_data_pyunicode);
    if (!data) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_TypeError, "__array_interface__ is missing 'data'");
        return ErrorRaised;
    }

    if (!PyTuple_Check(data) || PyTuple_GET_SIZE(data) != 2) {
        PyErr_SetString(PyExc_TypeError, "__array_interface__['data'] is not a tuple of length 2");
        return ErrorRaised;
    }

    PyObject* data_ptr_pylong = PyTuple_GET_ITEM(data, 0);
    if (!PyLong_Check(data_ptr_pylong)) {
        PyErr_SetString(PyExc_TypeError, "__array_interface__['data'][0] is not an integer");
        return ErrorRaised;
    }

    intptr_t host_ptr_int = pylong_as<intptr_t>(data_ptr_pylong);
    if (PyErr_Occurred()) return ErrorRaised;
    void* host_ptr = reinterpret_cast<void*>(host_ptr_int);

    // Get record size from the itemsize attribute of the numpy.void object
    PyPtr itemsize_obj = steal(PyObject_GetAttrString(pyobj, "itemsize"));
    if (!itemsize_obj) return ErrorRaised;
    int64_t record_size = pylong_as<int64_t>(itemsize_obj.get());
    if (PyErr_Occurred()) return ErrorRaised;
    if (record_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "Record itemsize must be positive");
        return ErrorRaised;
    }

    // Allocate device memory for the record
    CUdeviceptr device_ptr = 0;
    CUresult err = g_cuMemAlloc(&device_ptr, record_size);
    if (err != CUDA_SUCCESS) {
        const char* error_name = nullptr;
        g_cuGetErrorName(err, &error_name);
        PyErr_Format(PyExc_RuntimeError, "Failed to allocate device memory for record: %s",
                     error_name ? error_name : "unknown error");
        return ErrorRaised;
    }

    // Copy record data from host to device
    err = g_cuMemcpyHtoD(device_ptr, host_ptr, record_size);
    if (err != CUDA_SUCCESS) {
        g_cuMemFree(device_ptr);
        const char* error_name = nullptr;
        g_cuGetErrorName(err, &error_name);
        PyErr_Format(PyExc_RuntimeError, "Failed to copy record to device: %s",
                     error_name ? error_name : "unknown error");
        return ErrorRaised;
    }

    // Save info for copying back after kernel and for cleanup
    helper.record_copies.push_back({host_ptr, device_ptr, static_cast<size_t>(record_size)});

    // Store the device pointer
    size_t start_idx = helper.cuargs.size();
    helper.cuargs.push_back(cuda_arg_device_ptr(reinterpret_cast<void*>(device_ptr)));
    helper.arg_metadata.push_back({ArgMetadata::Kind::Scalar, start_idx, 0});

    return OK;
}

Status extract_device_record(PyObject* pyobj, LaunchHelper& helper) {
    // DeviceRecord: data is already on device — extract pointer from __cuda_array_interface__
    // and pass it as a single pointer (no allocation or copy).
    PyPtr dict = steal(PyObject_GetAttr(pyobj, g___cuda_array_interface___pyunicode));
    if (!dict || !PyDict_Check(dict.get())) {
        PyErr_SetString(PyExc_TypeError, "DeviceRecord missing __cuda_array_interface__");
        return ErrorRaised;
    }

    PyObject* data = PyDict_GetItemWithError(dict.get(), g_data_pyunicode);
    if (!data) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_TypeError, "__cuda_array_interface__ is missing 'data'");
        return ErrorRaised;
    }
    if (!PyTuple_Check(data) || PyTuple_GET_SIZE(data) != 2) {
        PyErr_SetString(PyExc_TypeError, "__cuda_array_interface__['data'] is not a 2-tuple");
        return ErrorRaised;
    }

    PyObject* data_ptr_pylong = PyTuple_GET_ITEM(data, 0);
    if (!PyLong_Check(data_ptr_pylong)) {
        PyErr_SetString(PyExc_TypeError, "__cuda_array_interface__['data'][0] is not an integer");
        return ErrorRaised;
    }

    intptr_t device_ptr_int = pylong_as<intptr_t>(data_ptr_pylong);
    if (PyErr_Occurred()) return ErrorRaised;
    void* device_ptr = reinterpret_cast<void*>(device_ptr_int);

    if (!helper.cuda_context)
        g_cuPointerGetAttribute(&helper.cuda_context, CU_POINTER_ATTRIBUTE_CONTEXT,
                                reinterpret_cast<CUdeviceptr>(device_ptr));

    size_t start_idx = helper.cuargs.size();
    helper.cuargs.push_back(cuda_arg_device_ptr(device_ptr));
    helper.arg_metadata.push_back({ArgMetadata::Kind::Scalar, start_idx, 0});
    return OK;
}

Status extract_cuda_args(PyObject* const* pyargs, size_t num_pyargs,
                         const std::vector<PythonArgKind>& arg_kinds,
                         const std::vector<bool>& constant_arg_flags,
                         LaunchHelper& helper) {
    assert(num_pyargs == arg_kinds.size());
    helper.cuargs.clear();
    helper.arg_metadata.clear();
    helper.constants.clear();

    // Free old TMA descriptor allocations from previous launch
    for (void* ptr : helper.aligned_tma_descriptors) {
        if (ptr) free_aligned_storage(ptr);
    }
    helper.aligned_tma_descriptors.clear();

    // Free old record device allocations from previous launch
    for (const auto& info : helper.record_copies) {
        if (info.device_ptr) g_cuMemFree(info.device_ptr);
    }
    helper.record_copies.clear();

    helper.cuda_context = nullptr;
    for (size_t i = 0; i < num_pyargs; ++i) {
        PyObject* pyobj = pyargs[i];
        bool is_constant = constant_arg_flags[i];

        switch (arg_kinds[i]) {
        case PythonArgKind::TorchTensorDlpack:
            if (!extract_torch_tensor_dlpack(pyobj, helper)) return ErrorRaised;
            break;
        case PythonArgKind::DlpackArray:
            if (!extract_dlpack(pyobj, helper)) return ErrorRaised;
            break;
        case PythonArgKind::CudaArray:
            if (!extract_cuda_array(pyobj, helper)) return ErrorRaised;
            break;
        case PythonArgKind::TMADescriptor:
            if (!extract_tma_descriptor(pyobj, helper)) return ErrorRaised;
            break;
        case PythonArgKind::PyLong:
            if (!extract_py_long(pyobj, is_constant, helper)) return ErrorRaised;
            break;
        case PythonArgKind::PyEnum:
            if (!extract_py_enum(pyobj, is_constant, helper)) return ErrorRaised;
            break;
        case PythonArgKind::PyFloat:
            extract_py_float(pyobj, is_constant, helper);
            break;
        case PythonArgKind::NpFloat16:
            if (!extract_np_float16(pyobj, is_constant, helper)) return ErrorRaised;
            break;
        case PythonArgKind::NpFloat32:
            extract_np_float32(pyobj, is_constant, helper);
            break;
        case PythonArgKind::PyComplex:
            extract_py_complex(pyobj, is_constant, helper);
            break;
        case PythonArgKind::NpComplex64:
            extract_np_complex64(pyobj, is_constant, helper);
            break;
        case PythonArgKind::NumpyVoid:
            if (!extract_numpy_void(pyobj, helper)) return ErrorRaised;
            break;
        case PythonArgKind::DeviceRecord:
            if (!extract_device_record(pyobj, helper)) return ErrorRaised;
            break;
        case PythonArgKind::NpDatetime:
            if (!extract_np_datetime(pyobj, is_constant, helper)) return ErrorRaised;
            break;
        }
    }
    return OK;
}

class ProfileMap {
    // NOTE: ideally, we'd store a std::vector<PyPtr> that owns references to type objects,
    // but std::unordered_map doesn't support heterogenous lookup.
    // So we manually increase the reference counts when inserting into the map.
    using Map = std::unordered_map<
            std::vector<PyTypeObject*>, PythonArgProfile, HashVector<PyTypeObject*>>;
    Map map_;

public:
    ProfileMap() = default;

    ProfileMap(const ProfileMap&) = delete;
    void operator=(const ProfileMap&) = delete;

    ~ProfileMap() {
        for (auto& e : map_) {
            for (PyTypeObject* obj : e.first)
                Py_DECREF(obj);
        }
    }

    PythonArgProfile* find(const std::vector<PyTypeObject*>& types) {
        Map::iterator it = map_.find(types);
        return it == map_.end() ? nullptr : &it->second;
    }

    PythonArgProfile* insert(std::vector<PyTypeObject*>&& types, PythonArgProfile&& profile) {
        auto [it, inserted] = map_.emplace(
                std::make_pair(std::move(types), std::move(profile)));
        CHECK(inserted);
        for (PyTypeObject* obj : it->first)
            Py_INCREF(obj);
        return &it->second;
    }
};

struct KernelDispatcher {
    using FamilyMap = std::unordered_map<
            std::vector<ParameterKind>,
            RefPtr<KernelFamily>,
            HashVector<ParameterKind>>;

    PyPtr compile_func;
    PyPtr ensure_context_func;
    std::vector<bool> constant_arg_flags;
    ProfileMap arg_profiles;
    FamilyMap kernel_families;
};

void get_pyarg_types(PyObject* const* pyargs, Py_ssize_t num_pyargs,
                     std::vector<PyTypeObject*>& pyarg_types) {
    pyarg_types.clear();
    for (Py_ssize_t i = 0; i < num_pyargs; ++i)
        pyarg_types.push_back(Py_TYPE(pyargs[i]));
}

Result<CudaKernelHandle> compile(PyObject* compile_func,
                           PyObject* const* pyargs, Py_ssize_t num_pyargs) {
    PyPtr pyargs_tuple = steal(PyTuple_New(num_pyargs));
    if (!pyargs_tuple) return ErrorRaised;

    for (Py_ssize_t i = 0; i < num_pyargs; ++i)
        PyTuple_SET_ITEM(pyargs_tuple.get(), i, Py_NewRef(pyargs[i]));

    PyPtr compile_result = steal(
            PyObject_CallFunctionObjArgs(compile_func, pyargs_tuple.get(), nullptr));
    if (!compile_result) return ErrorRaised;

    if (!PyTuple_Check(compile_result.get()))
        return raise(PyExc_TypeError, "Expected compile() to return a tuple, got %s",
                     Py_TYPE(compile_result.get())->tp_name);

    Py_ssize_t tuple_size = PyTuple_GET_SIZE(compile_result.get());
    if (tuple_size < 3 || tuple_size > 5)
        return raise(PyExc_TypeError, "Expected compile() to return a 3- to 5-tuple, got length %zd",
                     tuple_size);

    PyObject* py_cubin = PyTuple_GET_ITEM(compile_result.get(), 0);
    PyObject* py_cufunc_name = PyTuple_GET_ITEM(compile_result.get(), 1);
    PyObject* py_cooperative = PyTuple_GET_ITEM(compile_result.get(), 2);

    if (!PyBytes_Check(py_cubin) || !PyUnicode_Check(py_cufunc_name))
        return raise(PyExc_TypeError,
                     "Expected compile() to return (bytes, str, ...) for (cubin, func_name, ...),"
                     " got %s, %s",
                     Py_TYPE(py_cubin)->tp_name,
                     Py_TYPE(py_cufunc_name)->tp_name);

    const void* cubin = PyBytes_AsString(py_cubin);
    if (!cubin) return ErrorRaised;

    const char* cufunc_name = PyUnicode_AsUTF8(py_cufunc_name);
    if (!cufunc_name) return ErrorRaised;

    bool cooperative = (py_cooperative == Py_True);

    Result<CudaKernel> cukernel = load_cuda_kernel(cubin, cufunc_name);
    if (!cukernel.is_ok()) return ErrorRaised;

    CudaKernelHandle handle{std::move(*cukernel), {}, cooperative};
    if (tuple_size >= 4) {
        PyObject* post_load_cb = PyTuple_GET_ITEM(compile_result.get(), 3);
        if (post_load_cb != Py_None && PyCallable_Check(post_load_cb))
            handle.post_load_callback = newref(post_load_cb);
    }
    if (tuple_size == 5) {
        PyObject* py_check_error_code = PyTuple_GET_ITEM(compile_result.get(), 4);
        handle.check_error_code = (py_check_error_code != Py_False);
    }
    return handle;
}

Status ensure_numba_context(PyObject* ensure_context_func) {
    if (!ensure_context_func || ensure_context_func == Py_None)
        return OK;

    PyPtr ctx = steal(PyObject_CallNoArgs(ensure_context_func));
    if (!ctx) return ErrorRaised;

    return OK;
}

inline bool has_torch_tensor_input(const std::vector<PyTypeObject*>& pyarg_types) {
    return std::any_of(pyarg_types.begin(), pyarg_types.end(), [](PyTypeObject* pytype) {
        return PyType_IsSubtype(pytype, g_torch_Tensor_type);
    });
}

inline bool has_cupy_array_input(const std::vector<PyTypeObject*>& pyarg_types) {
    return std::any_of(pyarg_types.begin(), pyarg_types.end(), [](PyTypeObject* pytype) {
        return PyType_IsSubtype(pytype, g_cupy_ndarray_type);
    });
}

Result<CUstream> parse_stream(PyObject* py_stream) {
    PyPtr py_raw_stream;
    if (g_torch_cuda_Stream_type && PyObject_TypeCheck(py_stream, g_torch_cuda_Stream_type)) {
        py_raw_stream = getattr(py_stream, "cuda_stream");
        if (!py_raw_stream) return ErrorRaised;

    } else if (g_cupy_cuda_Stream_type && PyObject_TypeCheck(py_stream, g_cupy_cuda_Stream_type)) {
        py_raw_stream = getattr(py_stream, "ptr");
        if (!py_raw_stream) return ErrorRaised;

    } else if (g_numba_cuda_Stream_type
            && PyObject_TypeCheck(py_stream, g_numba_cuda_Stream_type)) {
        PyPtr py_stream_handle = getattr(py_stream, "handle");
        if (!py_stream_handle) return ErrorRaised;

        if (py_stream_handle.get() == Py_None)
            return static_cast<CUstream>(nullptr);

        // cuda.bindings.driver.CUstream supports int() but not .value
        PyPtr py_stream_int = steal(PyNumber_Long(py_stream_handle.get()));
        if (!py_stream_int) return ErrorRaised;
        py_raw_stream = std::move(py_stream_int);

    } else if (g_cuda_core_Stream_type
            && PyObject_TypeCheck(py_stream, g_cuda_core_Stream_type)) {
        // cuda.core.Stream has a handle property that returns the raw CUstream
        PyPtr py_stream_handle = getattr(py_stream, "handle");
        if (!py_stream_handle) return ErrorRaised;

        if (py_stream_handle.get() == Py_None)
            return static_cast<CUstream>(nullptr);

        // The handle is a cuda.bindings.driver.CUstream which supports int()
        PyPtr py_stream_int = steal(PyNumber_Long(py_stream_handle.get()));
        if (!py_stream_int) return ErrorRaised;
        py_raw_stream = std::move(py_stream_int);

    } else if (PyLong_Check(py_stream)) {
        py_raw_stream = newref(py_stream);
    } else {
        return raise(PyExc_TypeError, "Unsupported stream type %s.",
                     Py_TYPE(py_stream)->tp_name);
    }

    if (!PyLong_Check(py_raw_stream.get())) {
        return raise(PyExc_TypeError, "Raw stream pointer must be a long, got %s",
                     Py_TYPE(py_raw_stream.get())->tp_name);
    }

    CUstream stream = static_cast<CUstream>(PyLong_AsVoidPtr(py_raw_stream.get()));
    if (PyErr_Occurred()) return ErrorRaised;

    return stream;
}

Result<CUstream> call_current_stream_function_obj(PyObject* current_stream_func_obj,
                                                  PyObject* device) {
    PyPtr py_stream = steal(PyObject_CallFunctionObjArgs(
        current_stream_func_obj, device, nullptr));
    if (!py_stream) return ErrorRaised;

    return parse_stream(py_stream.get());
}

Result<CUstream> get_torch_current_stream() {
    if (g_torch_cuda_getCurrentRawStream) {
        // device_index -1 means current device
        PyPtr device_index = steal(PyLong_FromLong(-1));
        if (!device_index) return ErrorRaised;

        return call_current_stream_function_obj(
            g_torch_cuda_getCurrentRawStream, device_index.get());
    } else {
        return raise(PyExc_RuntimeError,
                     "torch._C._cuda_getCurrentRawStream is not available");
    }
}

Result<CUstream> get_cupy_current_stream() {
    if (g_cupy_cuda_get_current_stream) {
        // device_index -1 means current device
        PyPtr device_index = steal(PyLong_FromLong(-1));
        if (!device_index) return ErrorRaised;

        return call_current_stream_function_obj(
            g_cupy_cuda_get_current_stream, device_index.get());
    } else {
        return raise(PyExc_RuntimeError,
                     "cupy.cuda.get_current_stream is not available");
    }
}

Result<CUstream> get_current_stream(const std::vector<PyTypeObject*>& pyarg_types) {
    if (g_torch_Tensor_type && has_torch_tensor_input(pyarg_types)) {
        return get_torch_current_stream();
    } else if (g_cupy_ndarray_type && has_cupy_array_input(pyarg_types)) {
        return get_cupy_current_stream();
    } else {
        // fallback to NULL stream
        return static_cast<CUstream>(nullptr);
    }
    // TODO: support other libraries
}

struct ContextGuard {
    bool need_to_pop;

    ContextGuard() : need_to_pop(false) {}

    ContextGuard(const ContextGuard&) = delete;
    void operator=(const ContextGuard&) = delete;

    ~ContextGuard() {
        if (need_to_pop) {
            CUcontext old;
            CUresult res = g_cuCtxPopCurrent(&old);
            CHECK(res == CUDA_SUCCESS);
        }
    }
};

Status maybe_switch_context(CUcontext target, ContextGuard& guard) {
    if (!target) return OK;

    CUcontext current;
    CUresult res = g_cuCtxGetCurrent(&current);
    if (res != CUDA_SUCCESS) {
        return raise(PyExc_RuntimeError, "Failed to get current CUDA context: %s",
                     get_cuda_error(res));
    }

    if (current == target) return OK;

    res = g_cuCtxPushCurrent(target);
    if (res != CUDA_SUCCESS) {
        return raise(PyExc_RuntimeError, "Failed to switch CUDA context: %s",
                     get_cuda_error(res));
    }

    guard.need_to_pop = true;
    return OK;
}

struct Grid {
    enum { Len = 3 };
    unsigned dims[Len];
};

bool try_clarify_invalid_value_error(const Grid& grid) {
    // If we can't read the data we need to try to clarify, this is a no-op.
    CUdevice dev;
    if (g_cuCtxGetDevice(&dev) != CUDA_SUCCESS) return false;

    for (int i = 0; i < Grid::Len; ++i) {
        int v;
        CUdevice_attribute attr = static_cast<CUdevice_attribute>(
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X + i
        );
        if (g_cuDeviceGetAttribute(&v, attr, dev) != CUDA_SUCCESS) return false;

        if (grid.dims[i] > static_cast<unsigned>(v)) {
            raise(PyExc_ValueError, "Grid[%d] is too big: max=%d, got=%lu",
                  i, v, grid.dims[i]);
            return true;
        }
    }
    return false;
}

bool try_clarify_launch_out_of_resources_error(CUkernel kernel, const Grid& block) {
    // If we can't read the data we need to try to clarify, this is a no-op.
    CUfunction function;
    if (g_cuKernelGetFunction(&function, kernel) != CUDA_SUCCESS) return false;

    int registers_per_thread;
    if (g_cuFuncGetAttribute(&registers_per_thread, CU_FUNC_ATTRIBUTE_NUM_REGS, function)
            != CUDA_SUCCESS)
        return false;

    CUdevice device;
    if (g_cuCtxGetDevice(&device) != CUDA_SUCCESS) return false;

    int max_registers_per_block;
    if (g_cuDeviceGetAttribute(&max_registers_per_block,
            CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device) != CUDA_SUCCESS)
        return false;

    uint64_t threads_per_block = 1;
    for (unsigned dim : block.dims)
        threads_per_block *= dim;
    uint64_t required_registers = threads_per_block * registers_per_thread;
    if (required_registers <= static_cast<uint64_t>(max_registers_per_block)) return false;

    uint64_t suggested_max_registers = max_registers_per_block / threads_per_block;
    raise(PyExc_RuntimeError,
          "Failed to launch CUDA kernel: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES (%s). "
          "The kernel uses %d registers per thread and the launch block has %llu threads, "
          "requiring %llu registers per block, but the device limit is %d. Recompile the "
          "kernel with cuda.jit(max_registers=%llu) to force the compiler to use fewer "
          "registers, or reduce the threads per block.",
          get_cuda_error(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES), registers_per_thread,
          static_cast<unsigned long long>(threads_per_block),
          static_cast<unsigned long long>(required_registers), max_registers_per_block,
          static_cast<unsigned long long>(suggested_max_registers));
    return true;
}

Status launch(KernelDispatcher& dispatcher, Grid grid, Grid block, std::optional<Grid> cluster,
              std::optional<CUstream> stream, int sharedmem,
              PyObject* const* pyargs, Py_ssize_t num_pyargs) {
    LaunchHelperPtr helper = launch_helper_get();
    get_pyarg_types(pyargs, num_pyargs, helper->pyarg_types);

    CUstream launch_stream;
    if (stream.has_value()) {
        launch_stream = *stream;
    } else {
        Result<CUstream> current_stream = get_current_stream(helper->pyarg_types);
        if (!current_stream.is_ok()) return ErrorRaised;
        launch_stream = *current_stream;
    }

    PythonArgProfile* profile = dispatcher.arg_profiles.find(helper->pyarg_types);
    if (!profile) {
        // Slower path: classify args and find/create kernel family

        std::vector<PythonArgKind> arg_kinds;
        arg_kinds.reserve(num_pyargs);
        std::vector<ParameterKind> param_kinds;
        param_kinds.reserve(num_pyargs);
        for (Py_ssize_t i = 0; i < num_pyargs; ++i) {
            Result<std::pair<PythonArgKind, ParameterKind>> c = classify_arg(pyargs[i]);
            if (!c.is_ok()) return ErrorRaised;
            arg_kinds.push_back(c->first);
            param_kinds.push_back(c->second);
        }

        // Find or create kernel family for this parameter pattern
        KernelDispatcher::FamilyMap::iterator family_iter
                = dispatcher.kernel_families.find(param_kinds);
        if (family_iter == dispatcher.kernel_families.end()) {
            RefPtr<KernelFamily> new_family = steal(new KernelFamily());
            family_iter = dispatcher.kernel_families.emplace(
                    std::make_pair(std::move(param_kinds), std::move(new_family))).first;
        }

        profile = dispatcher.arg_profiles.insert(
                    std::move(helper->pyarg_types),
                    PythonArgProfile{family_iter->second, std::move(arg_kinds)});
    }

    // Try fast scalar extraction path
    profile->maybe_init_fast_path(dispatcher.constant_arg_flags);

    KernelFamily::KernelMap::iterator kernel_iter;
    ContextGuard ctx_guard;

    if (!profile->fast_extractors.empty()) {
        // Fast path: all args are simple scalars with no constants.
        // Single loop: extract values + build pointers simultaneously.
        // Clean up any leftover state from previous (non-scalar) launches
        // since LaunchHelper is reused from a freelist.
        for (void* ptr : helper->aligned_tma_descriptors) {
            if (ptr) free_aligned_storage(ptr);
        }
        helper->aligned_tma_descriptors.clear();
        for (const auto& info : helper->record_copies) {
            if (info.device_ptr) g_cuMemFree(info.device_ptr);
        }
        helper->record_copies.clear();

        size_t n = static_cast<size_t>(num_pyargs);
        helper->cuargs.resize(n);
        helper->cuarg_pointers.resize(n);
        CudaArg* base = helper->cuargs.data();
        const ScalarExtractor* extractors = profile->fast_extractors.data();
        for (size_t i = 0; i < n; ++i) {
            extractors[i](pyargs[i], &base[i]);
            helper->cuarg_pointers[i] = &base[i];
        }
        helper->constants.clear();
        helper->cuda_context = nullptr;

        if (!ensure_numba_context(dispatcher.ensure_context_func.get()))
            return ErrorRaised;

        KernelFamily::KernelMap& kernel_map = profile->family->kernels_by_constants;
        kernel_iter = kernel_map.find(helper->constants);
        if (kernel_iter == kernel_map.end()) {
            // Slowest path: need to compile a new kernel
            Result<CudaKernelHandle> kernel = compile(dispatcher.compile_func.get(), pyargs, num_pyargs);
            if (!kernel.is_ok()) return ErrorRaised;
            // Defer the post-load callback until after emplace so that racing
            // threads that compile the same kernel don't fire duplicate callbacks.
            PyPtr post_load_cb = std::move(kernel->post_load_callback);
            auto [it, inserted] = kernel_map.emplace(helper->constants, std::move(*kernel));
            kernel_iter = it;
            if (inserted && post_load_cb) {
                PyPtr py_handle = steal(PyLong_FromVoidPtr(
                        static_cast<void*>(kernel_iter->second.cukernel.lib.get())));
                if (!py_handle) return ErrorRaised;
                PyPtr cb_result = steal(
                        PyObject_CallOneArg(post_load_cb.get(), py_handle.get()));
                if (!cb_result) return ErrorRaised;
            }
        }

        if (!maybe_switch_context(kernel_iter->second.cukernel.lib.context(), ctx_guard))
            return ErrorRaised;
    } else {
        // Standard path: full extraction with per-type switch
        if (!extract_cuda_args(pyargs, num_pyargs, profile->arg_kinds,
                               dispatcher.constant_arg_flags, *helper)) {
            return ErrorRaised;
        }

        if (helper->cuda_context) {
            if (!maybe_switch_context(helper->cuda_context, ctx_guard))
                return ErrorRaised;
        } else {
            if (!ensure_numba_context(dispatcher.ensure_context_func.get()))
                return ErrorRaised;
        }

        KernelFamily::KernelMap& kernel_map = profile->family->kernels_by_constants;
        kernel_iter = kernel_map.find(helper->constants);
        if (kernel_iter == kernel_map.end()) {
            // Slowest path: need to compile a new kernel
            Result<CudaKernelHandle> kernel = compile(dispatcher.compile_func.get(), pyargs, num_pyargs);
            if (!kernel.is_ok()) return ErrorRaised;

            // Defer the post-load callback until after emplace so that racing
            // threads that compile the same kernel don't fire duplicate callbacks.
            PyPtr post_load_cb = std::move(kernel->post_load_callback);
            auto [it, inserted] = kernel_map.emplace(helper->constants, std::move(*kernel));
            kernel_iter = it;
            if (inserted && post_load_cb) {
                PyPtr py_handle = steal(PyLong_FromVoidPtr(
                        static_cast<void*>(kernel_iter->second.cukernel.lib.get())));
                if (!py_handle) return ErrorRaised;
                PyPtr cb_result = steal(
                        PyObject_CallOneArg(post_load_cb.get(), py_handle.get()));
                if (!cb_result) return ErrorRaised;
            }
        }

        if (!helper->cuda_context
                && !maybe_switch_context(kernel_iter->second.cukernel.lib.context(), ctx_guard))
            return ErrorRaised;

        helper->cuarg_pointers.clear();

        // Build kernel arguments in MLIR memref calling convention order.
        // For each array: (ptr, ptr, offset, shapes..., strides...)
        // For scalars: just the value
        // For TMA descriptors: the descriptor pointer
        // All values are already stored in the flat cuargs vector, we just build pointers to them.

        for (const ArgMetadata& meta : helper->arg_metadata) {
            if (meta.is_array()) {
                // Array: cuargs contains [ptr, ptr, offset, shapes..., strides...]
                // All already in correct order, just create pointers
                size_t num_values = 3 + 2 * meta.ndim;  // ptr, ptr, offset, ndim shapes, ndim strides
                for (size_t i = 0; i < num_values; ++i) {
                    helper->cuarg_pointers.push_back(&helper->cuargs[meta.start_idx + i]);
                }
            } else if (meta.is_tma_descriptor()) {
                // TMA descriptor: cuLaunchKernel expects kernel_params[i] to point to the descriptor data.
                // We pass the 128-byte aligned storage we created in extract_tma_descriptor.
                helper->cuarg_pointers.push_back(helper->cuargs[meta.start_idx].device_ptr);
            } else {
                // Scalar: always one pointer
                // ndim=0: regular scalar (1 entry)
                // ndim=1: complex64 (1 entry with packed f32 pair)
                // ndim=2: complex128 (2 consecutive f64 entries, 1 pointer to first)
                helper->cuarg_pointers.push_back(&helper->cuargs[meta.start_idx]);
            }
        }
    }

    // Configure dynamic shared memory if needed.
    //
    // Every function has a default cap on dynamic shared memory
    // (CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES). This default is the
    // device's standard per-block limit (typically 48 KiB) minus any static
    // shared memory the kernel uses, so it can be anything up to 48 KiB. When
    // the requested dynamic shared memory exceeds this default, we must
    // explicitly opt in to the larger limit, otherwise the driver rejects the
    // launch with CUDA_ERROR_INVALID_VALUE.
    //
    // We compare against the function's actual default (queried at runtime)
    // rather than a hard-coded 48 KiB so that the window [default, 48 KiB] is
    // reachable. A kernel asking for exactly 48 KiB while using even a single
    // byte of static shared memory has a default below 48 KiB and therefore
    // needs the opt-in.
    {
        // Convert CUkernel to CUfunction for attribute configuration
        CUfunction cu_function = nullptr;
        CUkernel cu_kernel = kernel_iter->second.cukernel.kernel;
        CUresult func_res = g_cuKernelGetFunction(&cu_function, cu_kernel);
        if (func_res != CUDA_SUCCESS) {
            return raise(PyExc_RuntimeError,
                        "Failed to get CUfunction from CUkernel: %s",
                        get_cuda_error(func_res));
        }

        // The function's default dynamic limit is the device's standard
        // per-block limit minus the kernel's static shared memory
        // (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK -
        //  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES).
        int default_dyn_limit = 0;
        CUresult get_res = g_cuFuncGetAttribute(
            &default_dyn_limit,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            cu_function
        );
        if (get_res != CUDA_SUCCESS) {
            return raise(PyExc_RuntimeError,
                        "Failed to query max dynamic shared memory size: %s",
                        get_cuda_error(get_res));
        }

        if (sharedmem > default_dyn_limit) {
            // Surface a clear error if the request exceeds what the device can
            // provide instead of letting the attribute set fail with an opaque
            // INVALID_VALUE.
            //
            // The device opt-in limit (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_
            // PER_BLOCK_OPTIN) covers the *total* per-block shared memory, i.e.
            // static + dynamic. The kernel's static shared memory therefore
            // reduces the dynamic budget, so we subtract it before comparing.
            // Otherwise we could opt in to a dynamic size that, combined with
            // static usage, overflows the device limit and fails at launch.
            int max_optin = 0;
            CUdevice dev;
            if (g_cuCtxGetDevice(&dev) == CUDA_SUCCESS) {
                g_cuDeviceGetAttribute(
                    &max_optin,
                    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                    dev
                );
            }

            int static_smem = 0;
            CUresult static_res = g_cuFuncGetAttribute(
                &static_smem,
                CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                cu_function
            );
            if (static_res != CUDA_SUCCESS) {
                return raise(PyExc_RuntimeError,
                            "Failed to query static shared memory size: %s",
                            get_cuda_error(static_res));
            }

            if (max_optin > 0 && sharedmem > max_optin - static_smem) {
                return raise(PyExc_ValueError,
                            "Requested dynamic shared memory (%d bytes) plus the kernel's "
                            "static shared memory (%d bytes) exceeds the device maximum "
                            "opt-in shared memory per block (%d bytes)",
                            sharedmem, static_smem, max_optin);
            }

            CUresult attr_res = g_cuFuncSetAttribute(
                cu_function,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                sharedmem
            );

            if (attr_res != CUDA_SUCCESS) {
                return raise(PyExc_RuntimeError,
                            "Failed to set max dynamic shared memory size to %d bytes: %s",
                            sharedmem, get_cuda_error(attr_res));
            }
        }
    }

    CUresult res;
    CUfunction cu_function = reinterpret_cast<CUfunction>(kernel_iter->second.cukernel.kernel);
    bool cooperative = kernel_iter->second.cooperative;

    if (cluster.has_value() || cooperative) {
        // Use cuLaunchKernelEx for cluster and/or cooperative launch
        CUlaunchAttribute launch_attrs[3];
        int num_attrs = 0;

        if (cluster.has_value()) {
            launch_attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            launch_attrs[num_attrs].value.clusterDim.x = cluster->dims[0];
            launch_attrs[num_attrs].value.clusterDim.y = cluster->dims[1];
            launch_attrs[num_attrs].value.clusterDim.z = cluster->dims[2];
            num_attrs++;

            launch_attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
            launch_attrs[num_attrs].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
            num_attrs++;
        }

        if (cooperative) {
            launch_attrs[num_attrs].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
            launch_attrs[num_attrs].value.cooperative = 1;
            num_attrs++;
        }

        CUlaunchConfig launch_config = {};
        launch_config.gridDimX = grid.dims[0];
        launch_config.gridDimY = grid.dims[1];
        launch_config.gridDimZ = grid.dims[2];
        launch_config.blockDimX = block.dims[0];
        launch_config.blockDimY = block.dims[1];
        launch_config.blockDimZ = block.dims[2];
        launch_config.sharedMemBytes = static_cast<unsigned int>(sharedmem);
        launch_config.hStream = launch_stream;
        launch_config.attrs = launch_attrs;
        launch_config.numAttrs = num_attrs;

        res = g_cuLaunchKernelEx(&launch_config, cu_function,
                                  helper->cuarg_pointers.data(), nullptr);
    } else {
        // Standard launch without cluster or cooperative
        res = g_cuLaunchKernel(
                cu_function,
                grid.dims[0], grid.dims[1], grid.dims[2],
                block.dims[0], block.dims[1], block.dims[2],
                sharedmem,
                launch_stream,
                helper->cuarg_pointers.data(),
                nullptr);
    }

    if (res != CUDA_SUCCESS) {
        if (res == CUDA_ERROR_INVALID_VALUE && try_clarify_invalid_value_error(grid))
            return ErrorRaised;
        if (res == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
                && try_clarify_launch_out_of_resources_error(
                    kernel_iter->second.cukernel.kernel, block))
            return ErrorRaised;

        const char* error_name = nullptr;
        g_cuGetErrorName(res, &error_name);
        if (!error_name) error_name = "UNKNOWN";
        return raise(PyExc_RuntimeError, "Failed to launch CUDA kernel: %s (%s)",
                     error_name, get_cuda_error(res));
    }

    // Check for kernel error codes (set by device-side assertion
    // replacements). Skipped when the compiler proved the kernel never
    // sets the error code, keeping the launch asynchronous.
    if (kernel_iter->second.check_error_code
            && !check_kernel_error_code(kernel_iter->second.cukernel.lib))
        return ErrorRaised;

    // Copy scalar records back from device to host
    for (const auto& info : helper->record_copies) {
        CUresult copy_res = g_cuMemcpyDtoH(info.host_ptr, info.device_ptr, info.size);
        if (copy_res != CUDA_SUCCESS) {
            return raise(PyExc_RuntimeError, "Failed to copy record back from device: %s",
                         get_cuda_error(copy_res));
        }
    }

    return OK;
}

Result<std::vector<bool>> parse_constant_arg_flags(PyObject* tuple) {
    if (!PyTuple_Check(tuple))
        return raise(PyExc_TypeError, "constant_arg_flags must be a tuple");

    std::vector<bool> constant_arg_flags;
    Py_ssize_t tuple_size = PyTuple_GET_SIZE(tuple);
    constant_arg_flags.reserve(tuple_size);
    for (Py_ssize_t i = 0; i < tuple_size; ++i) {
        PyObject* item = PyTuple_GET_ITEM(tuple, i);
        if (!PyBool_Check(item))
            return raise(PyExc_TypeError, "constant_arg_flags must be a tuple of booleans");

        int is_constant = PyObject_IsTrue(item);
        if (is_constant < 0) return ErrorRaised;
        constant_arg_flags.push_back(static_cast<bool>(is_constant));
    }
    return constant_arg_flags;
}

int KernelDispatcher_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char* keywords[] = {"", "", "", nullptr};
    PyObject* compile_func = nullptr;
    PyObject* py_constant_arg_flags = nullptr;
    PyObject* ensure_context_func = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", const_cast<char**>(keywords),
                                     &compile_func, &py_constant_arg_flags,
                                     &ensure_context_func))
        return -1;

    Result<std::vector<bool>> constant_arg_flags = parse_constant_arg_flags(py_constant_arg_flags);
    if (!constant_arg_flags.is_ok()) return -1;

    KernelDispatcher& dispatcher = py_unwrap<KernelDispatcher>(self);
    dispatcher.compile_func = newref(compile_func);
    dispatcher.ensure_context_func = newref(ensure_context_func);
    dispatcher.constant_arg_flags = std::move(*constant_arg_flags);
    return 0;
}


PyTypeObject KernelDispatcher_type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};

Result<Grid> parse_grid(PyObject* tuple) {
    if (!PyTuple_Check(tuple))
        return raise(PyExc_TypeError, "Grid must be a tuple");

    Py_ssize_t tuple_size = PyTuple_GET_SIZE(tuple);
    if (tuple_size > Grid::Len)
        return raise(PyExc_ValueError, "Grid dimensions must be at most %d, got length %zd",
                     Grid::Len, tuple_size);

    Grid grid;
    for (int i = 0; i < Grid::Len; ++i) {
        // Pad with 1s on the right if tuple size < Grid::Len
        unsigned long val = 1;
        if (i < tuple_size) {
            val = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(tuple, i));
            if (PyErr_Occurred()) return ErrorRaised;
            if (val > UINT_MAX)
                return raise(PyExc_ValueError, "Grid[%d] value too big: got=%lu",
                             i, val);
        }
        grid.dims[i] = val;
    }

    return grid;
}

struct LaunchConfiguration {
    vectorcallfunc vectorcall;
    PyPtr dispatcher;
    Grid grid;
    Grid block;
    std::optional<Grid> cluster;
    std::optional<CUstream> stream;
    int sharedmem;
};

PyObject* LaunchConfiguration_vectorcall(PyObject* self, PyObject *const *args,
                                             size_t nargsf, PyObject* kwnames) {
    if (kwnames) {
        PyErr_SetString(PyExc_TypeError, "Keyword arguments are not supported");
        return nullptr;
    }

    LaunchConfiguration& config = py_unwrap<LaunchConfiguration>(self);

    Py_ssize_t num_args = PyVectorcall_NARGS(nargsf);

    KernelDispatcher& dispatcher = py_unwrap<KernelDispatcher>(config.dispatcher.get());
    if (!launch(dispatcher, config.grid, config.block, config.cluster, config.stream, config.sharedmem, args, num_args))
        return nullptr;

    return Py_NewRef(Py_None);
}

PyObject* LaunchConfiguration_call(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (kwargs) {
        PyErr_SetString(PyExc_TypeError, "Keyword arguments are not supported");
        return nullptr;
    }

    LaunchConfiguration& config = py_unwrap<LaunchConfiguration>(self);

    KernelDispatcher& dispatcher = py_unwrap<KernelDispatcher>(config.dispatcher.get());

    PyObject** pyargs = &_PyTuple_CAST(args)->ob_item[0];
    Py_ssize_t num_pyargs = PyTuple_GET_SIZE(args);

    if (!launch(dispatcher, config.grid, config.block, config.cluster, config.stream, config.sharedmem, pyargs, num_pyargs))
        return nullptr;

    return Py_NewRef(Py_None);
}

int LaunchConfiguration_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    const char* keywords[] = {"dispatcher", "grid", "block", "stream", "sharedmem", "cluster", nullptr};
    PyObject* dispatcher = nullptr;
    PyObject* py_grid = nullptr;
    PyObject* py_block = nullptr;
    PyObject* py_stream = Py_None;
    PyObject* py_sharedmem = Py_None;
    PyObject* py_cluster = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!OO|OOO", const_cast<char**>(keywords),
                                     &KernelDispatcher_type, &dispatcher,
                                     &py_grid, &py_block, &py_stream, &py_sharedmem, &py_cluster))
        return -1;

    Result<Grid> grid = parse_grid(py_grid);
    if (!grid.is_ok()) return -1;

    Result<Grid> block = parse_grid(py_block);
    if (!block.is_ok()) return -1;

    std::optional<Grid> cluster = std::nullopt;
    if (py_cluster != Py_None) {
        Result<Grid> cluster_parsed = parse_grid(py_cluster);
        if (!cluster_parsed.is_ok()) return -1;
        cluster = *cluster_parsed;
    }

    std::optional<CUstream> stream = std::nullopt;
    if (py_stream != Py_None) {
        Result<CUstream> stream_parsed = parse_stream(py_stream);
        if (!stream_parsed.is_ok()) return -1;
        stream = *stream_parsed;
    }

    int sharedmem = 0;
    if (py_sharedmem != Py_None) {
        if (!PyLong_Check(py_sharedmem)) {
            PyErr_SetString(PyExc_TypeError, "sharedmem must be an integer");
            return -1;
        }
        sharedmem = pylong_as<int>(py_sharedmem);
        if (sharedmem < 0) {
            PyErr_Format(PyExc_ValueError, "sharedmem must be non-negative, got %d", sharedmem);
            return -1;
        }
    }

    LaunchConfiguration& config = py_unwrap<LaunchConfiguration>(self);
    config.vectorcall = LaunchConfiguration_vectorcall,
    config.dispatcher = newref(dispatcher);
    config.grid = *grid;
    config.block = *block;
    config.cluster = cluster;
    config.stream = stream;
    config.sharedmem = sharedmem;
    return 0;
}

PyTypeObject LaunchConfiguration_type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};


void try_get_torch_globals() {
    PyPtr torch = try_import("torch");
    if (!torch) return;

    // Save a reference to torch.Tensor
    if (PyPtr torch_Tensor = try_getattr(torch, "Tensor")) {
        if (PyType_Check(torch_Tensor.get()))
            g_torch_Tensor_type = reinterpret_cast<PyTypeObject*>(torch_Tensor.release());
    }

    // Save references to torch.cuda.current_stream, torch.cuda.Stream
    if (PyPtr torch_cuda = try_getattr(torch, "cuda")) {
        if (PyPtr torch_cuda_Stream = try_getattr(torch_cuda, "Stream")) {
            if (PyType_Check(torch_cuda_Stream.get())) {
                g_torch_cuda_Stream_type = reinterpret_cast<PyTypeObject*>(
                        torch_cuda_Stream.release());
            }
        }
    }

    // Save references to torch._C._to_dlpack, torch._C._cuda_getCurrentRawStream
    if (PyPtr torch_C = try_getattr(torch, "_C")) {
        g_torch_to_dlpack_func = try_getattr(torch_C, "_to_dlpack").release();
        g_torch_cuda_getCurrentRawStream = try_getattr(
                torch_C, "_cuda_getCurrentRawStream").release();
    }
}

void try_get_cupy_globals() {
    PyPtr cupy = try_import("cupy");
    if (!cupy) return;

    // Save a reference to cupy.ndarray
    if (PyPtr cupy_ndarray = try_getattr(cupy, "ndarray")) {
        if (PyType_Check(cupy_ndarray.get()))
            g_cupy_ndarray_type = reinterpret_cast<PyTypeObject*>(cupy_ndarray.release());
    }

    // Save references to cupy.cuda.get_current_stream, cupy.cuda.Stream
    if (PyPtr cupy_cuda = try_getattr(cupy, "cuda")) {
        g_cupy_cuda_get_current_stream = try_getattr(cupy_cuda, "get_current_stream").release();
        if (PyPtr cupy_cuda_Stream = try_getattr(cupy_cuda, "Stream")) {
            if (PyType_Check(cupy_cuda_Stream.get())) {
                g_cupy_cuda_Stream_type = reinterpret_cast<PyTypeObject*>(
                        cupy_cuda_Stream.release());
            }
        }
    }
}

void try_get_numba_globals() {
    PyPtr numba_cuda = try_import("numba_cuda_mlir.numba_cuda");
    if (!numba_cuda) return;

    // Save a reference to numba.cuda.driver.Stream
    if (PyPtr numba_cuda_driver = try_getattr(numba_cuda, "driver")) {
        if (PyPtr numba_cuda_Stream = try_getattr(numba_cuda_driver, "Stream")) {
            if (PyType_Check(numba_cuda_Stream.get())) {
                g_numba_cuda_Stream_type = reinterpret_cast<PyTypeObject*>(
                        numba_cuda_Stream.release());
            }
        }
    }
}

void try_get_cuda_core_globals() {
    PyPtr cuda_core = try_import("cuda.core");
    if (!cuda_core) return;

    // Save a reference to cuda.core.Stream
    if (PyPtr cuda_core_Stream = try_getattr(cuda_core, "Stream")) {
        if (PyType_Check(cuda_core_Stream.get())) {
            g_cuda_core_Stream_type = reinterpret_cast<PyTypeObject*>(
                    cuda_core_Stream.release());
        }
    }
}

void try_get_enum_globals() {
    PyPtr enum_module = try_import("enum");
    if (!enum_module) return;

    if (PyPtr enum_Enum = try_getattr(enum_module, "Enum")) {
        if (PyType_Check(enum_Enum.get()))
            g_enum_Enum_type = reinterpret_cast<PyTypeObject*>(enum_Enum.release());
    }
}

} // anonymous namespace


namespace std {
template<>
struct hash<ConstantArg> {
    size_t operator()(const ConstantArg& arg) const {
        switch (arg.type) {
        case ConstantArgType::INT64:
            return std::hash<int64_t>{}(arg.value.i64);
        case ConstantArgType::FLOAT64:
            // hash float bits directly
            uint64_t float_bits;
            std::memcpy(&float_bits, &arg.value.f64, sizeof(arg.value.f64));
            return std::hash<uint64_t>{}(float_bits);
        case ConstantArgType::STRING:
            return std::hash<std::string>{}(arg.str);
        }
        // unreachable code
        assert(false && "Unsupported constant arg type");
        return 0;
    }
};
} // std namespace


#define INIT_STRING_CONSTANT(ident) \
    if (!(g_##ident##_pyunicode = PyUnicode_InternFromString(#ident))) return ErrorRaised;

Status kernel_init(PyObject* m) {
    INIT_STRING_CONSTANT(__cuda_array_interface__);
    INIT_STRING_CONSTANT(__array_interface__);
    INIT_STRING_CONSTANT(typestr);
    INIT_STRING_CONSTANT(shape);
    INIT_STRING_CONSTANT(data);
    INIT_STRING_CONSTANT(strides);
    INIT_STRING_CONSTANT(__dlpack__);
    INIT_STRING_CONSTANT(int64);

    try_get_torch_globals();
    try_get_cupy_globals();
    try_get_numba_globals();
    try_get_cuda_core_globals();
    try_get_enum_globals();

    KernelDispatcher_type.tp_name = "numba_cuda_mlir._cext.KernelDispatcher";
    KernelDispatcher_type.tp_basicsize = sizeof(PythonWrapper<KernelDispatcher>);
    KernelDispatcher_type.tp_dealloc = pywrapper_dealloc<KernelDispatcher>;
    KernelDispatcher_type.tp_flags = Py_TPFLAGS_DEFAULT;
    KernelDispatcher_type.tp_init = KernelDispatcher_init;
    KernelDispatcher_type.tp_new = pywrapper_new<KernelDispatcher>;

    LaunchConfiguration_type.tp_name = "numba_cuda_mlir._cext.LaunchConfiguration";
    LaunchConfiguration_type.tp_basicsize = sizeof(PythonWrapper<LaunchConfiguration>);
    LaunchConfiguration_type.tp_dealloc = pywrapper_dealloc<LaunchConfiguration>;
    LaunchConfiguration_type.tp_vectorcall_offset =
        static_cast<Py_ssize_t>(offsetof(PythonWrapper<LaunchConfiguration>, object)
                                + offsetof(LaunchConfiguration, vectorcall));
    LaunchConfiguration_type.tp_call = LaunchConfiguration_call;
    LaunchConfiguration_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL;
    LaunchConfiguration_type.tp_init = LaunchConfiguration_init;
    LaunchConfiguration_type.tp_new = pywrapper_new<LaunchConfiguration>;

    if (PyType_Ready(&KernelDispatcher_type) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, "KernelDispatcher",
                reinterpret_cast<PyObject*>(&KernelDispatcher_type)) < 0)
        return ErrorRaised;

    if (PyType_Ready(&LaunchConfiguration_type) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, "LaunchConfiguration",
                reinterpret_cast<PyObject*>(&LaunchConfiguration_type)) < 0)
        return ErrorRaised;

    return OK;
}
