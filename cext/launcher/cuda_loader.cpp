/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cuda_loader.h"
#include "cuda_helper.h"

#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

class DynamicLoader {
public:
    DynamicLoader(const char* path) {
#ifdef _WIN32
        handle_ = LoadLibraryA(path);
#else
        handle_ = dlopen(path, RTLD_NOW);
#endif
    }

    DynamicLoader(const DynamicLoader&&) = delete;
    DynamicLoader& operator=(const DynamicLoader&&) = delete;

    template <typename F>
    F load_function(const char* name) {
#ifdef _WIN32
        auto ret = GetProcAddress(reinterpret_cast<HMODULE>(handle_), name);
#else
        void* ret = dlsym(handle_, name);
#endif
        return reinterpret_cast<F>(ret);
    }

    void* get_handle() const {
        return handle_;
    }

    void release() {
        handle_ = nullptr;
    }

    ~DynamicLoader() {
        if (handle_) {
#ifdef _WIN32
            FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
            dlclose(handle_);
#endif
        }
    }

    explicit operator bool () const {
        return handle_;
    }

private:
    void* handle_;
};

typedef CUresult (*cuGetProcAddress_v2_t)
    (const char *symbol, void **funcPtr, int cudaVersion,
     cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);



void* do_get_proc_address(cuGetProcAddress_v2_t getter, void* dlhandle,
                          const char* name, int cuda_version) {
    void* ret = nullptr;
    CUresult res = getter(name, &ret, cuda_version, CU_GET_PROC_ADDRESS_DEFAULT, nullptr);

    if (res != CUDA_SUCCESS) {
        raise(PyExc_RuntimeError,
              "Failed to load '%s' from the CUDA library: cuGetProcAddress_v2 returned %d",
              name, static_cast<int>(res));
        return nullptr;
    }

    if (!ret && dlhandle) {
#ifdef _WIN32
        ret = reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(dlhandle), name));
#else
        // Fallback to dlsym for functions not available via cuGetProcAddress_v2
        ret = dlsym(dlhandle, name);
#endif
    }

    if (!ret) {
        raise(PyExc_RuntimeError,
              "Function '%s' is not available in the CUDA library",
              name);
        return nullptr;
    }

    return ret;
}

template <typename F>
F get_proc_address(cuGetProcAddress_v2_t getter, void* dlhandle,
                   const char* name, int cuda_version) {
    return reinterpret_cast<F>(do_get_proc_address(getter, dlhandle, name, cuda_version));
}

} // anonymous namespace


#define DEFINE_CUDA_FUNCTION_GLOBAL(name, _cuda_version) \
    decltype(&name) g_##name;

FOREACH_CUDA_FUNCTION_TO_LOAD(DEFINE_CUDA_FUNCTION_GLOBAL)

#define GET_PROC_ADDRESS(name, cuda_ver) \
    g_##name = get_proc_address<decltype(&name)>(_cuGetProcAddress, loader.get_handle(), #name, cuda_ver); \
    if (!g_##name) \
        return ErrorRaised;


// A missing driver is deliberately NOT fatal here so cuda may be imported
// on a cpu only machine.
static bool g_cuda_available = false;
static std::string g_cuda_load_error;


static Status load_cuda_driver() {
    // loader is static to keep the handle alive for the lifetime of the process
#ifdef _WIN32
    static DynamicLoader loader("nvcuda.dll");
#else
    static DynamicLoader loader("libcuda.so.1");
#endif
    if (!loader) {
      return raise(PyExc_RuntimeError,
                   "Failed to load the CUDA dynamic library");
    }

    auto _cuGetProcAddress =
        loader.load_function<cuGetProcAddress_v2_t>("cuGetProcAddress_v2");
    if (!_cuGetProcAddress)
        return raise(PyExc_RuntimeError, "CUDA library contains no cuGetProcAddress_v2 symbol");

    FOREACH_CUDA_FUNCTION_TO_LOAD(GET_PROC_ADDRESS)

    return OK;
}

Status cuda_loader_init() {
    if (load_cuda_driver()) {
        g_cuda_available = true;
        return OK;
    }

    // defer driver search to ensure_cuda_available().
    g_cuda_available = false;
    if (PyErr_Occurred()) {
        PyObject *type, *value, *tb;
        PyErr_Fetch(&type, &value, &tb);
        if (value) {
            PyObject* str = PyObject_Str(value);
            if (str) {
                const char* c = PyUnicode_AsUTF8(str);
                if (c)
                    g_cuda_load_error = c;
                Py_DECREF(str);
            }
        }
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(tb);
        PyErr_Clear();
    }
    if (g_cuda_load_error.empty())
        g_cuda_load_error = "the CUDA driver library (libcuda.so.1) could not be loaded";
    return OK;
}

bool cuda_is_available() {
    return g_cuda_available;
}

Status ensure_cuda_available() {
    if (g_cuda_available)
        return OK;
    return raise(PyExc_RuntimeError,
                 "CUDA is not available: %s. A working NVIDIA driver is required "
                 "for GPU operations (importing numba_cuda_mlir succeeds without one).",
                 g_cuda_load_error.c_str());
}
