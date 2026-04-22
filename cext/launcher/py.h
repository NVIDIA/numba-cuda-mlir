/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ref_ptr.h"
#include <Python.h>
#include <optional>

#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 13
#define PyLong_AsInt PyLong_AsInt
#else
#define PyLong_AsInt _PyLong_AsInt
#endif

using PyPtr = RefPtr<PyObject>;

static inline void reference_add(PyObject& obj) {
    Py_INCREF(&obj);
}

static inline void reference_remove(PyObject& obj) {
    Py_DECREF(&obj);
}

template <typename T>
T pylong_as(PyObject* obj) {
    if constexpr (std::is_same_v<T, int>) {
        return PyLong_AsInt(obj);
    } else if constexpr (std::is_same_v<T, long> || std::is_same_v<T, long long>) {
        auto val = std::is_same_v<T, long> ? PyLong_AsLong(obj) : PyLong_AsLongLong(obj);
        if (!PyErr_Occurred())
            return val;
        // Value exceeds signed range (e.g. uint64 max) -- reinterpret as bits
        PyErr_Clear();
        unsigned long long uval = PyLong_AsUnsignedLongLong(obj);
        if (PyErr_Occurred())
            return -1;
        return static_cast<T>(uval);
    } else {
        static_assert(!sizeof(T*), "pylong_as<T> not implemented for given T");
    }
}

template <typename T>
struct PythonWrapper {
    PyObject_HEAD
    T object;
};

template <typename T>
T& py_unwrap(PyObject* pyobj) {
    PythonWrapper<T>* wrapper = reinterpret_cast<PythonWrapper<T>*>(pyobj);
    return wrapper->object;
}

template <typename T>
PyObject* pywrapper_new(PyTypeObject* type, PyObject*, PyObject*) {
    PyObject* ret = type->tp_alloc(type, 0);
    if (!ret) return nullptr;

    T& obj = py_unwrap<T>(ret);
    new (&obj) T();
    return ret;
}

template <typename T>
void pywrapper_dealloc(PyObject* self) {
    PythonWrapper<T>* wrapper = reinterpret_cast<PythonWrapper<T>*>(self);
    wrapper->object.~T();
    Py_TYPE(self)->tp_free(self);
}

struct OK_t{};
struct ErrorRaised_t{};

class [[nodiscard]] Status {
public:
    Status(OK_t) : ok_(true) {}
    Status(ErrorRaised_t) : ok_(false) {}

    explicit operator bool() const {
        return ok_;
    }

private:
    bool ok_;
};

static constexpr OK_t OK = {};
static constexpr ErrorRaised_t ErrorRaised = {};


template <typename T>
class [[nodiscard]] Result {
public:
    Result(ErrorRaised_t) : opt_(std::nullopt) {}

    Result(T&& val) : opt_(std::move(val)) {}

    Result(const Result& other) : opt_(other.opt_) {}

    Result(Result&& other) : opt_(std::move(other.opt_)) {}

    Result& operator= (const Result& other) {
        opt_ = other.opt_;
        return *this;
    }

    Result& operator= (Result&& other) {
        opt_ = std::move(other.opt_);
        return *this;
    }

    bool is_ok() const {
        return opt_.has_value();
    }

    T& operator* () {
        return *opt_;
    }

    const T& operator* () const {
        return *opt_;
    }

    T* operator-> () {
        return &*opt_;
    }

    const T* operator-> () const {
        return &*opt_;
    }

private:
    std::optional<T> opt_;
};

template <typename... Args>
ErrorRaised_t raise(PyObject* exctype, const char* fmt, Args&&... args) {
    PyErr_Format(exctype, fmt, std::forward<Args>(args)...);
    return ErrorRaised;
}

struct SavedException {
    PyPtr type, value, traceback;

    void normalize() {
        PyObject* tmp_type = type.release();
        PyObject* tmp_value = value.release();
        PyObject* tmp_traceback = traceback.release();
        PyErr_NormalizeException(&tmp_type, &tmp_value, &tmp_traceback);
        type = steal(tmp_type);
        value = steal(tmp_value);
        traceback = steal(tmp_traceback);
    }
};

static inline SavedException save_raised_exception() {
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    return SavedException{steal(type), steal(value), steal(traceback)};
}

void log_python_error(const char* filename, int line, const char* level, SavedException& exc,
                      const char* fmt, ...)
    __attribute__(( format(printf, 5, 6) ));

#define LOG_PYTHON_ERROR(level, exc, ...) \
        log_python_error(__FILE__, __LINE__, level, exc, __VA_ARGS__)

static inline PyPtr getattr(PyObject* obj, const char* attrname) {
    return steal(PyObject_GetAttrString(obj, attrname));
}

static inline PyPtr getattr(const PyPtr& obj, const char* attrname) {
    return getattr(obj.get(), attrname);
}

struct ErrorGuard {
    SavedException exc;

    ErrorGuard() {
        exc = save_raised_exception();
    }

    ErrorGuard(const ErrorGuard&) = delete;
    void operator=(const ErrorGuard&) = delete;

    ~ErrorGuard() {
        PyObject* tmp_type = exc.type.release();
        PyObject* tmp_value = exc.value.release();
        PyObject* tmp_traceback = exc.traceback.release();
        PyErr_Restore(tmp_type, tmp_value, tmp_traceback);
    }
};

static inline PyPtr try_getattr(PyObject* obj, const char* attrname,
                                SavedException* exc = nullptr) {
    ErrorGuard guard;
    PyPtr ret = getattr(obj, attrname);
    if (!ret && exc) *exc = save_raised_exception();
    return ret;
}

static inline PyPtr try_getattr(const PyPtr& obj, const char* attrname,
                                SavedException* exc = nullptr) {
    return try_getattr(obj.get(), attrname, exc);
}

static inline PyPtr try_import(const char* modname, SavedException* exc = nullptr) {
    ErrorGuard guard;
    PyPtr ret = steal(PyImport_ImportModule(modname));
    if (!ret && exc) *exc = save_raised_exception();
    return ret;
}
