// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

/*
 * Expose all functions as pointers in a dedicated C extension.
 */

#include "Python.h"

/* Visibility control macros */
#if defined(_WIN32) || defined(_WIN64)
    #define VISIBILITY_HIDDEN
    #define VISIBILITY_GLOBAL __declspec(dllexport)
#else
    #define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
    #define VISIBILITY_GLOBAL __attribute__((visibility("default")))
#endif

/* Define all runtime-required symbols in this C module, but do not
   export them outside the shared library if possible. */
#define NUMBA_EXPORT_FUNC(_rettype) VISIBILITY_HIDDEN _rettype
#define NUMBA_EXPORT_DATA(_vartype) VISIBILITY_HIDDEN _vartype

extern "C" {

/* Numba CUDA C++ helpers */
#include "_helperlib.cpp"

static PyObject *
build_c_helpers_dict(void)
{
    PyObject *dct = PyDict_New();
    if (dct == NULL)
        goto error;

#define _declpointer(name, value) do {                 \
    PyObject *o = PyLong_FromVoidPtr((void*)value);    \
    if (o == NULL) goto error;                         \
    if (PyDict_SetItemString(dct, name, o)) {          \
        Py_DECREF(o);                                  \
        goto error;                                    \
    }                                                  \
    Py_DECREF(o);                                      \
} while (0)

#define declmethod(func) _declpointer(#func, &numba_##func)

    /* Unicode string support */
    declmethod(extract_unicode);

#undef declmethod
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

static PyMethodDef ext_methods[] = {
    { NULL },
};

PyMODINIT_FUNC PyInit__helperlib(void) {
    static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT, "_helperlib", "No docs", -1, ext_methods, NULL, NULL, NULL, NULL };
    PyObject *m = PyModule_Create(&moduledef);

    if (m == NULL)
        return NULL;
#if !defined(Py_LIMITED_API) && defined(Py_GIL_DISABLED)
    if (PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED) < 0) {
        Py_DECREF(m);
        return NULL;
    }
#endif

    PyObject *c_helpers = build_c_helpers_dict();
    if (c_helpers == NULL)
        goto error;
    if (PyModule_AddObjectRef(m, "c_helpers", c_helpers)) {
        Py_DECREF(c_helpers);
        goto error;
    }
    Py_DECREF(c_helpers);

    if (PyModule_AddIntConstant(m, "long_min", LONG_MIN))
        goto error;
    if (PyModule_AddIntConstant(m, "long_max", LONG_MAX))
        goto error;
    if (PyModule_AddIntConstant(m, "py_buffer_size", sizeof(Py_buffer)))
        goto error;
    if (PyModule_AddIntConstant(m, "py_gil_state_size", sizeof(PyGILState_STATE)))
        goto error;
    if (PyModule_AddIntConstant(m, "py_unicode_1byte_kind", PyUnicode_1BYTE_KIND))
        goto error;
    if (PyModule_AddIntConstant(m, "py_unicode_2byte_kind", PyUnicode_2BYTE_KIND))
        goto error;
    if (PyModule_AddIntConstant(m, "py_unicode_4byte_kind", PyUnicode_4BYTE_KIND))
        goto error;
#if (PY_MAJOR_VERSION == 3)
#if ((PY_MINOR_VERSION == 10) || (PY_MINOR_VERSION == 11))
    if (PyModule_AddIntConstant(m, "py_unicode_wchar_kind", PyUnicode_WCHAR_KIND))
        goto error;
#endif
#endif

    return m;

error:
    Py_DECREF(m);
    return NULL;
}

} // extern "C"
