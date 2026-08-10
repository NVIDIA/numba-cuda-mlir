// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Poisoning shim for tests/test_symbol_isolation.py. Compiled in the
// build stage and shipped as a CI artifact; the test stage LD_PRELOADs
// it and expects that RTLD_DEEPBIND on the bundled libMLIRPythonCAPI.so
// keeps Support's PLT binding to the bundled copy. If isolation holds,
// this stub is never invoked; if it fires, the process aborts loudly.
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    void *ptr;
} MlirContext;

MlirContext mlirContextCreateWithThreading(int threading_enabled) {
    (void)threading_enabled;
    fputs("FAKE mlirContextCreateWithThreading called -- isolation broken\n",
          stderr);
    abort();
}
