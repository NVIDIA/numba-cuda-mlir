/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "check.h"

#include <cstdio>
#include <cstdlib>

void check_failed(const char* file, int line, const char* cond) {
    fprintf(stderr, "CHECK FAILED: %s:%d: %s\n", file, line, cond);
    abort();
}
