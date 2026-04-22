/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Like assert() but can't be disabled
#define CHECK(cond) do { \
        if (!(cond)) check_failed(__FILE__, __LINE__, #cond); \
    } while (0)

void check_failed(const char* file, int line, const char* cond);
