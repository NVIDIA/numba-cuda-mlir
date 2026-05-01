/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "py.h"
#include "cuda_shim.h"

Status cuda_helper_init(PyObject* m);

const char* get_cuda_error(CUresult res);
