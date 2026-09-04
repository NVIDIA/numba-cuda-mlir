/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-2-Clause
 */

// Globally needed variables
struct NRT_MemSys {
    struct {
      bool enabled;
      unsigned long long alloc;
      unsigned long long free;
      unsigned long long mi_alloc;
      unsigned long long mi_free;
    } stats;
  };

/* The Memory System object */
__device__ NRT_MemSys* TheMSys;

extern "C" __global__ void NRT_MemSys_set(NRT_MemSys *memsys_ptr);
