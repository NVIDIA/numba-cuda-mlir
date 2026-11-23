/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
// _mlir_libs/__init__.py imports _site_initialize_{0,1,...} when a package has
// no _mlirRegisterEverything, and calls register_dialects() for every Context.

#include "Registration.h"

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

NB_MODULE(_site_initialize_0, m) {
  m.doc() = "Registers the MLIR dialects, passes and LLVM IR translations "
            "numba-cuda-mlir uses";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    numbaCudaMlirRegisterDialects(registry);
  });

  numbaCudaMlirRegisterPasses();
}
