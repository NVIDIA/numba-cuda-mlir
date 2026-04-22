# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS NVVM70Ops.td)

mlir_tablegen(NVVM70Ops.h.inc -gen-op-decls)
mlir_tablegen(NVVM70Ops.cpp.inc -gen-op-defs)
mlir_tablegen(NVVM70OpsDialect.h.inc -gen-dialect-decls -dialect=nvvm70)
mlir_tablegen(NVVM70OpsDialect.cpp.inc -gen-dialect-defs -dialect=nvvm70)
mlir_tablegen(NVVM70OpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=nvvm70)
mlir_tablegen(NVVM70OpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=nvvm70)

add_public_tablegen_target(NVVM70OpsIncGen)
