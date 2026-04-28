# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS LLVM70Ops.td)

mlir_tablegen(LLVM70Ops.h.inc -gen-op-decls)
mlir_tablegen(LLVM70Ops.cpp.inc -gen-op-defs)
mlir_tablegen(LLVM70OpsDialect.h.inc -gen-dialect-decls -dialect=nvvm_llvm70)
mlir_tablegen(LLVM70OpsDialect.cpp.inc -gen-dialect-defs -dialect=nvvm_llvm70)
mlir_tablegen(LLVM70OpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=nvvm_llvm70)
mlir_tablegen(LLVM70OpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=nvvm_llvm70)

add_public_tablegen_target(LLVM70OpsIncGen)
