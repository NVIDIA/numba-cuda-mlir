/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- LLVM70.h - LLVM70 dialect definition -----------------------*- C++ -*-===//
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM70_DIALECT_LLVM70_H
#define LLVM70_DIALECT_LLVM70_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#define GET_ATTRDEF_CLASSES
#include "llvm70/Dialect/LLVM70OpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "llvm70/Dialect/LLVM70Ops.h.inc"

#include "llvm70/Dialect/LLVM70OpsDialect.h.inc"

#endif // LLVM70_DIALECT_LLVM70_H
