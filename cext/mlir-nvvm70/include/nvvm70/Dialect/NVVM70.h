/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- NVVM70.h - NVVM70 dialect definition -----------------------*- C++ -*-===//
// 
//===----------------------------------------------------------------------===//

#ifndef NVVM70_DIALECT_NVVM70_H
#define NVVM70_DIALECT_NVVM70_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#define GET_ATTRDEF_CLASSES
#include "nvvm70/Dialect/NVVM70OpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "nvvm70/Dialect/NVVM70Ops.h.inc"

#include "nvvm70/Dialect/NVVM70OpsDialect.h.inc"

#endif // NVVM70_DIALECT_NVVM70_H
