/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- Target.h - NVVM70 target registration ---------------------*- C++ -*-===//
//
// Registers the TargetAttrInterface for the #nvvm70.target attribute.
//
//===----------------------------------------------------------------------===//

#ifndef NVVM70_TARGET_TARGET_H
#define NVVM70_TARGET_TARGET_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace nvvm70 {
/// Registers the `TargetAttrInterface` for the `#nvvm70.target` attribute in the
/// given registry.
void registerNVVM70TargetInterfaceExternalModels(mlir::DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#nvvm70.target` attribute in the
/// registry associated with the given context.
void registerNVVM70TargetInterfaceExternalModels(mlir::MLIRContext &context);
} // namespace nvvm70

#endif // NVVM70_TARGET_TARGET_H
