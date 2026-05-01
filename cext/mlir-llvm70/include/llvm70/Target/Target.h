/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- Target.h - LLVM70 target registration ---------------------*- C++ -*-===//
//
// Registers the TargetAttrInterface for the #nvvm_llvm70.target attribute.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM70_TARGET_TARGET_H
#define LLVM70_TARGET_TARGET_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace llvm70 {
/// Registers the `TargetAttrInterface` for the `#nvvm_llvm70.target` attribute in the
/// given registry.
void registerLLVM70TargetInterfaceExternalModels(mlir::DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#nvvm_llvm70.target` attribute in the
/// registry associated with the given context.
void registerLLVM70TargetInterfaceExternalModels(mlir::MLIRContext &context);
} // namespace llvm70

#endif // LLVM70_TARGET_TARGET_H
