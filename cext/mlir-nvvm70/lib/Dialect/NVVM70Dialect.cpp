/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- NVVM70Dialect.cpp - MLIR dialect for NVVM70 ----------------*- C++ -*-===//
// 
//===----------------------------------------------------------------------===//

#include "nvvm70/Dialect/NVVM70.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

void mlir::nvvm70::NVVM70Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "nvvm70/Dialect/NVVM70Ops.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "nvvm70/Dialect/NVVM70OpsAttributes.cpp.inc"
      >();

  declarePromisedInterface<gpu::TargetAttrInterface, NVVM70TargetAttr>();
}

#define GET_TYPEDEF_CLASSES
#include "nvvm70/Dialect/NVVM70Ops.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "nvvm70/Dialect/NVVM70OpsAttributes.cpp.inc"

#include "nvvm70/Dialect/NVVM70OpsDialect.cpp.inc"

LogicalResult mlir::nvvm70::NVVM70Dialect::verifyOperationAttribute(
    Operation *op, NamedAttribute attr) {
  return success();
}

//===----------------------------------------------------------------------===//
// NVVM70 target attribute verification.
//===----------------------------------------------------------------------===//
LogicalResult nvvm70::NVVM70TargetAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, unsigned optLevel, StringRef triple,
    StringRef chip, StringRef libllvm, StringRef libnvvm, StringRef dataLayout,
    ArrayAttr files) {
  if (optLevel > 3) {
    emitError() << "The optimization level must be between 0 and 3.";
    return failure();
  }
  if (triple.empty()) {
    emitError() << "The target triple cannot be empty.";
    return failure();
  }
  if (chip.empty()) {
    emitError() << "The target chip cannot be empty.";
    return failure();
  }
  if (files && !llvm::all_of(files, [](Attribute attr) {
        return mlir::isa_and_nonnull<StringAttr>(attr);
      })) {
    emitError() << "All elements in the `link` array must be strings.";
    return failure();
  }
  return success();
}

LogicalResult nvvm70::NVVM70TargetAttr::verifyTarget(Operation *gpuModule) const {
  auto gpuModuleOp = llvm::dyn_cast<gpu::GPUModuleOp>(gpuModule);
  if (!gpuModuleOp) {
    return emitError(gpuModule->getLoc(),
                     "NVVM70 target attribute must be attached to a GPU module");
  }
  return success();
}
