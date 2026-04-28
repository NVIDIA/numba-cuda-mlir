// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===-- LLVM70Ops.td - LLVM70 target dialect definition ---------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM70_OPS
#define LLVM70_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// LLVM70 dialect definitions
//===----------------------------------------------------------------------===//

def LLVM70_Dialect : Dialect {
  let name = "nvvm_llvm70";
  let cppNamespace = "::mlir::llvm70";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let hasOperationAttrVerify = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// LLVM70 attribute base class
//===----------------------------------------------------------------------===//

class LLVM70_Attr<string attrName, string attrMnemonic,
    list<Trait> traits = []> : AttrDef<LLVM70_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// LLVM70 target attribute.
//===----------------------------------------------------------------------===//

def LLVM70_TargetAttr
    : LLVM70_Attr<"LLVM70Target", "target",
          [DeclareAttrInterfaceMethods<GPUTargetAttrVerifyInterface>]> {
  let description = [{
    GPU target attribute for compiling NVIDIA GPU modules through the old
    LLVM 7 C API (based on LLVM 7.0.1) and libnvvm.

    Examples:

    1. Target with default values.
    ```
      gpu.module @mymodule [#nvvm_llvm70.target] attributes {...} {
        ...
      }
    ```

    2. Target with `sm_80` chip and custom paths.
    ```
      gpu.module @mymodule [#nvvm_llvm70.target<chip = "sm_80">] {
        ...
      }
    ```
  }];
  let parameters = (ins
      DefaultValuedParameter<"unsigned", "3">:$O,
      StringRefParameter<"Target triple.", "\"nvptx64-nvidia-cuda\"">:$triple,
      StringRefParameter<"Target chip.", "\"sm_80\"">:$chip,
      StringRefParameter<"Path to old libLLVM.so.", "\"\"">:$libllvm,
      StringRefParameter<"Path to libnvvm.so.", "\"\"">:$libnvvm,
      StringRefParameter<"NVVM data layout.", "\"\"">:$dataLayout,
      OptionalParameter<"ArrayAttr", "Files to link to the module.">:$link
  );
  let assemblyFormat = [{
    (`<` struct($O, $triple, $chip, $libllvm, $libnvvm, $dataLayout, $link)^ `>`)?
  }];
  let builders = [AttrBuilder<
      (ins CArg<"unsigned", "3">:$O,
           CArg<"StringRef", "\"nvptx64-nvidia-cuda\"">:$triple,
           CArg<"StringRef", "\"sm_80\"">:$chip,
           CArg<"StringRef", "\"\"">:$libllvm,
           CArg<"StringRef", "\"\"">:$libnvvm,
           CArg<"StringRef", "\"\"">:$dataLayout,
           CArg<"ArrayAttr", "nullptr">:$linkFiles),
      [{
      return Base::get($_ctxt, O, triple, chip, libllvm, libnvvm, dataLayout, linkFiles);
    }]>];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

#endif // LLVM70_OPS
