# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(LLVM70Translate
  LLVM70IRBuilder.cpp
  LibNVVMCompiler.cpp
  LLVM70Target.cpp
  Dialect/LLVM70Dialect.cpp
  Target/Target.cpp
)

add_dependencies(LLVM70Translate LLVM70OpsIncGen)
configure_llvm70_target(LLVM70Translate)

target_include_directories(LLVM70Translate PUBLIC
  ${LLVM70_SOURCE_DIR}/include
  ${LLVM70_BINARY_DIR}/include
)

set_target_properties(LLVM70Translate PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
  target_compile_options(LLVM70Translate PRIVATE /wd4706)
endif()

target_link_libraries(LLVM70Translate
  PUBLIC
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIROpenMPDialect
  MLIRIR
  MLIRSupport
  LLVMSupport
  ${CMAKE_DL_LIBS}
)

install(TARGETS LLVM70Translate DESTINATION lib)

# Shared library with C API for in-process use from Python.
# Set LLVM70_MLIR_PYTHON_CAPI to the path of libMLIRPythonCAPI.so to link
# against it and share TypeIDs with the Python MLIR bindings, enabling
# raw Operation* passing without text roundtrip.
add_library(MLIRToLLVM70 SHARED CAPI.cpp)
add_dependencies(MLIRToLLVM70 LLVM70OpsIncGen)
configure_llvm70_target(MLIRToLLVM70)
target_include_directories(MLIRToLLVM70 PRIVATE
  ${LLVM70_SOURCE_DIR}/include
  ${LLVM70_BINARY_DIR}/include
)
if(WIN32)
  set_target_properties(MLIRToLLVM70 PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
if(MSVC)
  target_compile_options(MLIRToLLVM70 PRIVATE /wd4706)
endif()

if(MSVC AND LLVM70_MLIR_PYTHON_CAPI)
  message(STATUS "MLIRToLLVM70: linking against ${LLVM70_MLIR_PYTHON_CAPI}")
  target_link_libraries(MLIRToLLVM70 PRIVATE
    ${LLVM70_MLIR_PYTHON_CAPI}
    LLVM70Translate
    ${CMAKE_DL_LIBS}
  )
  # MLIRPythonCAPI.dll exports MLIR support symbols that are also present in
  # the static MLIR libraries pulled in by the C++ bridge.
  target_link_options(MLIRToLLVM70 PRIVATE /FORCE:MULTIPLE)
else()
  target_link_libraries(MLIRToLLVM70 PRIVATE LLVM70Translate ${CMAKE_DL_LIBS})
endif()

if(LLVM70_MLIR_PYTHON_CAPI AND NOT MSVC)
  message(STATUS "MLIRToLLVM70: linking against ${LLVM70_MLIR_PYTHON_CAPI}")
  target_link_libraries(MLIRToLLVM70 PRIVATE ${LLVM70_MLIR_PYTHON_CAPI})
endif()

install(TARGETS MLIRToLLVM70 DESTINATION lib)
