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

target_include_directories(LLVM70Translate PUBLIC
  ${LLVM70_SOURCE_DIR}/include
  ${LLVM70_BINARY_DIR}/include
)

set_target_properties(LLVM70Translate PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
  target_compile_options(LLVM70Translate PRIVATE /EHsc /wd4706)
endif()
if(WIN32)
  target_compile_definitions(LLVM70Translate PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
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
# Set LLVM70_MLIR_PYTHON_CAPI to libMLIRPythonCAPI.so, or to the
# MLIRPythonCAPI.lib import library on Windows, to share TypeIDs with the Python
# MLIR bindings and enable raw Operation* passing without text roundtrip.
add_library(MLIRToLLVM70 SHARED CAPI.cpp)
add_dependencies(MLIRToLLVM70 LLVM70OpsIncGen)
target_include_directories(MLIRToLLVM70 PRIVATE
  ${LLVM70_SOURCE_DIR}/include
  ${LLVM70_BINARY_DIR}/include
)
if(WIN32)
  set_target_properties(MLIRToLLVM70 PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
if(MSVC)
  target_compile_options(MLIRToLLVM70 PRIVATE /EHsc /wd4706)
endif()
if(WIN32)
  target_compile_definitions(MLIRToLLVM70 PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
target_link_libraries(MLIRToLLVM70 PRIVATE LLVM70Translate ${CMAKE_DL_LIBS})

if(LLVM70_MLIR_PYTHON_CAPI)
  message(STATUS "MLIRToLLVM70: linking against ${LLVM70_MLIR_PYTHON_CAPI}")
  target_link_libraries(MLIRToLLVM70 PRIVATE ${LLVM70_MLIR_PYTHON_CAPI})
  if(MSVC)
    # MLIRPythonCAPI.dll exports MLIR support symbols that are also present in
    # the static MLIR libraries pulled in by the C++ bridge.
    target_link_options(MLIRToLLVM70 PRIVATE /FORCE:MULTIPLE)
  endif()
endif()

install(TARGETS MLIRToLLVM70 DESTINATION lib)
