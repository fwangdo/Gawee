cmake_minimum_required(VERSION 3.20)
project(GaweeMLIR LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wno-reserved-user-defined-literal)

option(GAWEE_ENABLE_ONNX_PROTO "Enable ONNX protobuf parsing in ONNXMLIREmitter" OFF)
set(GAWEE_ONNX_INCLUDE_DIR "" CACHE PATH "Directory that contains onnx/onnx_pb.h and generated protobuf headers")

# Match LLVM's RTTI setting (LLVM was built with RTTI disabled)
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()

#===----------------------------------------------------------------------===#
# Find MLIR
#===----------------------------------------------------------------------===#

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

#===----------------------------------------------------------------------===#
# Include paths
#===----------------------------------------------------------------------===#

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/Gawee)
include_directories(${CMAKE_SOURCE_DIR}/include/Gawee/generated)

#===----------------------------------------------------------------------===#
# Gawee Dialect Library
#===----------------------------------------------------------------------===#

add_library(GaweeDialect
  lib/Gawee/GaweeDialect.cpp
)

# dependency A -> B. 
target_link_libraries(GaweeDialect
  PUBLIC
  MLIRIR
  MLIRSupport
)

#===----------------------------------------------------------------------===#
# Conversion Library (for later)
#===----------------------------------------------------------------------===#

add_library(GaweeConversion
  lib/Conversion/GaweeToLinalg.cpp
)

target_link_libraries(GaweeConversion
  PUBLIC
  GaweeDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRTransforms
)

#===----------------------------------------------------------------------===#
# gawee-opt Tool
#===----------------------------------------------------------------------===#

add_executable(gawee-opt
  tools/gawee-opt.cpp
)

# Get proper LLVM library handling
llvm_map_components_to_libnames(llvm_libs Support Core)

target_link_libraries(gawee-opt
  PRIVATE
  GaweeDialect
  GaweeConversion
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRRewrite
  MLIRTransforms
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRArithDialect
  MLIRArithTransforms
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  # LLVM dialect and conversions (Phase 7)
  MLIRLLVMDialect
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRReconcileUnrealizedCasts
  MLIRIR
  MLIRSupport
  ${llvm_libs}
)

#===----------------------------------------------------------------------===#
# Emitter Library (JSON -> MLIR)
#===----------------------------------------------------------------------===#

add_library(GaweeEmit
  lib/Emit/MLIREmitter.cpp
)

target_include_directories(GaweeEmit
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(GaweeEmit
  PUBLIC
  GaweeDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRIR
  MLIRSupport
)

#===----------------------------------------------------------------------===#
# Emitter Library (ONNX -> MLIR scaffold)
#===----------------------------------------------------------------------===#

add_library(GaweeONNXEmit
  lib/Emit/ONNXMLIREmitter.cpp
)

target_include_directories(GaweeONNXEmit
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

if(GAWEE_ENABLE_ONNX_PROTO)
  target_compile_definitions(GaweeONNXEmit PUBLIC GAWEE_ENABLE_ONNX_PROTO)

  if(NOT GAWEE_ONNX_INCLUDE_DIR)
    message(FATAL_ERROR
      "GAWEE_ENABLE_ONNX_PROTO=ON requires GAWEE_ONNX_INCLUDE_DIR to point to ONNX C++ headers")
  endif()

  if(NOT EXISTS "${GAWEE_ONNX_INCLUDE_DIR}/onnx/onnx.proto")
    message(FATAL_ERROR
      "GAWEE_ONNX_INCLUDE_DIR must contain onnx/onnx.proto. Got: ${GAWEE_ONNX_INCLUDE_DIR}")
  endif()

  if(NOT EXISTS "${GAWEE_ONNX_INCLUDE_DIR}/onnx/onnx_pb.h")
    message(FATAL_ERROR
      "GAWEE_ONNX_INCLUDE_DIR must contain onnx/onnx_pb.h. Got: ${GAWEE_ONNX_INCLUDE_DIR}")
  endif()

  find_package(Protobuf REQUIRED CONFIG)

  set(GAWEE_ONNX_PROTO_GEN_DIR "${CMAKE_BINARY_DIR}/onnx_proto_gen")
  file(MAKE_DIRECTORY "${GAWEE_ONNX_PROTO_GEN_DIR}")
  protobuf_generate(
    LANGUAGE cpp
    PROTOC_OUT_DIR "${GAWEE_ONNX_PROTO_GEN_DIR}"
    IMPORT_DIRS "${GAWEE_ONNX_INCLUDE_DIR}"
    OUT_VAR GAWEE_ONNX_PROTO_SRCS
    PROTOS "${GAWEE_ONNX_INCLUDE_DIR}/onnx/onnx.proto"
  )

  target_sources(GaweeONNXEmit PRIVATE ${GAWEE_ONNX_PROTO_SRCS})

  target_include_directories(GaweeONNXEmit
    PUBLIC
    ${GAWEE_ONNX_INCLUDE_DIR}
    ${GAWEE_ONNX_PROTO_GEN_DIR}
  )

  target_link_libraries(GaweeONNXEmit
    PUBLIC
    protobuf::libprotobuf
  )
endif()

target_link_libraries(GaweeONNXEmit
  PUBLIC
  GaweeDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRIR
  MLIRSupport
)

#===----------------------------------------------------------------------===#
# gawee-translate Tool (JSON -> MLIR)
#===----------------------------------------------------------------------===#

add_executable(gawee-translate
  tools/gawee-translate.cpp
)

target_link_libraries(gawee-translate
  PRIVATE
  GaweeEmit
  GaweeDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRIR
  MLIRSupport
  ${llvm_libs}
)

#===----------------------------------------------------------------------===#
# gawee-onnx-translate Tool (ONNX -> MLIR scaffold)
#===----------------------------------------------------------------------===#

add_executable(gawee-onnx-translate
  tools/gawee-onnx-translate.cpp
)

target_link_libraries(gawee-onnx-translate
  PRIVATE
  GaweeONNXEmit
  GaweeDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRIR
  MLIRSupport
  ${llvm_libs}
)
