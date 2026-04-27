cmake_minimum_required(VERSION 3.20)
project(GaweeMLIR LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Wno-reserved-user-defined-literal)

set(GAWEE_ONNX_INCLUDE_DIR "" CACHE PATH "Directory that contains the onnx/ package with onnx.proto and onnx_pb.h")

if(NOT GAWEE_ONNX_INCLUDE_DIR AND EXISTS "${CMAKE_SOURCE_DIR}/../.venv/lib/python3.12/site-packages/onnx/onnx_pb.h")
  set(GAWEE_ONNX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../.venv/lib/python3.12/site-packages"
      CACHE PATH "Directory that contains the onnx/ package with onnx.proto and onnx_pb.h" FORCE)
endif()

if(NOT Protobuf_DIR AND EXISTS "/opt/homebrew/Cellar/protobuf@21/21.12/lib/cmake/protobuf/protobuf-config.cmake")
  set(Protobuf_DIR "/opt/homebrew/Cellar/protobuf@21/21.12/lib/cmake/protobuf"
      CACHE PATH "Path to protobuf-config.cmake" FORCE)
endif()

if(NOT Protobuf_PROTOC_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/../.venv/lib/python3.12/site-packages/torch/bin/protoc")
  set(Protobuf_PROTOC_EXECUTABLE
      "${CMAKE_SOURCE_DIR}/../.venv/lib/python3.12/site-packages/torch/bin/protoc"
      CACHE FILEPATH "Path to protoc used for ONNX proto generation" FORCE)
endif()

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

if(NOT MLIR_TABLEGEN_EXE)
  find_program(MLIR_TABLEGEN_EXE
    NAMES mlir-tblgen
    HINTS "${LLVM_TOOLS_BINARY_DIR}" "$ENV{LLVM_INSTALL_DIR}/bin" "$ENV{HOME}/llvm-install/bin")
endif()

if(NOT MLIR_TABLEGEN_EXE)
  message(FATAL_ERROR "Could not find mlir-tblgen. Set LLVM_INSTALL_DIR or add mlir-tblgen to PATH.")
endif()

#===----------------------------------------------------------------------===#
# Include paths
#===----------------------------------------------------------------------===#

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/Gawee)
include_directories(${CMAKE_SOURCE_DIR}/include/Gawee/generated)

#===----------------------------------------------------------------------===#
# TableGen regeneration for Gawee dialect
#===----------------------------------------------------------------------===#

set(GAWEE_GENERATED_DIR "${CMAKE_SOURCE_DIR}/include/Gawee/generated")
set(GAWEE_TBLGEN_COMMON_INCLUDES
  -I "${LLVM_INCLUDE_DIRS}"
  -I "${MLIR_INCLUDE_DIRS}"
  -I "${CMAKE_SOURCE_DIR}/include"
  -I "${CMAKE_SOURCE_DIR}/include/Gawee")

set(GAWEE_GENERATED_FILES
  "${GAWEE_GENERATED_DIR}/GaweeDialect.h.inc"
  "${GAWEE_GENERATED_DIR}/GaweeDialect.cpp.inc"
  "${GAWEE_GENERATED_DIR}/GaweeOps.h.inc"
  "${GAWEE_GENERATED_DIR}/GaweeOps.cpp.inc")

add_custom_command(
  OUTPUT ${GAWEE_GENERATED_FILES}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GAWEE_GENERATED_DIR}"
  COMMAND ${MLIR_TABLEGEN_EXE} --gen-dialect-decls
          -I "${LLVM_INCLUDE_DIRS}"
          "${CMAKE_SOURCE_DIR}/include/Gawee/GaweeDialect.td"
          -o "${GAWEE_GENERATED_DIR}/GaweeDialect.h.inc"
  COMMAND ${MLIR_TABLEGEN_EXE} --gen-dialect-defs
          -I "${LLVM_INCLUDE_DIRS}"
          "${CMAKE_SOURCE_DIR}/include/Gawee/GaweeDialect.td"
          -o "${GAWEE_GENERATED_DIR}/GaweeDialect.cpp.inc"
  COMMAND ${MLIR_TABLEGEN_EXE} --gen-op-decls
          ${GAWEE_TBLGEN_COMMON_INCLUDES}
          "${CMAKE_SOURCE_DIR}/include/Gawee/GaweeOps.td"
          -o "${GAWEE_GENERATED_DIR}/GaweeOps.h.inc"
  COMMAND ${MLIR_TABLEGEN_EXE} --gen-op-defs
          ${GAWEE_TBLGEN_COMMON_INCLUDES}
          "${CMAKE_SOURCE_DIR}/include/Gawee/GaweeOps.td"
          -o "${GAWEE_GENERATED_DIR}/GaweeOps.cpp.inc"
  COMMAND ${CMAKE_COMMAND}
          -DGAWEE_OPS_CPP=${GAWEE_GENERATED_DIR}/GaweeOps.cpp.inc
          -P "${CMAKE_SOURCE_DIR}/cmake/FixGeneratedGaweeOps.cmake"
  DEPENDS
    "${CMAKE_SOURCE_DIR}/include/Gawee/GaweeDialect.td"
    "${CMAKE_SOURCE_DIR}/include/Gawee/GaweeOps.td"
  COMMENT "Regenerating Gawee dialect TableGen outputs"
  VERBATIM)

add_custom_target(GaweeIncGen DEPENDS ${GAWEE_GENERATED_FILES})

#===----------------------------------------------------------------------===#
# Gawee Dialect Library
#===----------------------------------------------------------------------===#

add_library(GaweeDialect
  lib/Gawee/GaweeDialect.cpp
)

add_dependencies(GaweeDialect GaweeIncGen)

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
  lib/Conversion/LinalgTransform.cpp
  lib/Conversion/LinalgFusion.cpp
  lib/Conversion/LinalgScheduling.cpp
  lib/Conversion/LinalgVectorization.cpp
  lib/Conversion/LinalgVerification.cpp
  lib/Conversion/BufferizePrep.cpp
)

add_dependencies(GaweeConversion GaweeIncGen)

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

add_dependencies(gawee-opt GaweeIncGen)

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

add_dependencies(GaweeEmit GaweeIncGen)

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

if(NOT GAWEE_ONNX_INCLUDE_DIR)
  message(FATAL_ERROR
    "GAWEE_ONNX_INCLUDE_DIR must point to ONNX C++ headers because ONNXMLIREmitter is now always enabled")
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

add_dependencies(GaweeONNXEmit GaweeIncGen)

target_include_directories(GaweeONNXEmit
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${GAWEE_ONNX_INCLUDE_DIR}
  ${GAWEE_ONNX_PROTO_GEN_DIR}
)

target_link_libraries(GaweeONNXEmit
  PUBLIC
  protobuf::libprotobuf
)

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

add_dependencies(gawee-translate GaweeIncGen)

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

add_dependencies(gawee-onnx-translate GaweeIncGen)

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
