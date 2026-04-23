//===----------------------------------------------------------------------===//
// gawee-onnx-translate: ONNX Graph to Gawee MLIR Translator
//===----------------------------------------------------------------------===//
//
// This tool is the ONNX-front parallel to gawee-translate.
//
// Current status:
//   - scaffold only
//   - wires command line, MLIR context, and ONNXMLIREmitter
//   - real ONNX parsing/emission is TODO
//
//===----------------------------------------------------------------------===//

#include "Emit/ONNXMLIREmitter.h"
#include "Gawee/GaweeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input ONNX file>"),
                                          cl::Required);

static cl::opt<std::string> outputFilename("o",
                                           cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Gawee ONNX to MLIR translator\n");

  MLIRContext context;
  context.loadDialect<gawee::GaweeDialect>();
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<linalg::LinalgDialect>();
  context.loadDialect<tensor::TensorDialect>();

  gawee::ONNXMLIREmitter emitter(&context);
  auto module = emitter.emitFromFile(inputFilename);
  if (!module) {
    errs() << "Error: " << emitter.getError() << "\n";
    return 1;
  }

  std::error_code ec;
  raw_fd_ostream output(outputFilename, ec);
  if (ec) {
    errs() << "Error: Could not open output file '" << outputFilename
           << "': " << ec.message() << "\n";
    return 1;
  }

  module->print(output);
  output << "\n";
  return 0;
}
