//===----------------------------------------------------------------------===//
// gawee-translate: JSON Graph to Gawee MLIR Translator
//===----------------------------------------------------------------------===//
//
// This tool reads a JSON graph file and emits Gawee MLIR.
//
// Usage:
//   gawee-translate <input.json> [-o output.mlir]
//
// Examples:
//   gawee-translate graph.json              # Print to stdout
//   gawee-translate graph.json -o out.mlir  # Write to file
//
//===----------------------------------------------------------------------===//

#include "Emit/MLIREmitter.h"
#include "Gawee/GaweeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input JSON file>"),
                                          cl::Required);

static cl::opt<std::string> outputFilename("o",
                                           cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Gawee JSON to MLIR translator\n");

  // Read input file
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = fileOrErr.getError()) {
    errs() << "Error: Could not open input file '" << inputFilename
           << "': " << ec.message() << "\n";
    return 1;
  }

  // Parse JSON
  auto jsonOrErr = json::parse(fileOrErr.get()->getBuffer());
  if (!jsonOrErr) {
    errs() << "Error: Failed to parse JSON: "
           << toString(jsonOrErr.takeError()) << "\n";
    return 1;
  }

  const auto *graph = jsonOrErr->getAsObject();
  if (!graph) {
    errs() << "Error: JSON root must be an object\n";
    return 1;
  }

  // Create MLIR context and load dialects
  MLIRContext context;
  context.loadDialect<gawee::GaweeDialect>();
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<func::FuncDialect>();

  // Emit MLIR
  gawee::MLIREmitter emitter(&context);
  auto module = emitter.emit(*graph);

  if (!module) {
    errs() << "Error: " << emitter.getError() << "\n";
    return 1;
  }

  // Open output file
  std::error_code ec;
  raw_fd_ostream output(outputFilename, ec);
  if (ec) {
    errs() << "Error: Could not open output file '" << outputFilename
           << "': " << ec.message() << "\n";
    return 1;
  }

  // Print MLIR
  module->print(output);
  output << "\n";

  return 0;
}
