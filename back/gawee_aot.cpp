#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct TensorTypeSpec {
  std::string dtype;
  std::vector<int64_t> shape;
};

struct FuncSignature {
  std::string entry;
  std::vector<TensorTypeSpec> args;
  std::vector<TensorTypeSpec> returns;
};

std::string readFile(const fs::path &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Could not open file: " + path.string());
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string trim(std::string_view value) {
  size_t begin = 0;
  size_t end = value.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return std::string(value.substr(begin, end - begin));
}

std::vector<std::string> splitTopLevelCommaList(const std::string &text) {
  std::vector<std::string> items;
  int angleDepth = 0;
  int parenDepth = 0;
  size_t itemStart = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    char ch = text[i];
    if (ch == '<') {
      ++angleDepth;
    } else if (ch == '>') {
      --angleDepth;
    } else if (ch == '(') {
      ++parenDepth;
    } else if (ch == ')') {
      --parenDepth;
    } else if (ch == ',' && angleDepth == 0 && parenDepth == 0) {
      items.push_back(trim(std::string_view(text).substr(itemStart, i - itemStart)));
      itemStart = i + 1;
    }
  }
  if (itemStart < text.size()) {
    items.push_back(trim(std::string_view(text).substr(itemStart)));
  }
  return items;
}

TensorTypeSpec parseMemRefType(const std::string &text) {
  std::smatch match;
  std::regex pattern(R"(memref<([0-9x]+)x([a-z0-9]+)>)");
  if (!std::regex_search(text, match, pattern)) {
    throw std::runtime_error("Only static memref types are supported: " + text);
  }

  TensorTypeSpec spec;
  std::string shapePart = match[1].str();
  spec.dtype = match[2].str();
  auto dims = splitTopLevelCommaList(std::regex_replace(shapePart, std::regex("x"), ","));
  for (const auto &dim : dims) {
    spec.shape.push_back(std::stoll(dim));
  }
  return spec;
}

FuncSignature parseFuncSignature(const fs::path &path, const std::string &entry) {
  std::string text = readFile(path);
  std::regex funcPattern(
      "func\\.func\\s+@" + entry +
      R"(\(([\s\S]*?)\)\s*(->\s*([\s\S]*?))?\s*\{)");
  std::smatch match;
  if (!std::regex_search(text, match, funcPattern)) {
    throw std::runtime_error("Could not find func.func @" + entry + " in " +
                             path.string());
  }

  FuncSignature signature;
  signature.entry = entry;

  std::string argsBlock = match[1].str();
  for (const auto &item : splitTopLevelCommaList(argsBlock)) {
    if (item.empty()) {
      continue;
    }
    auto colonPos = item.find(':');
    if (colonPos == std::string::npos) {
      throw std::runtime_error("Malformed function argument: " + item);
    }
    signature.args.push_back(parseMemRefType(trim(item.substr(colonPos + 1))));
  }

  std::string returnsBlock = trim(match[3].str());
  if (!returnsBlock.empty()) {
    if (returnsBlock.rfind("(", 0) == 0 && returnsBlock.back() == ')') {
      returnsBlock = returnsBlock.substr(1, returnsBlock.size() - 2);
    }
    for (const auto &item : splitTopLevelCommaList(returnsBlock)) {
      if (item.empty()) {
        continue;
      }
      signature.returns.push_back(parseMemRefType(item));
    }
  }

  return signature;
}

std::string cppScalarType(const std::string &dtype) {
  if (dtype == "f32") return "float";
  if (dtype == "f64") return "double";
  if (dtype == "i32") return "int32_t";
  if (dtype == "i64") return "int64_t";
  throw std::runtime_error("Unsupported dtype: " + dtype);
}

std::string descriptorType(const TensorTypeSpec &spec) {
  std::ostringstream out;
  out << "gawee::back::MemRefDescriptor<" << cppScalarType(spec.dtype) << ", "
      << spec.shape.size() << ">";
  return out.str();
}

std::string emitShapeList(const std::vector<int64_t> &shape) {
  std::ostringstream out;
  out << "{";
  for (size_t i = 0; i < shape.size(); ++i) {
    out << shape[i];
    if (i + 1 != shape.size()) {
      out << ", ";
    }
  }
  out << "}";
  return out.str();
}

std::string emitFunctionPrototype(const FuncSignature &sig) {
  std::ostringstream out;
  if (sig.returns.empty()) {
    out << "extern \"C\" void " << sig.entry << "(";
  } else if (sig.returns.size() == 1) {
    out << "extern \"C\" " << descriptorType(sig.returns[0]) << " " << sig.entry
        << "(";
  } else {
    throw std::runtime_error("Multiple return values are not supported yet");
  }

  bool first = true;
  for (const auto &arg : sig.args) {
    size_t rank = arg.shape.size();
    if (!first) out << ", ";
    out << cppScalarType(arg.dtype) << " *";
    out << ", " << cppScalarType(arg.dtype) << " *";
    out << ", int64_t";
    for (size_t i = 0; i < rank; ++i) out << ", int64_t";
    for (size_t i = 0; i < rank; ++i) out << ", int64_t";
    first = false;
  }
  out << ");\n";
  return out.str();
}

std::string emitLauncher(const FuncSignature &sig, int numOutputArgs) {
  if (numOutputArgs < 0 ||
      static_cast<size_t>(numOutputArgs) > sig.args.size()) {
    throw std::runtime_error("Invalid --num-output-args value");
  }
  size_t numInputArgs = sig.args.size() - static_cast<size_t>(numOutputArgs);

  std::ostringstream out;
  out << "#include <cstdint>\n"
      << "#include <filesystem>\n"
      << "#include <iostream>\n"
      << "#include <string>\n"
      << "#include \"runtime_support.h\"\n\n";
  out << emitFunctionPrototype(sig) << "\n";
  out << "int main(int argc, char **argv) {\n"
      << "  if (argc != 3) {\n"
      << "    std::cerr << \"Usage: \" << argv[0] << \" <inputs_dir> <outputs_dir>\\n\";\n"
      << "    return 1;\n"
      << "  }\n"
      << "  namespace fs = std::filesystem;\n"
      << "  fs::path inputsDir(argv[1]);\n"
      << "  fs::path outputsDir(argv[2]);\n"
      << "  fs::create_directories(outputsDir);\n";

  for (size_t i = 0; i < numInputArgs; ++i) {
    const auto &arg = sig.args[i];
    out << "  auto arg" << i << "Tensor = gawee::back::loadNpy<"
        << cppScalarType(arg.dtype) << ">(inputsDir / \"arg" << i
        << ".npy\", " << emitShapeList(arg.shape) << ");\n";
    out << "  auto arg" << i << "Desc = gawee::back::makeDescriptor<"
        << cppScalarType(arg.dtype) << ", " << arg.shape.size()
        << ">(arg" << i << "Tensor);\n";
  }
  for (size_t i = numInputArgs; i < sig.args.size(); ++i) {
    const auto &arg = sig.args[i];
    out << "  auto arg" << i << "Tensor = gawee::back::makeZeroTensor<"
        << cppScalarType(arg.dtype) << ">(" << emitShapeList(arg.shape) << ");\n";
    out << "  auto arg" << i << "Desc = gawee::back::makeDescriptor<"
        << cppScalarType(arg.dtype) << ", " << arg.shape.size()
        << ">(arg" << i << "Tensor);\n";
  }

  out << "  ";
  if (!sig.returns.empty()) {
    out << "auto result = ";
  }
  out << sig.entry << "(";
  bool first = true;
  for (size_t i = 0; i < sig.args.size(); ++i) {
    if (!first) out << ", ";
    out << "arg" << i << "Desc.allocated, arg" << i << "Desc.aligned, arg" << i
        << "Desc.offset";
    for (size_t dim = 0; dim < sig.args[i].shape.size(); ++dim) {
      out << ", arg" << i << "Desc.sizes[" << dim << "]";
    }
    for (size_t dim = 0; dim < sig.args[i].shape.size(); ++dim) {
      out << ", arg" << i << "Desc.strides[" << dim << "]";
    }
    first = false;
  }
  out << ");\n";

  for (size_t i = numInputArgs; i < sig.args.size(); ++i) {
    const auto &arg = sig.args[i];
    out << "  gawee::back::saveTensorBuffer(outputsDir / \"output"
        << (i - numInputArgs) << ".npy\", arg" << i << "Tensor);\n";
  }
  if (!sig.returns.empty()) {
    out << "  gawee::back::saveReturnedMemRef(outputsDir / \"output"
        << numOutputArgs << ".npy\", result);\n";
  }

  out << "  return 0;\n}\n";
  return out.str();
}

void runChecked(const std::string &cmd) {
  int rc = std::system(cmd.c_str());
  if (rc != 0) {
    throw std::runtime_error("Command failed: " + cmd);
  }
}

fs::path translateIfNeeded(const fs::path &input, const fs::path &llvmBin) {
  if (input.extension() == ".ll") {
    return input;
  }
  if (input.extension() != ".mlir") {
    throw std::runtime_error("Unsupported lowered input: " + input.string());
  }

  fs::path out = fs::temp_directory_path() / "gawee_aot_lowered.ll";
  fs::path mlirTranslate = llvmBin / "mlir-translate";
  std::ostringstream cmd;
  cmd << mlirTranslate.string() << " --mlir-to-llvmir "
      << input.string() << " -o " << out.string();
  runChecked(cmd.str());
  return out;
}

void writeText(const fs::path &path, const std::string &text) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not write file: " + path.string());
  }
  out << text;
}

void buildExecutable(const fs::path &abiSource,
                     const fs::path &loweredInput,
                     const fs::path &output,
                     const std::string &entry,
                     int numOutputArgs,
                     const fs::path &llvmBin) {
  FuncSignature sig = parseFuncSignature(abiSource, entry);
  std::string launcher = emitLauncher(sig, numOutputArgs);

  fs::path launcherPath = fs::temp_directory_path() / "gawee_aot_launcher.cpp";
  writeText(launcherPath, launcher);

  fs::path llvmIr = translateIfNeeded(loweredInput, llvmBin);
  fs::create_directories(output.parent_path());

  std::ostringstream cmd;
  cmd << "/usr/bin/clang++ -std=c++17 -O2 -I" << fs::absolute("back").string()
      << " " << launcherPath.string() << " " << llvmIr.string()
      << " -o " << output.string();
  runChecked(cmd.str());
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error(
          "Usage: gawee-aot build --abi-source <forward_memref.mlir> "
          "--input <lowered.mlir|lowered.ll> --output <runner> "
          "[--entry forward] [--num-output-args N] [--llvm-bin <dir>]");
    }

    std::string command = argv[1];
    if (command != "build") {
      throw std::runtime_error("Only the 'build' subcommand is supported");
    }

    fs::path abiSource;
    fs::path input;
    fs::path output;
    std::string entry = "forward";
    int numOutputArgs = 0;
    fs::path llvmBin = fs::path(std::getenv("HOME")) / "llvm-install" / "bin";

    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      auto nextValue = [&](const std::string &flag) -> std::string {
        if (i + 1 >= argc) {
          throw std::runtime_error("Missing value for " + flag);
        }
        return argv[++i];
      };

      if (arg == "--abi-source") {
        abiSource = nextValue(arg);
      } else if (arg == "--input") {
        input = nextValue(arg);
      } else if (arg == "--output") {
        output = nextValue(arg);
      } else if (arg == "--entry") {
        entry = nextValue(arg);
      } else if (arg == "--num-output-args") {
        numOutputArgs = std::stoi(nextValue(arg));
      } else if (arg == "--llvm-bin") {
        llvmBin = nextValue(arg);
      } else {
        throw std::runtime_error("Unknown option: " + arg);
      }
    }

    if (abiSource.empty() || input.empty() || output.empty()) {
      throw std::runtime_error(
          "--abi-source, --input, and --output are required");
    }

    buildExecutable(abiSource, input, output, entry, numOutputArgs, llvmBin);
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "gawee-aot error: " << ex.what() << "\n";
    return 1;
  }
}
