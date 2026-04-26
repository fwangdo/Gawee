#include "runtime_support.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct CompareConfig {
  fs::path actualDir;
  fs::path expectedDir;
  double atol = 1e-5;
  double rtol = 1e-5;
};

struct BenchmarkConfig {
  fs::path runner;
  fs::path inputsDir;
  fs::path outputsDir;
  int warmup = 3;
  int iterations = 10;
};

std::string quote(const fs::path &path) {
  return "\"" + path.string() + "\"";
}

std::vector<fs::path> listNpyFiles(const fs::path &dir) {
  if (!fs::exists(dir)) {
    throw std::runtime_error("Directory does not exist: " + dir.string());
  }

  std::vector<fs::path> files;
  for (const auto &entry : fs::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".npy") {
      files.push_back(entry.path().filename());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

template <typename T>
bool compareTypedFile(const fs::path &actualPath, const fs::path &expectedPath,
                      double atol, double rtol, double &maxDiff) {
  auto expectedInfo = gawee::back::readNpyHeader(expectedPath);
  auto actual = gawee::back::loadNpy<T>(actualPath, expectedInfo.shape);
  auto expected = gawee::back::loadNpy<T>(expectedPath, expectedInfo.shape);
  return gawee::back::allClose(actual, expected, atol, rtol, &maxDiff);
}

bool compareFile(const fs::path &actualPath, const fs::path &expectedPath,
                 double atol, double rtol, double &maxDiff) {
  auto expectedInfo = gawee::back::readNpyHeader(expectedPath);
  auto actualInfo = gawee::back::readNpyHeader(actualPath);

  if (expectedInfo.dtypeTag != actualInfo.dtypeTag) {
    throw std::runtime_error("Dtype mismatch: " + actualPath.string() +
                             " vs " + expectedPath.string());
  }
  if (expectedInfo.shape != actualInfo.shape) {
    throw std::runtime_error("Shape mismatch: " + actualPath.string() +
                             " vs " + expectedPath.string());
  }

  if (expectedInfo.dtypeTag == "<f4") {
    return compareTypedFile<float>(actualPath, expectedPath, atol, rtol,
                                   maxDiff);
  }
  if (expectedInfo.dtypeTag == "<f8") {
    return compareTypedFile<double>(actualPath, expectedPath, atol, rtol,
                                    maxDiff);
  }
  if (expectedInfo.dtypeTag == "<i4") {
    return compareTypedFile<int32_t>(actualPath, expectedPath, 0.0, 0.0,
                                     maxDiff);
  }
  if (expectedInfo.dtypeTag == "<i8") {
    return compareTypedFile<int64_t>(actualPath, expectedPath, 0.0, 0.0,
                                     maxDiff);
  }

  throw std::runtime_error("Unsupported dtype for comparison: " +
                           expectedInfo.dtypeTag);
}

int runCompare(const CompareConfig &config) {
  std::vector<fs::path> expectedFiles = listNpyFiles(config.expectedDir);
  if (expectedFiles.empty()) {
    throw std::runtime_error("No .npy files found in " +
                             config.expectedDir.string());
  }

  double globalMaxDiff = 0.0;
  for (const fs::path &filename : expectedFiles) {
    fs::path expectedPath = config.expectedDir / filename;
    fs::path actualPath = config.actualDir / filename;
    if (!fs::exists(actualPath)) {
      throw std::runtime_error("Missing actual output: " + actualPath.string());
    }

    double fileMaxDiff = 0.0;
    bool ok = compareFile(actualPath, expectedPath, config.atol, config.rtol,
                          fileMaxDiff);
    globalMaxDiff = std::max(globalMaxDiff, fileMaxDiff);
    std::cout << filename.string() << ": " << (ok ? "PASS" : "FAIL")
              << ", max_abs_diff=" << std::setprecision(8) << fileMaxDiff
              << "\n";
    if (!ok) {
      return 2;
    }
  }

  std::cout << "all outputs matched"
            << ", max_abs_diff=" << std::setprecision(8) << globalMaxDiff
            << "\n";
  return 0;
}

double runOnce(const BenchmarkConfig &config) {
  fs::create_directories(config.outputsDir);
  std::ostringstream cmd;
  cmd << quote(config.runner) << " " << quote(config.inputsDir) << " "
      << quote(config.outputsDir);

  auto begin = std::chrono::steady_clock::now();
  int rc = std::system(cmd.str().c_str());
  auto end = std::chrono::steady_clock::now();
  if (rc != 0) {
    throw std::runtime_error("Runner execution failed");
  }

  std::chrono::duration<double, std::milli> elapsed = end - begin;
  return elapsed.count();
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  double index = p * static_cast<double>(values.size() - 1);
  size_t lower = static_cast<size_t>(index);
  size_t upper = std::min(lower + 1, values.size() - 1);
  double frac = index - static_cast<double>(lower);
  return values[lower] * (1.0 - frac) + values[upper] * frac;
}

int runBenchmark(const BenchmarkConfig &config) {
  for (int i = 0; i < config.warmup; ++i) {
    (void)runOnce(config);
  }

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(config.iterations));
  for (int i = 0; i < config.iterations; ++i) {
    samples.push_back(runOnce(config));
  }

  double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  double avg = sum / static_cast<double>(samples.size());
  auto [minIt, maxIt] = std::minmax_element(samples.begin(), samples.end());

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "benchmark summary (ms)\n";
  std::cout << "  warmup: " << config.warmup << "\n";
  std::cout << "  iterations: " << config.iterations << "\n";
  std::cout << "  min: " << *minIt << "\n";
  std::cout << "  p50: " << percentile(samples, 0.50) << "\n";
  std::cout << "  avg: " << avg << "\n";
  std::cout << "  p95: " << percentile(samples, 0.95) << "\n";
  std::cout << "  max: " << *maxIt << "\n";
  std::cout << "note: this is end-to-end runner latency, so input load and "
               "output save costs are included.\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error(
          "Usage: gawee-eval <compare|benchmark> [options]");
    }

    std::string command = argv[1];
    if (command == "compare") {
      CompareConfig config;
      for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto nextValue = [&](const std::string &flag) -> std::string {
          if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + flag);
          }
          return argv[++i];
        };

        if (arg == "--actual") {
          config.actualDir = nextValue(arg);
        } else if (arg == "--expected") {
          config.expectedDir = nextValue(arg);
        } else if (arg == "--atol") {
          config.atol = std::stod(nextValue(arg));
        } else if (arg == "--rtol") {
          config.rtol = std::stod(nextValue(arg));
        } else {
          throw std::runtime_error("Unknown option: " + arg);
        }
      }

      if (config.actualDir.empty() || config.expectedDir.empty()) {
        throw std::runtime_error("--actual and --expected are required");
      }
      return runCompare(config);
    }

    if (command == "benchmark") {
      BenchmarkConfig config;
      for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto nextValue = [&](const std::string &flag) -> std::string {
          if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + flag);
          }
          return argv[++i];
        };

        if (arg == "--runner") {
          config.runner = nextValue(arg);
        } else if (arg == "--inputs") {
          config.inputsDir = nextValue(arg);
        } else if (arg == "--outputs") {
          config.outputsDir = nextValue(arg);
        } else if (arg == "--warmup") {
          config.warmup = std::stoi(nextValue(arg));
        } else if (arg == "--iters") {
          config.iterations = std::stoi(nextValue(arg));
        } else {
          throw std::runtime_error("Unknown option: " + arg);
        }
      }

      if (config.runner.empty() || config.inputsDir.empty() ||
          config.outputsDir.empty()) {
        throw std::runtime_error(
            "--runner, --inputs, and --outputs are required");
      }
      return runBenchmark(config);
    }

    throw std::runtime_error("Unsupported command: " + command);
  } catch (const std::exception &ex) {
    std::cerr << "gawee-eval error: " << ex.what() << "\n";
    return 1;
  }
}
