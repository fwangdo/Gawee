#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace gawee::back {

template <typename T, size_t Rank>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

template <typename T>
struct TensorBuffer {
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::vector<T> data;
};

struct NpyHeaderInfo {
  std::string dtypeTag;
  std::vector<int64_t> shape;
};

inline std::vector<int64_t> contiguousStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  int64_t running = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = running;
    running *= shape[static_cast<size_t>(i)];
  }
  return strides;
}

inline int64_t elementCount(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return 1;
  }
  int64_t count = 1;
  for (int64_t dim : shape) {
    count *= dim;
  }
  return count;
}

template <typename T>
TensorBuffer<T> makeZeroTensor(const std::vector<int64_t> &shape) {
  TensorBuffer<T> tensor;
  tensor.shape = shape;
  tensor.strides = contiguousStrides(shape);
  tensor.data.assign(static_cast<size_t>(elementCount(shape)), T{});
  return tensor;
}

template <typename T, size_t Rank>
MemRefDescriptor<T, Rank> makeDescriptor(TensorBuffer<T> &tensor) {
  if (tensor.shape.size() != Rank) {
    throw std::runtime_error("Rank mismatch while building memref descriptor");
  }
  MemRefDescriptor<T, Rank> desc{};
  desc.allocated = tensor.data.data();
  desc.aligned = tensor.data.data();
  desc.offset = 0;
  for (size_t i = 0; i < Rank; ++i) {
    desc.sizes[i] = tensor.shape[i];
    desc.strides[i] = tensor.strides[i];
  }
  return desc;
}

inline std::string trim(std::string_view value) {
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

inline std::string dtypeTagForF32() { return "<f4"; }
inline std::string dtypeTagForF64() { return "<f8"; }
inline std::string dtypeTagForI32() { return "<i4"; }
inline std::string dtypeTagForI64() { return "<i8"; }

template <typename T>
std::string dtypeTag();

template <>
inline std::string dtypeTag<float>() { return dtypeTagForF32(); }
template <>
inline std::string dtypeTag<double>() { return dtypeTagForF64(); }
template <>
inline std::string dtypeTag<int32_t>() { return dtypeTagForI32(); }
template <>
inline std::string dtypeTag<int64_t>() { return dtypeTagForI64(); }

inline std::vector<int64_t> parseShapeFromHeader(const std::string &header) {
  auto begin = header.find('(');
  auto end = header.find(')', begin);
  if (begin == std::string::npos || end == std::string::npos) {
    throw std::runtime_error("Failed to parse NPY shape");
  }
  std::string payload = header.substr(begin + 1, end - begin - 1);
  std::vector<int64_t> shape;
  std::stringstream ss(payload);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    if (item.empty()) {
      continue;
    }
    shape.push_back(std::stoll(item));
  }
  return shape;
}

inline NpyHeaderInfo readNpyHeader(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Could not open NPY file: " + path.string());
  }

  char magic[6];
  in.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") {
    throw std::runtime_error("Invalid NPY header: " + path.string());
  }

  char version[2];
  in.read(version, 2);
  uint16_t headerLen = 0;
  if (version[0] == 1) {
    uint16_t raw = 0;
    in.read(reinterpret_cast<char *>(&raw), sizeof(uint16_t));
    headerLen = raw;
  } else {
    uint32_t raw = 0;
    in.read(reinterpret_cast<char *>(&raw), sizeof(uint32_t));
    headerLen = static_cast<uint16_t>(raw);
  }

  std::string header(headerLen, '\0');
  in.read(header.data(), headerLen);
  if (header.find("'fortran_order': False") == std::string::npos) {
    throw std::runtime_error("Only C-order NPY arrays are supported");
  }

  NpyHeaderInfo info;
  info.shape = parseShapeFromHeader(header);

  auto descrPos = header.find("'descr': '");
  if (descrPos == std::string::npos) {
    throw std::runtime_error("Failed to parse NPY dtype: " + path.string());
  }
  descrPos += std::string("'descr': '").size();
  auto descrEnd = header.find("'", descrPos);
  if (descrEnd == std::string::npos) {
    throw std::runtime_error("Failed to parse NPY dtype end: " + path.string());
  }
  info.dtypeTag = header.substr(descrPos, descrEnd - descrPos);
  return info;
}

template <typename T>
TensorBuffer<T> loadNpy(const std::filesystem::path &path,
                        const std::vector<int64_t> &expectedShape) {
  NpyHeaderInfo info = readNpyHeader(path);
  std::vector<int64_t> shape = info.shape;
  std::string expectedDtype = dtypeTag<T>();
  if (info.dtypeTag != expectedDtype) {
    throw std::runtime_error("NPY dtype does not match requested tensor type");
  }
  if (shape.size() != expectedShape.size()) {
    throw std::runtime_error("Input tensor rank mismatch for " + path.string());
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (expectedShape[i] >= 0 && shape[i] != expectedShape[i]) {
      throw std::runtime_error("Input tensor shape mismatch for " + path.string());
    }
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Could not open input tensor: " + path.string());
  }
  char magic[6];
  in.read(magic, 6);
  char version[2];
  in.read(version, 2);
  if (version[0] == 1) {
    uint16_t raw = 0;
    in.read(reinterpret_cast<char *>(&raw), sizeof(uint16_t));
    in.seekg(raw, std::ios::cur);
  } else {
    uint32_t raw = 0;
    in.read(reinterpret_cast<char *>(&raw), sizeof(uint32_t));
    in.seekg(static_cast<std::streamoff>(raw), std::ios::cur);
  }

  TensorBuffer<T> tensor;
  tensor.shape = shape;
  tensor.strides = contiguousStrides(shape);
  tensor.data.resize(static_cast<size_t>(elementCount(shape)));
  in.read(reinterpret_cast<char *>(tensor.data.data()),
          static_cast<std::streamsize>(tensor.data.size() * sizeof(T)));
  return tensor;
}

template <typename T>
void saveNpy(const std::filesystem::path &path, const std::vector<int64_t> &shape,
             const T *data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Could not open output tensor: " + path.string());
  }

  std::ostringstream shapeStream;
  shapeStream << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    shapeStream << shape[i];
    if (shape.size() == 1) {
      shapeStream << ",";
    } else if (i + 1 != shape.size()) {
      shapeStream << ", ";
    }
  }
  shapeStream << ")";

  std::string header = "{'descr': '" + dtypeTag<T>() +
                       "', 'fortran_order': False, 'shape': " +
                       shapeStream.str() + ", }";
  while ((10 + header.size() + 1) % 16 != 0) {
    header.push_back(' ');
  }
  header.push_back('\n');

  const char magic[] = "\x93NUMPY";
  out.write(magic, 6);
  char version[2] = {1, 0};
  out.write(version, 2);
  uint16_t headerLen = static_cast<uint16_t>(header.size());
  out.write(reinterpret_cast<const char *>(&headerLen), sizeof(uint16_t));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(elementCount(shape) * sizeof(T)));
}

template <typename T, size_t Rank>
void saveReturnedMemRef(const std::filesystem::path &path,
                        const MemRefDescriptor<T, Rank> &desc) {
  std::vector<int64_t> shape(desc.sizes, desc.sizes + Rank);
  saveNpy(path, shape, desc.aligned + desc.offset);
}

template <typename T>
void saveTensorBuffer(const std::filesystem::path &path,
                      const TensorBuffer<T> &tensor) {
  saveNpy(path, tensor.shape, tensor.data.data());
}

template <typename T>
double maxAbsDiff(const TensorBuffer<T> &lhs, const TensorBuffer<T> &rhs) {
  if (lhs.shape != rhs.shape) {
    throw std::runtime_error("Shape mismatch while comparing tensors");
  }

  double maxDiff = 0.0;
  for (size_t i = 0; i < lhs.data.size(); ++i) {
    double diff = std::abs(static_cast<double>(lhs.data[i]) -
                           static_cast<double>(rhs.data[i]));
    maxDiff = std::max(maxDiff, diff);
  }
  return maxDiff;
}

template <typename T>
bool allClose(const TensorBuffer<T> &actual, const TensorBuffer<T> &expected,
              double atol, double rtol, double *maxDiffOut = nullptr) {
  if (actual.shape != expected.shape) {
    return false;
  }

  double maxDiff = 0.0;
  for (size_t i = 0; i < actual.data.size(); ++i) {
    double a = static_cast<double>(actual.data[i]);
    double b = static_cast<double>(expected.data[i]);
    double diff = std::abs(a - b);
    maxDiff = std::max(maxDiff, diff);
    double threshold = atol + rtol * std::abs(b);
    if (diff > threshold) {
      if (maxDiffOut) {
        *maxDiffOut = maxDiff;
      }
      return false;
    }
  }

  if (maxDiffOut) {
    *maxDiffOut = maxDiff;
  }
  return true;
}

} // namespace gawee::back
