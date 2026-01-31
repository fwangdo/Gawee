/**
 * @file Graph.h
 * @brief Graph representation for Gawee IR
 *
 * This file defines the in-memory representation of a neural network graph.
 * The graph consists of:
 * - Values: Tensors that flow between operations (activations)
 * - Nodes: Operations that transform values (Conv, ReLU, etc.)
 * - Weights: Binary data files for model parameters
 */

#ifndef GAWEE_GRAPH_H
#define GAWEE_GRAPH_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>

namespace gawee {

/**
 * @brief Represents tensor metadata for a value in the graph
 *
 * A Value represents a tensor that flows between operations.
 * It contains shape and dtype information, but not the actual data
 * (except for constants which have a file path to binary data).
 */
struct Value {
    std::string id;                      // Unique identifier
    std::vector<int64_t> shape;          // Tensor shape, e.g., [1, 64, 224, 224]
    std::string dtype;                   // Data type, e.g., "float32"
    std::optional<std::string> path;     // Path to binary file (for constants)

    // Helper: Check if this value is a constant (has binary data)
    bool isConstant() const { return path.has_value(); }

    // Helper: Get total number of elements
    int64_t numElements() const {
        int64_t n = 1;
        for (auto dim : shape) n *= dim;
        return n;
    }
};

/**
 * @brief Represents weight/parameter metadata
 *
 * Weights are stored in Node attributes and point to binary files.
 */
struct WeightRef {
    std::vector<int64_t> shape;          // Weight shape
    std::string dtype;                   // Data type
    std::string path;                    // Path to binary file
};

/**
 * @brief Represents an operation in the graph
 *
 * A Node performs a computation on input values to produce output values.
 * Examples: Conv, ReLU, MaxPool, Linear, etc.
 */
struct Node {
    std::string opType;                  // Operation type: "Conv", "Relu", etc.
    std::string name;                    // Node name for debugging
    std::vector<std::string> inputs;     // Input value IDs
    std::vector<std::string> outputs;    // Output value IDs

    // Attributes vary by operation type
    // Common attributes stored in a flexible map
    std::unordered_map<std::string, int64_t> intAttrs;
    std::unordered_map<std::string, double> floatAttrs;
    std::unordered_map<std::string, std::string> stringAttrs;
    std::unordered_map<std::string, std::vector<int64_t>> intArrayAttrs;
    std::unordered_map<std::string, WeightRef> weightAttrs;

    // Helper methods to get typed attributes with defaults
    int64_t getInt(const std::string& key, int64_t defaultVal = 0) const {
        auto it = intAttrs.find(key);
        return (it != intAttrs.end()) ? it->second : defaultVal;
    }

    std::vector<int64_t> getIntArray(const std::string& key) const {
        auto it = intArrayAttrs.find(key);
        return (it != intArrayAttrs.end()) ? it->second : std::vector<int64_t>{};
    }

    const WeightRef* getWeight(const std::string& key) const {
        auto it = weightAttrs.find(key);
        return (it != weightAttrs.end()) ? &it->second : nullptr;
    }
};

/**
 * @brief Represents the entire neural network graph
 *
 * The Graph is the top-level container that holds all values and nodes.
 * It also tracks which values are graph inputs and outputs.
 */
class Graph {
public:
    // Graph inputs and outputs (value IDs)
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    // All values in the graph, keyed by ID
    std::unordered_map<std::string, Value> values;

    // All nodes in topological order
    std::vector<Node> nodes;

    // Base directory for loading binary weight files
    std::string baseDir;

    // Lookup a value by ID
    const Value* getValue(const std::string& id) const {
        auto it = values.find(id);
        return (it != values.end()) ? &it->second : nullptr;
    }

    // Debug: Print graph summary
    void dump() const;
};

} // namespace gawee

#endif // GAWEE_GRAPH_H
