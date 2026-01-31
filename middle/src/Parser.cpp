/**
 * @file Parser.cpp
 * @brief JSON parser implementation for Gawee IR
 *
 * This file implements the parsing logic for the JSON graph format.
 * We use nlohmann/json library for JSON parsing - it's header-only
 * and provides a clean, modern C++ API.
 */

#include "gawee/Parser.h"

// nlohmann/json - a popular header-only JSON library for C++
// https://github.com/nlohmann/json
#include "../third_party/json.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>

// Use 'json' as an alias for the nlohmann::json type
using json = nlohmann::json;

namespace gawee {

/**
 * @brief Parse a single Value from JSON
 *
 * JSON format:
 * {
 *   "id": "conv1",
 *   "shape": [1, 64, 112, 112],
 *   "dtype": "torch.float32",
 *   "path": "constants/xxx.bin"  // optional, only for constants
 * }
 */
bool Parser::parseValue(const void* jsonPtr, Value& value) {
    const json& j = *static_cast<const json*>(jsonPtr);

    try {
        // Required fields
        value.id = j.at("id").get<std::string>();
        value.dtype = j.at("dtype").get<std::string>();

        // Shape can be null for some values
        if (j.contains("shape") && !j["shape"].is_null()) {
            for (const auto& dim : j["shape"]) {
                value.shape.push_back(dim.get<int64_t>());
            }
        }

        // Optional path for constants
        if (j.contains("path")) {
            value.path = j["path"].get<std::string>();
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Parser] Error parsing value: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Parse a WeightRef from JSON
 *
 * JSON format:
 * {
 *   "shape": [64, 3, 7, 7],
 *   "dtype": "float32",
 *   "path": "weights/conv1_weight_0.bin"
 * }
 */
bool Parser::parseWeightRef(const void* jsonPtr, WeightRef& weight) {
    const json& j = *static_cast<const json*>(jsonPtr);

    try {
        weight.dtype = j.at("dtype").get<std::string>();
        weight.path = j.at("path").get<std::string>();

        for (const auto& dim : j.at("shape")) {
            weight.shape.push_back(dim.get<int64_t>());
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Parser] Error parsing weight: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Parse a single Node from JSON
 *
 * JSON format:
 * {
 *   "op_type": "Conv",
 *   "name": "conv1",
 *   "inputs": ["x"],
 *   "outputs": ["conv1"],
 *   "attrs": {
 *     "kernel_size": [7, 7],
 *     "stride": [2, 2],
 *     "weight": { "shape": [...], "dtype": "...", "path": "..." },
 *     ...
 *   }
 * }
 */
bool Parser::parseNode(const void* jsonPtr, Node& node) {
    const json& j = *static_cast<const json*>(jsonPtr);

    try {
        // Basic fields
        node.opType = j.at("op_type").get<std::string>();
        node.name = j.at("name").get<std::string>();

        // Input/output connections
        for (const auto& inp : j.at("inputs")) {
            node.inputs.push_back(inp.get<std::string>());
        }
        for (const auto& out : j.at("outputs")) {
            node.outputs.push_back(out.get<std::string>());
        }

        // Parse attributes
        if (j.contains("attrs")) {
            const json& attrs = j["attrs"];

            for (auto it = attrs.begin(); it != attrs.end(); ++it) {
                const std::string& key = it.key();
                const json& val = it.value();

                // Skip internal attributes
                if (key == "target" || key == "op" || key == "mod" || key == "op_type") {
                    continue;
                }

                // Determine attribute type and store appropriately
                if (val.is_number_integer()) {
                    // Integer attribute (e.g., out_channels, groups)
                    node.intAttrs[key] = val.get<int64_t>();
                }
                else if (val.is_number_float()) {
                    // Float attribute (e.g., eps, momentum)
                    node.floatAttrs[key] = val.get<double>();
                }
                else if (val.is_boolean()) {
                    // Boolean as integer (e.g., inplace, ceil_mode)
                    node.intAttrs[key] = val.get<bool>() ? 1 : 0;
                }
                else if (val.is_string()) {
                    // String attribute
                    node.stringAttrs[key] = val.get<std::string>();
                }
                else if (val.is_array()) {
                    // Array of integers (e.g., kernel_size, stride, padding)
                    std::vector<int64_t> arr;
                    for (const auto& elem : val) {
                        if (elem.is_number_integer()) {
                            arr.push_back(elem.get<int64_t>());
                        }
                    }
                    if (!arr.empty()) {
                        node.intArrayAttrs[key] = arr;
                    }
                }
                else if (val.is_object() && val.contains("path")) {
                    // Weight reference (has path, shape, dtype)
                    WeightRef weight;
                    if (parseWeightRef(&val, weight)) {
                        node.weightAttrs[key] = weight;
                    }
                }
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Parser] Error parsing node: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Load a complete graph from JSON file
 *
 * Main entry point for parsing. Reads the JSON file and constructs
 * the in-memory Graph representation.
 */
std::unique_ptr<Graph> Parser::load(const std::string& jsonPath) {
    // Open and parse JSON file
    std::ifstream file(jsonPath);
    if (!file.is_open()) {
        std::cerr << "[Parser] Cannot open file: " << jsonPath << std::endl;
        return nullptr;
    }

    json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        std::cerr << "[Parser] JSON parse error: " << e.what() << std::endl;
        return nullptr;
    }

    // Create graph
    auto graph = std::make_unique<Graph>();

    // Store base directory for loading weights
    std::filesystem::path p(jsonPath);
    graph->baseDir = p.parent_path().string();

    // Parse inputs
    if (j.contains("inputs")) {
        for (const auto& inp : j["inputs"]) {
            graph->inputs.push_back(inp.get<std::string>());
        }
    }

    // Parse outputs
    if (j.contains("outputs")) {
        for (const auto& out : j["outputs"]) {
            graph->outputs.push_back(out.get<std::string>());
        }
    }

    // Parse values
    if (j.contains("values")) {
        for (auto it = j["values"].begin(); it != j["values"].end(); ++it) {
            Value value;
            if (parseValue(&it.value(), value)) {
                graph->values[it.key()] = value;
            }
        }
    }

    // Parse nodes
    if (j.contains("nodes")) {
        for (const auto& nodeJson : j["nodes"]) {
            Node node;
            if (parseNode(&nodeJson, node)) {
                graph->nodes.push_back(node);
            }
        }
    }

    std::cout << "[Parser] Loaded graph from " << jsonPath << std::endl;
    std::cout << "[Parser]   Values: " << graph->values.size() << std::endl;
    std::cout << "[Parser]   Nodes: " << graph->nodes.size() << std::endl;

    return graph;
}

// Template instantiation for common types
template<>
std::vector<float> WeightLoader::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[WeightLoader] Cannot open: " << path << std::endl;
        return {};
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate number of elements
    size_t numElements = fileSize / sizeof(float);

    // Read data
    std::vector<float> data(numElements);
    file.read(reinterpret_cast<char*>(data.data()), fileSize);

    return data;
}

} // namespace gawee
