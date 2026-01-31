/**
 * @file Parser.h
 * @brief JSON parser for Gawee IR graph files
 *
 * This parser reads the JSON + binary format exported by the Python translator.
 * The format consists of:
 * - graph.json: Graph structure (nodes, values, connections)
 * - weights/[name].bin: Binary weight files
 * - constants/[name].bin: Binary constant files
 */

#ifndef GAWEE_PARSER_H
#define GAWEE_PARSER_H

#include "Graph.h"
#include <string>
#include <memory>

namespace gawee {

/**
 * @brief Parser for Gawee IR JSON format
 *
 * Usage:
 *   auto graph = Parser::load("path/to/graph.json");
 *   if (graph) {
 *       graph->dump();
 *   }
 */
class Parser {
public:
    /**
     * @brief Load a graph from a JSON file
     *
     * @param jsonPath Path to graph.json file
     * @return Unique pointer to Graph, or nullptr on error
     *
     * The parser will:
     * 1. Read and parse the JSON file
     * 2. Create Value objects for all tensors
     * 3. Create Node objects for all operations
     * 4. Resolve weight references to binary file paths
     */
    static std::unique_ptr<Graph> load(const std::string& jsonPath);

private:
    // Internal parsing methods
    static bool parseValue(const void* json, Value& value);
    static bool parseNode(const void* json, Node& node);
    static bool parseWeightRef(const void* json, WeightRef& weight);
};

/**
 * @brief Utility to load binary weight data
 *
 * Usage:
 *   auto data = WeightLoader::load<float>(graph->baseDir + "/" + weight.path);
 */
class WeightLoader {
public:
    /**
     * @brief Load binary data from file
     *
     * @tparam T Data type (float, int32_t, etc.)
     * @param path Path to .bin file
     * @return Vector of loaded data, empty on error
     */
    template<typename T>
    static std::vector<T> load(const std::string& path);
};

} // namespace gawee

#endif // GAWEE_PARSER_H
