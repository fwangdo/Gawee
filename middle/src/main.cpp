/**
 * @file main.cpp
 * @brief Example usage of the Gawee parser
 *
 * This demonstrates how to:
 * 1. Load a graph from JSON
 * 2. Inspect the graph structure
 * 3. Load weight data from binary files
 */

#include "gawee/Parser.h"
#include <iostream>

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph.json>" << std::endl;
        std::cerr << "Example: " << argv[0] << " jsondata/graph.json" << std::endl;
        return 1;
    }

    std::string jsonPath = argv[1];

    // Load the graph
    std::cout << "Loading graph from: " << jsonPath << std::endl;
    auto graph = gawee::Parser::load(jsonPath);

    if (!graph) {
        std::cerr << "Failed to load graph!" << std::endl;
        return 1;
    }

    // Print graph summary
    graph->dump();

    // Example: Load weight data for the first Conv node
    std::cout << "\n=== Loading Weight Example ===" << std::endl;
    for (const auto& node : graph->nodes) {
        if (node.opType == "Conv") {
            if (auto* weight = node.getWeight("weight")) {
                std::string weightPath = graph->baseDir + "/" + weight->path;
                std::cout << "Loading weight from: " << weightPath << std::endl;

                auto data = gawee::WeightLoader::load<float>(weightPath);
                if (!data.empty()) {
                    std::cout << "  Loaded " << data.size() << " floats" << std::endl;
                    std::cout << "  First 5 values: ";
                    for (size_t i = 0; i < std::min<size_t>(5, data.size()); ++i) {
                        std::cout << data[i] << " ";
                    }
                    std::cout << std::endl;
                }
            }
            break; // Just show the first Conv
        }
    }

    return 0;
}
