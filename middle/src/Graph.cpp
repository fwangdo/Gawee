/**
 * @file Graph.cpp
 * @brief Implementation of Graph methods
 */

#include "gawee/Graph.h"
#include <iostream>
#include <iomanip>

namespace gawee {

void Graph::dump() const {
    std::cout << "=== Gawee Graph ===" << std::endl;

    // Print inputs
    std::cout << "\n[Inputs]" << std::endl;
    for (const auto& inputId : inputs) {
        const Value* v = getValue(inputId);
        if (v) {
            std::cout << "  " << inputId << ": shape=[";
            for (size_t i = 0; i < v->shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << v->shape[i];
            }
            std::cout << "], dtype=" << v->dtype << std::endl;
        }
    }

    // Print outputs
    std::cout << "\n[Outputs]" << std::endl;
    for (const auto& outputId : outputs) {
        const Value* v = getValue(outputId);
        if (v) {
            std::cout << "  " << outputId << ": shape=[";
            for (size_t i = 0; i < v->shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << v->shape[i];
            }
            std::cout << "], dtype=" << v->dtype << std::endl;
        }
    }

    // Print nodes
    std::cout << "\n[Nodes] (" << nodes.size() << " total)" << std::endl;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const Node& node = nodes[i];
        std::cout << "  [" << i << "] " << node.opType << " (" << node.name << ")" << std::endl;

        // Inputs
        std::cout << "      inputs: [";
        for (size_t j = 0; j < node.inputs.size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << node.inputs[j];
        }
        std::cout << "]" << std::endl;

        // Outputs
        std::cout << "      outputs: [";
        for (size_t j = 0; j < node.outputs.size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << node.outputs[j];
        }
        std::cout << "]" << std::endl;

        // Print some key attributes based on op type
        if (node.opType == "Conv") {
            auto kernel = node.getIntArray("kernel_size");
            auto stride = node.getIntArray("stride");
            auto padding = node.getIntArray("padding");
            std::cout << "      kernel_size=[";
            for (size_t j = 0; j < kernel.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << kernel[j];
            }
            std::cout << "], stride=[";
            for (size_t j = 0; j < stride.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << stride[j];
            }
            std::cout << "], padding=[";
            for (size_t j = 0; j < padding.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << padding[j];
            }
            std::cout << "]" << std::endl;

            // Weight info
            if (auto* w = node.getWeight("weight")) {
                std::cout << "      weight: shape=[";
                for (size_t j = 0; j < w->shape.size(); ++j) {
                    if (j > 0) std::cout << ", ";
                    std::cout << w->shape[j];
                }
                std::cout << "], path=" << w->path << std::endl;
            }
        }
    }

    // Summary
    std::cout << "\n[Summary]" << std::endl;
    std::cout << "  Values: " << values.size() << std::endl;
    std::cout << "  Nodes: " << nodes.size() << std::endl;
    std::cout << "  Inputs: " << inputs.size() << std::endl;
    std::cout << "  Outputs: " << outputs.size() << std::endl;
}

} // namespace gawee
