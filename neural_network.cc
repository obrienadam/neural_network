#include "neural_network.h"
#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <stdexcept>
#include <iostream>

namespace nn {

void NeuralNetwork::initialize(size_t input_dim, size_t output_dim, const std::vector<size_t> &hidden_layer_dims) {
    layer_bounds = {0u, input_dim};

    for(size_t hidden_layer_dim: hidden_layer_dims) {
        layer_bounds.push_back(layer_bounds.back() + hidden_layer_dim);
    }

    layer_bounds.push_back(layer_bounds.back() + output_dim);

    biases_.resize(num_non_input_layers());
    input_weights_.resize(num_non_input_layers());
    for(size_t layer = 0u; layer < num_non_input_layers(); ++layer) {
        biases_.at(layer).setZero(num_nodes_in_layer(layer + 1u));
        input_weights_.at(layer).setZero(num_nodes_in_layer(layer + 1u), num_nodes_in_layer(layer));
    }
}

Eigen::VectorXd NeuralNetwork::compute_output(std::vector<double> inputs) const {
    if(inputs.size() != num_nodes_in_layer(0)) {
        throw std::runtime_error("Bad input size!");
    }

    Eigen::VectorXd values = Eigen::Map<Eigen::VectorXd>(inputs.data(), inputs.size());

    for(size_t layer = 0u; layer < num_non_input_layers(); ++layer) {
        values = input_weights_.at(layer) * values + biases_.at(layer);

        for(int i = 0; i < values.rows(); ++i) {
            values(i) = activation_func_(values(i));
        }
    }

    return values;
}

}
