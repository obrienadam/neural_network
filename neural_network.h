#pragma once

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Macros.h>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

namespace nn {

class NeuralNetwork {
public:
  void initialize(size_t input_dim, size_t output_dim,
                  const std::vector<size_t> &hidden_layer_dims);

  Eigen::VectorXd compute_output(std::vector<double> inputs) const;

  Eigen::MatrixXd compute_output(Eigen::MatrixXd inputs) const;

  size_t num_layers() const { return layer_bounds.size() - 1u; }

  size_t num_hidden_layers() const { return num_layers() - 2u; }

  size_t num_non_input_layers() const { return num_layers() - 1u; }

  size_t num_nodes_in_layer(size_t layer_no) const {
    return layer_bounds.at(layer_no + 1) - layer_bounds.at(layer_no);
  }

  size_t num_nodes() const {
    return std::accumulate(layer_bounds.begin(), layer_bounds.end(), 0u);
  }

  size_t num_non_input_nodes() const {
    return num_nodes() - num_nodes_in_layer(0u);
  }

  void set_activation_func(std::function<double(double)> f) {
    activation_func_ = std::move(f);
  }

  size_t input_size() const { return num_nodes_in_layer(0); }

protected:
  std::vector<size_t> layer_bounds;

  std::vector<Eigen::VectorXd> biases_;

  std::vector<Eigen::MatrixXd> input_weights_;

  std::function<double(double)> activation_func_ = [](const double x) {
    return std::abs(x);
  };
};

} // namespace nn
