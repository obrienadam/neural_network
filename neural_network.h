#pragma once

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Macros.h>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

namespace nn {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;

enum ActivationFunction { RELU, SOFTMAX};

class NeuralNetwork {
public:

  void initialize(std::vector<size_t> layer_sizes);

  Matrix infer(Matrix inputs) const;

  std::vector<Matrix> compute_activations(const Matrix& input) const;

  void randomize_weights(size_t seed = 0);
  
  void randomize_biases(size_t seed = 1);

  size_t num_layers() const {
    return layer_sizes_.size();
  }

protected:

  std::vector<size_t> layer_sizes_;

  std::vector<Matrix> weights_;

  std::vector<RowVector> biases_;

  std::vector<ActivationFunction> activation_funcs_;

  std::vector<Matrix> activations;
};

} // namespace nn
