#include "neural_network.h"
#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

namespace nn {

void NeuralNetwork::initialize(std::vector<size_t> layer_sizes) {
  if (layer_sizes.size() < 2) {
    throw std::invalid_argument("Must be at least two layers.");
  }

  layer_sizes_ = std::move(layer_sizes);

  weights_.resize(num_layers() - 1);
  biases_.resize(num_layers() - 1);

  for (int i = 0; i < num_layers() - 1; ++i) {
    weights_[i].setZero(layer_sizes_[i], layer_sizes_[i + 1]);
    biases_[i].setZero(layer_sizes_[i + 1]);
  }

  activation_funcs_.assign(num_layers() - 1, RELU);
  activation_funcs_.back() = SOFTMAX;
}

Matrix NeuralNetwork::infer(Matrix inputs) const {
  if (inputs.cols() != layer_sizes_[0]) {
    throw std::invalid_argument("Bad input size!");
  }

  for (int i = 1; i < num_layers(); ++i) {
    inputs = (inputs * weights_[i - 1]).rowwise() + biases_[i - 1];

    switch (activation_funcs_[i - 1]) {
    case RELU:
      inputs = inputs.unaryExpr([](double z) { return std::max(0., z); });
      break;

    case SOFTMAX:
      for (int i = 0; i < inputs.rows(); ++i) {
        double sum_exp_z = 0.;
        double m = inputs.row(i).maxCoeff();

        for (int j = 0; j < inputs.cols(); ++j) {
          sum_exp_z += std::exp(inputs(i, j) - m);
        }

        for (int j = 0; j < inputs.cols(); ++j) {
          inputs(i, j) = std::exp(inputs(i, j) - m) / sum_exp_z;
        }
      }

      break;
    }
  }

  return inputs;
}

std::vector<Matrix>
NeuralNetwork::compute_activations(const Matrix &inputs) const {
  if (inputs.cols() != layer_sizes_[0]) {
    throw std::invalid_argument("Bad input size!");
  }

  std::vector<Matrix> result;

  for (int i = 1; i < num_layers(); ++i) {
    const Matrix &previous_activation = i == 1 ? inputs : result.back();

    auto op =
        (previous_activation * weights_[i - 1]).rowwise() + biases_[i - 1];

    switch (activation_funcs_[i - 1]) {
    case RELU:
      result.emplace_back(
          op.unaryExpr([](double z) { return std::max(0., z); }));
      break;

    case SOFTMAX:
      auto z = op.eval();

      for (int i = 0; i < z.rows(); ++i) {
        double sum_exp_z = 0.;
        double m = z.row(i).maxCoeff();

        for (int j = 0; j < z.cols(); ++j) {
          sum_exp_z += std::exp(z(i, j) - m);
        }

        for (int j = 0; j < z.cols(); ++j) {
          z(i, j) = std::exp(z(i, j) - m) / sum_exp_z;
        }
      }

      result.emplace_back(std::move(z));
      break;
    }
  }

  return result;
}

void NeuralNetwork::randomize_weights(size_t seed) {
  std::mt19937_64 gen{seed};
  std::normal_distribution<double> dist(0., 10.);

  for (auto &weight_matrix : weights_) {
    weight_matrix = weight_matrix.unaryExpr([&](double) { return dist(gen); });
  }
}

void NeuralNetwork::randomize_biases(size_t seed) {
  std::mt19937_64 gen{seed};
  std::normal_distribution<double> dist(1., 1.);

  for (auto &bias : biases_) {
    bias = bias.unaryExpr([&](double) { return dist(gen); });
  }
}

} // namespace nn
