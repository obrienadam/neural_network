#include <cmath>
#include <iostream>
#include <vector>

#include "neural_network.h"

int main() {
  using namespace std;

  nn::NeuralNetwork nn;
  nn.initialize({3, 4, 4, 2});
  nn.randomize_weights();
  nn.randomize_biases();

  nn::Matrix inputs(7, 3);

  inputs << 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1;

 // cout << nn.infer(inputs) << '\n';

  for(const auto& a: nn.compute_activations(inputs)) {
    std::cout << a << "\n\n";
  }

  return 0;
}
