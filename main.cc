#include <cmath>
#include <vector>
#include <iostream>

#include "neural_network.h"

int main() {
    using namespace std;

    nn::NeuralNetwork nn;
    
    nn.initialize(8, 5, {9, 9, 9});
    nn.set_activation_func([](double x) {
        return std::log(1 + std::exp(x));
    });

    std::vector<double> input_vals(nn.input_size());
    std::iota(input_vals.begin(), input_vals.end(), 1.);
    auto output = nn.compute_output(input_vals);

    std::cout << output << std::endl;

    return 0;
}
