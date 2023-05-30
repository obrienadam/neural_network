#pragma once

#include <Eigen/Dense>

namespace nn {

class LabeledExamples{

    void resize(size_t num_features, size_t label_size, size_t data_set_size);

    protected:
    Eigen::MatrixXd features_;
    Eigen::MatrixXd target_vectors_;
};

}
