#pragma once

#include <memory>
#include <algorithm>
#include "layers/basic_layer.hpp"
#include "layers/linear_layer.hpp"
#include "math/vec_utils.hpp"

class normalization_layer : public basic_layer
{
    std::unique_ptr<linear_layer> linear;
public:
    normalization_layer(size_t size);
    ~normalization_layer();

    void init(size_t prev_size) override;

    vec<float> forward(const vec<float>& in);
    vec<float> backprop(const vec<float>& grads, dataset_config_t config);
};
