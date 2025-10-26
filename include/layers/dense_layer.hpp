#pragma once

#include <memory>
#include "layers/basic_layer.hpp"
#include "layers/linear_layer.hpp"
#include "layers/activation_layer.hpp"

class dense_layer : public basic_layer
{
    std::unique_ptr<linear_layer> linear;
    std::unique_ptr<activation_layer> act;

public:
    dense_layer(size_t size, const std::string& activ);
    ~dense_layer();

    void init(size_t prev_size) override;

    vec<float> forward(const vec<float>& in);
    vec<float> backprop(const vec<float>& grad, dataset_config_t config);
};
