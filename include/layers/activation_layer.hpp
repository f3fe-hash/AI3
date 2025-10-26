#pragma once

#include <string>
#include <memory>
#include "layers/basic_layer.hpp"
#include "math/activation_functions.hpp"

class activation_layer : public basic_layer
{
    std::unique_ptr<Activation> activation;

public:
    activation_layer(size_t size, const std::string& activ);
    ~activation_layer();

    vec<float> forward(const vec<float>& in) override;
    vec<float> backprop(const vec<float>& grads, dataset_config_t config) override;
};
