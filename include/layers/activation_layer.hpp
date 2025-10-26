#pragma once

#include <string>
#include <iostream>

#include "layers/basic_layer.hpp"
#include "math/activations.hpp"

class activation_layer : public basic_layer
{
    // Activation & derivative
    activation act, dact;
    // saved input from forward for use in backprop
    vec<float> last_input;

public:
    activation_layer(size_t size, const std::string& activ);
    ~activation_layer();

    vec<float> forward(const vec<float>& in);
    vec<float> backprop(const vec<float>& grads, dataset_config_t config);
};
