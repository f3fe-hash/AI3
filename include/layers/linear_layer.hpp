#pragma once

#include "layers/basic_layer.hpp"
#include "math/dataset.hpp"

class linear_layer : public basic_layer
{
    vec2<float> weights;
    vec<float> biases;

    // gradients accumulated by backprop
    vec2<float> weight_grads;
    vec<float> bias_grads;

public:
    linear_layer(size_t size);
    ~linear_layer();

    vec<float> forward(const vec<float>& in);
    vec<float> backprop(const vec<float>& grads, dataset_config_t config);

    void init(size_t prev_size);

    // accessors for parameter gradients (const refs)
    const vec2<float>& get_weight_grads() const { return weight_grads; }
    const vec<float>& get_bias_grads() const { return bias_grads; }
};
