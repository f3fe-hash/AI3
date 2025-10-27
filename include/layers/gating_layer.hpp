#pragma once
#include "basic_layer.hpp"
#include <cmath>

class gating_layer : public basic_layer
{
    vec<float> alpha;      // learnable gating params
    vec<float> galpha;     // gradients
    vec<float> last_input; // store last input for backprop

public:
    gating_layer(size_t size);
    ~gating_layer();

    void init(size_t prev_size) override;
    vec<float> forward(const vec<float>& in);
    vec<float> backprop(const vec<float>& grads, dataset_config_t config);
};
