#include "layers/linear_layer.hpp"

linear_layer::linear_layer(size_t size) : basic_layer(size)
{
    // delay initialization until we know the input size (from forward)
    weights.clear();
    biases.clear();
    weight_grads.clear();
    bias_grads.clear();
}

linear_layer::~linear_layer()
{
}

void linear_layer::init(size_t prev_size)
{
    this->prev_size = prev_size;
    if (weights.empty())
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        weights.resize(size);
        biases.resize(size);

        for (uint i = 0; i < size; ++i)
        {
            weights[i].resize(prev_size); // Each neuron connects to all neurons in previous layer
            for (uint j = 0; j < prev_size; ++j)
                weights[i][j] = dist(*gen);
            biases[i] = dist(*gen);
        }
    }
}

vec<float> linear_layer::forward(const vec<float>& in)
{
    if (prev_size == 0)
        return in;
    
    // Save input for use in backprop
    last_input = in;

    vec<float> out(size, 0.00f);
    for (size_t i = 0; i < size; ++i)
        out[i] = mult_add(in, weights[i], biases[i]);
    
    return out;
}

vec<float> linear_layer::backprop(const vec<float>& grads, dataset_config_t config)
{
    if (prev_size == 0 || grads.size() != size)
        return {};

    const size_t in_sz = last_input.size();
    vec<float> input_grads(in_sz, 0.0f);

    if (weight_grads.size() != size || (in_sz > 0 && weight_grads[0].size() != in_sz))
        weight_grads.assign(size, vec<float>(in_sz, 0.0f));
    bias_grads.assign(size, 0.0f);

    for (size_t out_i = 0; out_i < size; ++out_i)
    {
        // bias gradient is simply the output gradient
        bias_grads[out_i] = grads[out_i];

        for (size_t in_i = 0; in_i < in_sz; ++in_i)
        {
            // compute gradient for this weight (outer product)
            const float wgrad = grads[out_i] * last_input[in_i];
            weight_grads[out_i][in_i] = wgrad;

            // accumulate gradient w.r.t. input using the original weight
            if (!weights.empty())
                input_grads[in_i] += weights[out_i][in_i] * grads[out_i];

            // apply gradient descent update to the weight (after using old weight)
            if (!weights.empty())
                weights[out_i][in_i] -= config.lr * wgrad;
        }

        // apply gradient descent update to the bias
        if (!biases.empty())
            biases[out_i] -= config.lr * bias_grads[out_i];
    }

    return input_grads;
}
