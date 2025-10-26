#include "layers/normalization_layer.hpp"

normalization_layer::normalization_layer(size_t size) : basic_layer(size)
{
    linear = std::make_unique<linear_layer>(size);
}

normalization_layer::~normalization_layer()
{}

void normalization_layer::init(size_t prev_size)
{
    // propagate RNG to children (basic_layer::set_gen is non-virtual and was
    // already called on this object by the network, so forward the stored
    // generator to inner layers before they initialize their parameters)
    if (linear)
        linear->set_gen(this->gen);
    linear->init(prev_size);
}

vec<float> normalization_layer::forward(const vec<float>& in)
{
    float mean = 0.00f;
    float var = 0.00f;
    vec<float> norm(in.size());
    last_input = in;

    vec<float> centered(in.size());
    for (size_t i = 0; i < in.size(); ++i)
        centered[i] = in[i] - mean;

    for (size_t i = 0 ; i < in.size(); ++i)
        mean += in[i];
    mean /= in.size();

    for (size_t i = 0; i < in.size(); ++i)
        var += centered[i] * centered[i];
    var /= (in.size() - 1);

    float inv_std = 1.0f / std::sqrt(var + 1e-8f);
    for (size_t i = 0; i < in.size(); ++i)
        norm[i] = centered[i] * inv_std;

    return linear->forward(norm);
}

vec<float> normalization_layer::backprop(const vec<float>& grad, dataset_config_t config)
{
    // Backprop through linear layer first
    vec<float> grad_norm = linear->backprop(grad, config);

    size_t n = grad_norm.size();
    float mean = 0.0f;
    float var = 0.0f;
    vec<float> in = last_input; // You need to save the input in forward pass

    for (float v : in)
        mean += v;
    mean /= n;

    for (float v : in)
        var += (v - mean) * (v - mean);
    var /= (n - 1);

    float eps = 1e-8f;
    float inv_std = 1.0f / std::sqrt(var + eps);

    // Compute gradient through normalization
    float grad_sum = 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        grad_sum += grad_norm[i];
        dot += (in[i] - mean) * grad_norm[i];
    }
    dot /= (n - 1);

    vec<float> grad_input(n);
    for (size_t i = 0; i < n; ++i)
        grad_input[i] = inv_std * (grad_norm[i] - grad_sum / n - (in[i] - mean) * dot);

    return grad_input;
}
