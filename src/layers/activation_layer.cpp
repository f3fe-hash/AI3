#include "layers/activation_layer.hpp"

activation_layer::activation_layer(size_t size, const std::string& activ) : basic_layer(size)
{
    activation = Activation::create(activ);
}

activation_layer::~activation_layer() = default;

vec<float> activation_layer::forward(const vec<float>& in)
{
    return activation->forward(in);
}

vec<float> activation_layer::backprop(const vec<float>& grads, dataset_config_t config)
{
    (void)config;
    return activation->backward(grads);
}