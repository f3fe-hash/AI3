#include "layers/dense_layer.hpp"

dense_layer::dense_layer(size_t size, const std::string& activ) : basic_layer(size)
{
    linear = std::make_unique<linear_layer>(size);
    act = std::make_unique<activation_layer>(size, activ);
}

dense_layer::~dense_layer()
{}

void dense_layer::init(size_t prev_size)
{
    // propagate RNG to children (basic_layer::set_gen is non-virtual and was
    // already called on this object by the network, so forward the stored
    // generator to inner layers before they initialize their parameters)
    if (linear) linear->set_gen(this->gen);
    if (act) act->set_gen(this->gen);

    linear->init(prev_size);
    act->init(size);
}

vec<float> dense_layer::forward(const vec<float>& in)
{
    return act->forward(linear->forward(in));
}

vec<float> dense_layer::backprop(const vec<float>& grad, dataset_config_t config)
{
    return linear->backprop(act->backprop(grad, config), config);
}
