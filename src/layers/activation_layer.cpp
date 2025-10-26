#include "layers/activation_layer.hpp"

activation_layer::activation_layer(size_t size, const std::string& activ) : basic_layer(size)
{
    if (activ == "relu")
    {
        act = relu_;
        dact = drelu_;
    }
    else if (activ == "tanh")
    {
        act = tanh_;
        dact = dtanh_;
    }
    else if (activ == "sigmoid")
    {
        act = sigmoid_;
        dact = dsigmoid_;
    }
    else if (activ == "lrelu" || activ == "leaky_relu")
    {
        act = lrelu_;
        dact = dlrelu_;
    }
    else
        std::cout << "Unknown activation: " << activ << std::endl;
}

activation_layer::~activation_layer()
{}

vec<float> activation_layer::forward(const vec<float>& in)
{
    // save input so derivative can be evaluated at the same point
    last_input = in;
    if (act)
        return act(in);
    // fallback: identity
    return in;
}

vec<float> activation_layer::backprop(const vec<float>& grads, dataset_config_t config)
{
    (void) config;

    // If no derivative function set, assume identity
    if (!dact)
        return grads;

    // compute derivative vector at saved input
    vec<float> der = dact(last_input);

    // elementwise multiply derivative by incoming grads
    if (der.size() != grads.size())
        return vec<float>();

    vec<float> out(grads.size());
    for (size_t i = 0; i < grads.size(); ++i)
        out[i] = grads[i] * der[i];

    return out;
}