#include "nn.hpp"

NeuralNetwork::NeuralNetwork(vec<basic_layer *> layers_, const std::string& loss_type)
{
    gen = std::make_shared<std::mt19937>(rd());
    loss_functions = get_loss(loss_type);
    for (auto& layer : layers_)
        add_layer(layer);
}

NeuralNetwork::~NeuralNetwork()
{}

vec<float> NeuralNetwork::forward(vec<float> in)
{
    vec<float> out = in;
    for (auto& layer : layers)
        out = layer->forward(out);
    
    return out;
}

float NeuralNetwork::backprop(const dataset_t& dataset)
{
    float loss_ = 0.00f;
    for (size_t i = 0; i < dataset.size; ++i)
    {
        const vec<float>& X = dataset.data[i].first;
        const vec<float>& Y = dataset.data[i].second;

        // Forward pass
        vec<float> out = this->forward(X);

        loss_ += loss_functions.loss(out, Y);

        // Compute gradient of loss wrt output
        vec<float> grad = loss_functions.grad(Y, out);

        // Backward pass
        for (int l = layers.size() - 1; l >= 0; --l)
            grad = layers[l]->backprop(grad, dataset.config);
    }

    return loss_;
}
