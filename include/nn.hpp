#pragma once

#include <random>
#include <memory>
#include <thread>
#include <mutex>
#include <fstream>

#include "layers/basic_layer.hpp"
#include "math/dataset.hpp"
#include "math/losses.hpp"

class NeuralNetwork
{
    vec<basic_layer *> layers;
    loss_pair loss_functions;

    std::random_device rd{};
    std::shared_ptr<std::mt19937> gen;

public:
    NeuralNetwork(vec<basic_layer *> layers_={}, const std::string& loss_type="mse");
    ~NeuralNetwork();

    void add_layer(basic_layer* layer)
    {
        layer->set_gen(gen);

        if (layers.size() != 0)
            layer->init(layers[layers.size() - 1]->get_size());
        else
            layer->init(0);

        layers.push_back(layer);
    }

    vec<float> forward(vec<float> in);
    float backprop(const dataset_t& dataset);

    float test(const dataset_t& test);
};

// Return the index of the largest value in a vector
inline int argmax(const vec<float>& data)
{
    int idx = 0;
    float max_val = data[0];
    for (size_t i = 1; i < data.size(); ++i)
    {
        if (data[i] > max_val)
        {
            max_val = data[i];
            idx = i;
        }
    }
    
    return idx;
}
