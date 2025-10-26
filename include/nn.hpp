#pragma once

#include <random>
#include <memory>

#include "layers/basic_layer.hpp"
#include "math/dataset.hpp"

class NeuralNetwork
{
    vec<basic_layer *> layers;

    std::random_device rd{};
    std::shared_ptr<std::mt19937> gen;

public:
    NeuralNetwork(vec<basic_layer *> layers_={});
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
};
