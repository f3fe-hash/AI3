#include <memory>
#include <iostream>
#include <iomanip> // for std::setprecision

#include "nn.hpp"
#include "math/dataset.hpp"
#include "layers/linear_layer.hpp"
#include "layers/activation_layer.hpp"
#include "layers/dense_layer.hpp"

// 2 decimal places
#define PRECISION 2

void print_vector(vec<float> data);

int main()
{
    const vec2<float> X = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    const vec2<float> y = {
        {0},
        {1},
        {1},
        {0}
    };

    dataset_t dataset = create_dataset(X, y);
    dataset.config.lr = 0.01;
    //dataset.config.num_batches = 1;

    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(
        (vec<basic_layer *>){
            new dense_layer(2, "tanh"),
            new dense_layer(3, "relu"),
            new dense_layer(1, "sigmoid")
        }
    );

    for (uint i = 0; i < 10000; ++i)
        nn->backprop(dataset);

    for (size_t i = 0; i < X.size(); ++i)
    {
        vec<float> out = nn->forward(X[i]);
        float l = loss(y[i], out);
        std::cout << "Input: ";
        print_vector(X[i]);
        std::cout << " | Output: ";
        print_vector(out);
        std::cout << " | Loss: " << std::fixed << std::setprecision(PRECISION) << l << std::endl;
    }
}

void print_vector(vec<float> data)
{
    std::cout << "[";
    for (size_t i = 0; i < data.size(); ++i)
    {
        std::cout << std::fixed << std::setprecision(PRECISION) << data[i];
        if (i < data.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]";
}
