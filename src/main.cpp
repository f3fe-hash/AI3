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

    // For binary classification (XOR), use BCE loss
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(
        (vec<basic_layer *>){
            new dense_layer(2, "relu"),
            new dense_layer(10, "tanh"),    // hidden layer with tanh
            new dense_layer(1, "sigmoid")  // output layer with sigmoid for [0,1]
        },
        "bce"  // BCE loss for binary classification
    );

    std::cout << "Training XOR with BCE loss..." << std::endl;
    const size_t epochs = 10000;
    for (uint i = 0; i < epochs; ++i)
    {
        float loss = nn->backprop(dataset);
        if (i % 1000 == 0)
        {
            std::cout << "Epoch " << i << "/" << epochs 
                     << " - Loss: " << std::fixed << std::setprecision(PRECISION) 
                     << loss << std::endl;
        }
    }

    for (size_t i = 0; i < X.size(); ++i)
    {
        vec<float> out = nn->forward(X[i]);
        // Use BCE loss for evaluation
        float l = bce_loss(out, y[i]);
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
