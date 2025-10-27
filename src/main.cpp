#include <memory>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <SFML/Graphics.hpp>

#include "nn.hpp"
#include "math/dataset.hpp"

#include "layers/linear_layer.hpp"
#include "layers/activation_layer.hpp"
#include "layers/dense_layer.hpp"
#include "layers/normalization_layer.hpp"
#include "layers/gating_layer.hpp"

// 2 decimal places
#define PRECISION 2

void show_image(const vec<float>& data, int n, uint width = 28, uint height = 28);
void print_vector(vec<float> data);
int argmax(const vec<float>& data);

int main()
{
    dataset_t dataset = load_csv_dataset("datasets/fashion/fashion_mnist_train.csv", true);
    dataset.config.lr = 0.01;
    dataset.config.num_batches = 32;

    dataset_t test_dataset = dataset;

    // Output data size
    size_t ds = dataset.data[0].first.size();
    std::cout << "Dataset size: " <<  dataset.size << std::endl;
    std::cout << "Dataset input size: " <<  dataset.data[0].first.size() << std::endl;

    // Create neural network
    std::unique_ptr<NeuralNetwork> nn = std::make_unique<NeuralNetwork>(
        (vec<basic_layer *>){
            new normalization_layer(ds), // Normalize inputs & linear
            new activation_layer(ds, "tanh"), // Tanh
            new dense_layer(32, "tanh"), // Dense
            new dense_layer(16, "tanh"), // Dense
            new dense_layer(10, "softmax") // Output, softmax
        },
        "cce"
    );

    std::cout << "Training MNIST..." << std::endl;
    const size_t epochs = 50;
    for (uint i = 0; i < epochs; ++i)
    {
        float loss = nn->backprop(dataset);
        if (i)
        {
            std::cout << "Epoch " << i << "/" << epochs 
                     << " - Loss: " << std::fixed << std::setprecision(PRECISION) 
                     << loss << std::endl;
        }
    }

    std::cout << "Testing MNIST..." << std::endl;
    float accuracy = nn->test(test_dataset);
    std::cout << "Accuracy: " << accuracy * 100 << '%' << std::endl;
}

// Display a single MNIST-like image (28x28 pixels) in a window
void show_image(const vec<float>& data, int n, uint width, uint height)
{
    if (data.size() != width * height)
        throw std::runtime_error(std::string("Image data size mismatch: ") +
            std::to_string(width)  + "x" +
            std::to_string(height) + " != "
            + std::to_string(data.size()));

    sf::RenderWindow window(sf::VideoMode(width * 10, height * 10), std::string("MNIST Image") + std::to_string(n));
    sf::Image img;
    img.create(width, height);

    for (uint y = 0; y < height; ++y)
    {
        for (uint x = 0; x < width; ++x)
        {
            float val = data[y * width + x]; // pixel value
            uint8_t gray = static_cast<uint8_t>(std::round(val * 255.0f)); // scale 0..1 to 0..255
            img.setPixel(x, y, sf::Color(gray, gray, gray));
        }
    }

    sf::Texture texture;
    texture.loadFromImage(img);
    sf::Sprite sprite(texture);
    sprite.setScale(10.f, 10.f); // scale up for visibility

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(sprite);
        window.display();
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