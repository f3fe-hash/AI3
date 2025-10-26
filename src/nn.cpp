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
    size_t batch_count = dataset.config.num_batches;
    if (batch_count == 0) batch_count = 1;

    size_t batch_size = (dataset.size + batch_count - 1) / batch_count;
    float loss_total = 0.0f;
    std::mutex mtx; // for safely accumulating loss

    auto worker = [&](size_t start, size_t end)
    {
        float thread_loss = 0.0f;

        // Create local copies of layers for gradient accumulation
        vec<vec2<float>> weight_grads(layers.size());
        vec<vec<float>> bias_grads(layers.size());

        for (size_t i = start; i < end; ++i)
        {
            const vec<float>& X = dataset.data[i].first;
            const vec<float>& Y = dataset.data[i].second;

            vec<float> out = this->forward(X);
            thread_loss += loss_functions.loss(out, Y);

            vec<float> grad = loss_functions.grad(Y, out);

            for (int l = layers.size() - 1; l >= 0; --l)
                grad = layers[l]->backprop(grad, dataset.config);
        }

        // Accumulate loss
        std::lock_guard<std::mutex> lock(mtx);
        loss_total += thread_loss;
    };

    std::vector<std::thread> threads;
    for (size_t b = 0; b < batch_count; ++b)
    {
        size_t start = b * batch_size;
        size_t end = std::min(start + batch_size, dataset.size);
        threads.emplace_back(worker, start, end);
    }

    for (auto& t : threads)
        t.join();

    return loss_total / dataset.size;
}

void NeuralNetwork::save(const std::string& filename)
{}

void NeuralNetwork::load(const std::string& filename)
{}
