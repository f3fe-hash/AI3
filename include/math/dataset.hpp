#pragma once

#include <cmath>
#include "math/vec_utils.hpp"

using data_pair = std::pair<vec<float>, vec<float>>;

struct dataset_config_t
{
    float lr;
};

struct dataset_t
{
    vec<data_pair> data;
    size_t size;
    dataset_config_t config;
};

inline dataset_t create_dataset(const vec2<float>& X, const vec2<float>& y)
{
    dataset_t dataset;
    dataset.size = X.size();
    dataset.data.resize(dataset.size);
    for (size_t i = 0; i < dataset.size; ++i)
        dataset.data[i] = (data_pair){X[i], y[i]};
    
    return dataset;
}

inline float loss(const vec<float>& pred_out, const vec<float> real_out)
{
    float loss = 0.00f;
    for (size_t i = 0; i < pred_out.size(); ++i)
        loss += std::pow(pred_out[i] - real_out[i], 2);
    
    return loss;
}

inline vec<float> loss_grad(const vec<float>& real_y, const vec<float>& pred_y, uint64_t n)
{
    vec<float> grad(n);
    for (size_t i = 0; i < n; ++i)
    {
        // d/d(pred) of (pred - real)^2 = 2 * (pred - real)
        grad[i] = 2 * (pred_y[i] - real_y[i]);
    }

    return grad;
}
