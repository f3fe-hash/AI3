#pragma once

#include <cmath>
#include <functional>
#include "math/vec_utils.hpp"

using activation = std::function<vec<float>(vec<float>)>;

// ReLU
inline vec<float> relu_(vec<float> x)
{
    vec<float> out(x.size());
    std::transform(x.begin(), x.end(), out.begin(), [](float x_){
        return (x_ > 0) ? x_ : 0;
    });

    return out;
}

// Tanh
inline vec<float> tanh_(vec<float> x)
{
    vec<float> out(x.size());
    std::transform(x.begin(), x.end(), out.begin(), [](float x_){
        return std::tanh(x_);
    });

    return out;
}

// Sigmoid
inline vec<float> sigmoid_(vec<float> x)
{
    vec<float> out(x.size());
    std::transform(x.begin(), x.end(), out.begin(), [](float x_){
        return 1.0f / (1.0f + std::exp(-x_));
    });

    return out;
}

// Leaky ReLU
inline vec<float> lrelu_(vec<float> x)
{
    constexpr float alpha = 0.01f;
    vec<float> out(x.size());
    std::transform(x.begin(), x.end(), out.begin(), [](float x_){
        return (x_ > 0.0f) ? x_ : alpha * x_;
    });

    return out;
}

// ===========================
// Derivatives of activations
// ===========================

// ReLU
inline vec<float> drelu_(vec<float> x)
{
    vec<float> out(x.size());
    std::transform(x.begin(), x.end(), out.begin(), [](float x_){
        return (x_ > 0) ? 1 : 0;
    });

    return out;
}

// Tanh
inline vec<float> dtanh_(vec<float> x)
{
    vec<float> out(x.size());
    std::transform(x.begin(), x.end(), out.begin(), [](float x_){
        float t = std::tanh(x_);
        return 1.0f - t * t;
    });

    return out;
}

// Sigmoid
inline vec<float> dsigmoid_(vec<float> x)
{
    vec<float> out(x.size());
    std::transform(x.begin(), x.end(), out.begin(), [](float x_){
        float s = 1.0f / (1.0f + std::exp(-x_));
        return s * (1.0f - s);
    });

    return out;
}

// Leaky ReLU
inline vec<float> dlrelu_(vec<float> x)
{
    constexpr float alpha = 0.01f;
    vec<float> out(x.size());
    std::transform(x.begin(), x.end(), out.begin(), [](float x_){
        return (x_ > 0.0f) ? 1.0f : alpha;
    });

    return out;
}