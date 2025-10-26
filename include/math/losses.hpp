#pragma once

#include <cmath>
#include <string>
#include <functional>
#include "math/vec_utils.hpp"

// Loss function type: takes predicted and target vectors, returns scalar loss
using loss_fn = std::function<float(const vec<float>&, const vec<float>&)>;

// Loss gradient type: takes target and predicted vectors, returns gradient w.r.t. predictions
using loss_grad_fn = std::function<vec<float>(const vec<float>&, const vec<float>&)>;

// Struct to hold both loss and its gradient
struct loss_pair {
    loss_fn loss;
    loss_grad_fn grad;
};

// Mean Squared Error loss
inline float mse_loss(const vec<float>& pred, const vec<float>& target)
{
    float loss = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i)
    {
        const float diff = pred[i] - target[i];
        loss += diff * diff;
    }
    return loss / pred.size();
}

inline vec<float> mse_grad(const vec<float>& target, const vec<float>& pred)
{
    vec<float> grad(pred.size());
    const float scale = 2.0f / pred.size();
    for (size_t i = 0; i < pred.size(); ++i)
        grad[i] = scale * (pred[i] - target[i]);
    
    return grad;
}

// Binary Cross-Entropy loss
inline float bce_loss(const vec<float>& pred, const vec<float>& target)
{
    float loss = 0.0f;
    const float eps = 1e-7f;  // prevent log(0)
    for (size_t i = 0; i < pred.size(); ++i)
    {
        const float p = std::max(std::min(pred[i], 1.0f - eps), eps);
        loss += -(target[i] * std::log(p) + (1.0f - target[i]) * std::log(1.0f - p));
    }
    return loss / pred.size();
}

inline vec<float> bce_grad(const vec<float>& target, const vec<float>& pred)
{
    vec<float> grad(pred.size());
    const float eps = 1e-7f;
    const float scale = 1.0f / pred.size();
    for (size_t i = 0; i < pred.size(); ++i)
    {
        const float p = std::max(std::min(pred[i], 1.0f - eps), eps);
        grad[i] = scale * (p - target[i]) / (p * (1.0f - p));
    }
    return grad;
}

// Factory function to get loss and gradient functions by name
inline loss_pair get_loss(const std::string& name)
{
    if (name == "mse" || name == "mean-squared-error")
        return {mse_loss, mse_grad};
    
    else if (name == "bce" || name == "binary-cross-entropy")
        return {bce_loss, bce_grad};
    
    // Default to MSE if unknown
    return {mse_loss, mse_grad};
}