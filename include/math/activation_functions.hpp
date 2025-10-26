#pragma once

#include <cmath>
#include <numeric>
#include <functional>
#include <memory>
#include <iostream>
#include "math/vec_utils.hpp"

// Base class for activations (allows virtual dispatch for proper derivatives)
class Activation
{
public:
    virtual ~Activation() = default;
    virtual vec<float> forward(const vec<float>& x) = 0;
    virtual vec<float> backward(const vec<float>& grad) = 0;
    
    // Factory method
    static std::unique_ptr<Activation> create(const std::string& name);
};

// ReLU activation
class ReLU : public Activation
{
    vec<float> last_input;
public:
    vec<float> forward(const vec<float>& x) override
    {
        last_input = x;
        vec<float> out(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            out[i] = x[i] > 0 ? x[i] : 0;

        return out;
    }
    
    vec<float> backward(const vec<float>& grad) override
    {
        vec<float> out(grad.size());
        for (size_t i = 0; i < grad.size(); ++i)
            out[i] = last_input[i] > 0 ? grad[i] : 0;
        return out;
    }
};

// Softmax activation
class Softmax : public Activation
{
    vec<float> last_output;  // store softmax output for backward pass
public:
    vec<float> forward(const vec<float>& x) override
    {
        // Subtract max for numerical stability
        const float max_x = *std::max_element(x.begin(), x.end());
        vec<float> exp_x(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            exp_x[i] = std::exp(x[i] - max_x);
        
        // Compute sum and normalize
        const float sum = std::accumulate(exp_x.begin(), exp_x.end(), 0.0f);
        for (size_t i = 0; i < x.size(); ++i)
            exp_x[i] /= sum;
        
        last_output = exp_x;  // save for backward pass
        return exp_x;
    }
    
    vec<float> backward(const vec<float>& grad) override
    {
        vec<float> out(grad.size(), 0.0f);

        // Compute Jacobian-vector product: J * grad where J_ij = s_i * (delta_ij - s_j)
        for (size_t i = 0; i < grad.size(); ++i)
        {
            for (size_t j = 0; j < grad.size(); ++j)
                out[i] += last_output[i] * ((i == j ? 1.0f : 0.0f) - last_output[j]) * grad[j];
        }
        return out;
    }
};

// Sigmoid activation
class Sigmoid : public Activation
{
    vec<float> last_output;
public:
    vec<float> forward(const vec<float>& x) override
    {
        vec<float> out(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            out[i] = 1.0f / (1.0f + std::exp(-x[i]));
        last_output = out;
        return out;
    }
    
    vec<float> backward(const vec<float>& grad) override
    {
        vec<float> out(grad.size());
        for (size_t i = 0; i < grad.size(); ++i)
            out[i] = last_output[i] * (1.0f - last_output[i]) * grad[i];
        return out;
    }
};

// Tanh activation
class Tanh : public Activation
{
    vec<float> last_output;
public:
    vec<float> forward(const vec<float>& x) override
    {
        vec<float> out(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            out[i] = std::tanh(x[i]);
        last_output = out;
        return out;
    }
    
    vec<float> backward(const vec<float>& grad) override
    {
        vec<float> out(grad.size());
        for (size_t i = 0; i < grad.size(); ++i)
            out[i] = (1.0f - last_output[i] * last_output[i]) * grad[i];
        return out;
    }
};

// Implementation of factory method
inline std::unique_ptr<Activation> Activation::create(const std::string& name)
{
    if (name == "relu")
        return std::make_unique<ReLU>();
    else if (name == "softmax")
        return std::make_unique<Softmax>();
    else if (name == "sigmoid")
        return std::make_unique<Sigmoid>();
    else if (name == "tanh")
        return std::make_unique<Tanh>();
    
    std::cout << "Unknown activation: " << name << ", defaulting to ReLU\n";
    return std::make_unique<ReLU>();
}