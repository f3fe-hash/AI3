#include "layers/gating_layer.hpp"
#include <algorithm>

gating_layer::gating_layer(size_t size) : basic_layer(size)
{
    alpha.clear();
    galpha.clear();
}

gating_layer::~gating_layer()
{}

void gating_layer::init(size_t prev_size)
{
    this->prev_size = prev_size;
    if (alpha.empty())
    {
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        alpha.resize(size);
        galpha.resize(size);
        for (size_t i = 0; i < size; ++i)
            alpha[i] = dist(*gen); // small random start
    }
}

vec<float> gating_layer::forward(const vec<float>& in)
{
    last_input = in;
    vec<float> out(size, 0.0f);

    for (size_t i = 0; i < size; ++i)
    {
        float x = in[i];

        // 1️⃣ Clip x to avoid log(0), sqrt(negatives), and exp overflow
        x = std::clamp(x, -20.0f, 20.0f);

        // 2️⃣ Ensure positive for sqrt/log
        float safe_x = std::max(std::fabs(x), 1e-6f);

        // 3️⃣ Compute the inner safely
        float exp_neg_x = std::exp(-x);
        float inner = std::sqrt(safe_x) / (1.0f + exp_neg_x);

        // 4️⃣ Avoid log(0)
        inner = std::max(inner, 1e-6f);

        // 5️⃣ Compute gating
        float gated = x * std::exp(alpha[i] * std::log(inner));

        // 6️⃣ Avoid inf/nan
        if (!std::isfinite(gated))
            gated = 0.0f;

        out[i] = gated;
    }

    return out;
}

vec<float> gating_layer::backprop(const vec<float>& grads, dataset_config_t config)
{
    vec<float> dinputs(size, 0.0f);

    for (size_t i = 0; i < size; ++i)
    {
        float x = last_input[i];
        x = std::clamp(x, -20.0f, 20.0f);
        float safe_x = std::max(std::fabs(x), 1e-6f);
        float exp_neg_x = std::exp(-x);
        float inner = std::sqrt(safe_x) / (1.0f + exp_neg_x);
        inner = std::max(inner, 1e-6f);

        float gated = x * std::exp(alpha[i] * std::log(inner));

        // Derivatives
        float dlog_inner_dx = (0.5f / safe_x) - (exp_neg_x / (1.0f + exp_neg_x));
        float dgated_dx = std::exp(alpha[i] * std::log(inner)) * (1.0f + alpha[i] * x * dlog_inner_dx);
        float dgated_dalpha = gated * std::log(inner);

        if (!std::isfinite(dgated_dx)) dgated_dx = 0.0f;
        if (!std::isfinite(dgated_dalpha)) dgated_dalpha = 0.0f;

        dinputs[i] = grads[i] * dgated_dx;
        galpha[i] += grads[i] * dgated_dalpha;
    }

    for (size_t i = 0; i < size; ++i)
    {
        galpha[i] = std::clamp(galpha[i], -5.0f, 5.0f);
        alpha[i] -= config.lr * galpha[i];
    }

    return dinputs;
}
