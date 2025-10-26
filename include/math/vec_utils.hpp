#pragma once

#include <vector>
#include <cmath>
#include <cstdint>

template <typename T>
using vec = std::vector<T>;

template <typename T>
using vec2 = std::vector<std::vector<T>>;

template <typename T>
using vec3 = std::vector<std::vector<std::vector<T>>>;

// =====================
// 1D vector operations
// =====================

inline float mult_add(const vec<float>& a, const vec<float>& b, float c) noexcept
{
    float sum = 0.00f;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    
    return sum + c;
}

// =====================
// 2D vector operations
// =====================

inline vec<float> mult_add2(const vec2<float>& a, const vec2<float>& b, const vec<float>& c) noexcept
{
    vec<float> sum = c;
    for (size_t i = 0; i < a.size(); ++i)
    {
        for (size_t j = 0; j < a[i].size(); ++j)
            sum[i] += a[i][j] * b[i][j];
    }
    
    return sum;
}
