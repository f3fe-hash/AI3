#pragma once

#include <vector>
#include <memory>
#include <random>
#include <cstdint>

#include "math/vec_utils.hpp"
#include "math/dataset.hpp"

class basic_layer
{
protected:
    size_t size;
    size_t prev_size;

    std::shared_ptr<std::mt19937> gen;
    vec<float> last_input;

public:
    basic_layer(size_t size) : size(size), prev_size(0)
    {}

    virtual ~basic_layer();

    inline size_t get_size()
        { return size; }
    
    inline void set_gen(std::shared_ptr<std::mt19937> gen)
        { this->gen = gen; }

    virtual void init(size_t prev_size);
    virtual vec<float> forward(const vec<float>& in) = 0;
    virtual vec<float> backprop(const vec<float>& grads, dataset_config_t config) = 0;
};
