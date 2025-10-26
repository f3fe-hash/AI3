#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <cmath>
#include "math/vec_utils.hpp"

using data_pair = std::pair<vec<float>, vec<float>>;

struct dataset_config_t
{
    float lr;
    ushort num_batches;
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

inline dataset_t load_csv_dataset(const std::string& filename, bool has_header = false, char delimiter = ',')
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Failed to open dataset file: " + filename);

    std::string line;
    vec<vec<float>> X;
    vec<vec<float>> y;

    bool first_line = true;
    while (std::getline(file, line))
    {
        if (first_line && has_header)
        {
            first_line = false;
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        vec<float> row;

        while (std::getline(ss, cell, delimiter))
            row.push_back(std::stof(cell));

        if (row.empty())
            continue;

        // --- MNIST: label, 784 = pixels ---
        float label_val = row.front();
        row.erase(row.begin());

        // one-hot encode label
        vec<float> label(10, 0.0f);
        label[(int)label_val] = 1.0f;

        for (auto& n : row)
            n /= 255;

        X.push_back(row);
        y.push_back(label);
    }

    file.close();
    dataset_t dataset = create_dataset(X, y);
    return dataset;
}
