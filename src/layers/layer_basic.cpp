#include "layers/basic_layer.hpp"

basic_layer::~basic_layer() = default;

void basic_layer::init(size_t prev_size)
{
	this->prev_size = prev_size;
}
