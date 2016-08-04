#include <iostream>

#include "cost.hpp"
#include "layer.hpp"
#include "activations.hpp"

int main()
{
	using namespace nn;
	layer<LayerType::kEigenRegular, ActivationType::kSigmoid, 30, 256> l1;

}