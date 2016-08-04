#pragma once
#include <algorithm>
#include <math.h>

#include "settings.hpp"

#pragma warning( disable : 4244 ) // type casting warning

namespace nn
{
	enum class ActivationType
	{
		kSigmoid,
		kRelu,
		kTanh
	};

	template<ActivationType t> real activation(real in);
	template<ActivationType t> real activation_derivative(real in);

	// sigmoid
	template<>
	real activation<ActivationType::kSigmoid>(real in) { return 1.0 / (1.0 + exp(-in)); }
	template<>
	real activation_derivative<ActivationType::kSigmoid>(real in)
	{
		const real x = activation<ActivationType::kSigmoid>(in); 
		return x * (1.0 - x);
	}

	// RELU
	template<>
	real activation<ActivationType::kRelu>(real in) { return std::max(in, real(0.0)); }
	template<>
	real activation_derivative<ActivationType::kRelu>(real in) { return in > 0.0 ? 1.0 : 0.0; }

	// tanh
	template<>
	real activation<ActivationType::kTanh>(real in) { return (1.0 - exp(-2.0 * in)) / (1.0 + exp(-2.0 * in)); }
	template<>
	real activation_derivative<ActivationType::kTanh>(real in)
	{
		const real x = activation<ActivationType::kTanh>(in);
		return 1.0 - x * x;
	}
}