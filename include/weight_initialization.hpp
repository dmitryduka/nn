#pragma once
#include <random>
#include "settings.hpp"

namespace nn
{
	enum class WeightInitializationType
	{
		kZeros,
		kGaussian,
		kWeightedGaussian,
		kUniform,
		kSequentialDebug
	};

	template<WeightInitializationType type> struct weight_initalization {};
	template<>
	struct weight_initalization<WeightInitializationType::kGaussian>
	{
		real operator()(real) const
		{
			static std::mt19937 rng;
			static std::normal_distribution<real> nd(real(0.0), real(1.0));
			return nd(rng);
		}
	};

	template<>
	struct weight_initalization<WeightInitializationType::kWeightedGaussian>
	{
		real m_n;
		weight_initalization(real n) : m_n(n) {}
		real operator()(real) const
		{
			static std::mt19937 rng;
			static std::normal_distribution<real> nd(real(0.0), real(1.0 / sqrt(m_n)));
			return nd(rng);
		}
	};

	template<>
	struct weight_initalization<WeightInitializationType::kUniform>
	{
		real operator()(real) const
		{
			static std::mt19937 rng;
			static std::uniform_real_distribution<real> nd(real(0.0), real(1.0));
			return nd(rng);
		}
	};
}
