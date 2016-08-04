#pragma once

#include "settings.hpp"

namespace nn
{
	template<typename T>
	real mse(const T& y, const T& z)
	{
		real cost = 0.0;
		for (int i = 0; i < x.size(); ++i)
		{
			auto d = x - y;
			cost += d * d;
		}
		return cost;
	}
}