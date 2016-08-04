#pragma once
#include <stdint.h>
#include "settings.hpp"
#include "activations.hpp"
#if USE_EIGEN == 1
#include <Eigen/Dense>
#endif

namespace nn
{
	enum class LayerType
	{
		kEigenRegular
	};

	template<LayerType type, ActivationType activationType, uint32_t unitsInLayer, uint32_t unitsInPreviousLayer>
	class layer { };

#if USE_EIGEN == 1
	template<ActivationType activationType, uint32_t unitsInLayer, uint32_t unitsInPreviousLayer>
	class layer<LayerType::kEigenRegular, activationType, unitsInLayer, unitsInPreviousLayer>
	{
	public:
		using WeightType = Eigen::Matrix<real, unitsInLayer, unitsInPreviousLayer>;
		using InputType = Eigen::Matrix<real, unitsInPreviousLayer, 1>;
		using OutputType = Eigen::Matrix<real, unitsInLayer, 1>;

		OutputType computeWeightedSum(const InputType& input) { return m_weight * input + m_bias; }

		OutputType computeActivations(const OutputType& output)
		{
			return output.unaryExpr(&activation<activationType>);
		}
	private:
		WeightType m_weight;
		OutputType m_bias;
	};
#endif
}