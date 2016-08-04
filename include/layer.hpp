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
		kEigenInput,
		kEigenRegular
	};

	template<LayerType type, ActivationType activationType, uint32_t unitsInLayer, uint32_t unitsInPreviousLayer = 1>
	class layer { };

#if USE_EIGEN == 1
	template<ActivationType activationType, uint32_t unitsInLayer>
	class layer<LayerType::kEigenInput, activationType, unitsInLayer, 1>
	{
	public:
		using MatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
		static const uint32_t UnitsInLayer = unitsInLayer;
		static MatrixType computeActivations(const MatrixType& output) { return output.unaryExpr(&activation<activationType>); }
	};

	template<ActivationType activationType, uint32_t unitsInLayer, uint32_t unitsInPreviousLayer>
	class layer<LayerType::kEigenRegular, activationType, unitsInLayer, unitsInPreviousLayer>
	{
	public:
		using MatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
		static const uint32_t UnitsInLayer = unitsInLayer;
		static const uint32_t UnitsInPreviousLayer = unitsInPreviousLayer;

		layer()
		{
			m_weight = MatrixType::Random(unitsInLayer, unitsInPreviousLayer);
			m_bias = MatrixType::Random(unitsInLayer, 1);
		}

		void computeWeightedSum(const MatrixType& input) { m_z = m_weight * input + m_bias; }
		void computeActivations() { m_a = m_z.unaryExpr(&activation<activationType>); }

		const MatrixType& getWeightedSum() const { return m_z; }
		const MatrixType& getOutput() const { return m_a; }
	private:
		MatrixType m_z;
		MatrixType m_a;
		MatrixType m_weight;
		MatrixType m_bias;
	};
#endif
}