#pragma once
#include <stdint.h>
#include "settings.hpp"
#include "weight_initialization.hpp"
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

	template<LayerType type, 
		ActivationType activationType, 
		WeightInitializationType weightInitializationType, 
		uint32_t unitsInLayer, uint32_t unitsInPreviousLayer = 1>
	class layer { };

#if USE_EIGEN == 1
	template<ActivationType activationType, WeightInitializationType weightInitializationType, uint32_t unitsInLayer>
	class layer<LayerType::kEigenInput, activationType, weightInitializationType, unitsInLayer, 1>
	{
	public:
		using MatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
		static const uint32_t UnitsInLayer = unitsInLayer;
		static MatrixType computeActivations(const MatrixType& output) { return output.unaryExpr(&activation<activationType>); }
	};

	template<ActivationType activationType, WeightInitializationType weightInitializationType, uint32_t unitsInLayer, uint32_t unitsInPreviousLayer>
	class layer<LayerType::kEigenRegular, activationType, weightInitializationType, unitsInLayer, unitsInPreviousLayer>
	{
	public:
		using MatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
		static const uint32_t UnitsInLayer = unitsInLayer;
		static const uint32_t UnitsInPreviousLayer = unitsInPreviousLayer;

		layer()
		{
			// gaussian weight initialization
			m_weight = MatrixType::Zero(unitsInLayer, unitsInPreviousLayer).unaryExpr(weight_initalization<weightInitializationType>());
			m_bias = MatrixType::Zero(unitsInLayer, 1).unaryExpr(weight_initalization<weightInitializationType>());
		}

		void computeWeightedSum(const MatrixType& input) { m_z = m_weight * input + m_bias; }
		void computeActivations() { m_a = m_z.unaryExpr(&activation<activationType>); }
		void computeActivationDerivatives() { m_da = m_z.unaryExpr(&activation_derivative<activationType>); }

		const MatrixType& getWeightedSum() const { return m_z; }
		const MatrixType& getOutput() const { return m_a; }
	private:
		MatrixType m_z;
		MatrixType m_a;
		MatrixType m_da;
		MatrixType m_weight;
		MatrixType m_bias;
	};
#endif
}