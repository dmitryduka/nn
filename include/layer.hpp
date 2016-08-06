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
		kEigenRegular
	};

	// any layer has some amount of units and activation function
#if USE_EIGEN == 1
	template<LayerType type> class layer;

	template<>
	class layer<LayerType::kEigenRegular>
	{
	public:
		using MatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;

		layer(uint32_t unitsInLayer, uint32_t unitsInPreviousLayer) : m_unitsInLayer(unitsInLayer), m_unitsInPreviousLayer(unitsInPreviousLayer) {}
		template<ActivationType activationType>
		void setActivationType()
		{
			m_activation = activation<activationType>;
			m_activationDerivative = activation_derivative<activationType>;
		}
		template<WeightInitializationType weightInitializationType>
		void initializeWeights()
		{
			m_weight = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer()).unaryExpr(weight_initalization<weightInitializationType>());
			m_nabla_w = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer());
			m_bias = MatrixType::Zero(UnitsInLayer(), 1).unaryExpr(weight_initalization<weightInitializationType>());
			m_nabla_b = MatrixType::Zero(UnitsInLayer(), 1);
		}

		template<>
		void initializeWeights<WeightInitializationType::kWeightedGaussian>()
		{
			m_weight = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer()).unaryExpr(weight_initalization<WeightInitializationType::kWeightedGaussian>(UnitsInLayer()));
			m_nabla_w = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer());
			m_bias = MatrixType::Zero(UnitsInLayer(), 1).unaryExpr(weight_initalization<WeightInitializationType::kWeightedGaussian>(UnitsInLayer()));
			m_nabla_b = MatrixType::Zero(UnitsInLayer(), 10);
		}

		void computeWeightedSum(const MatrixType& input) 
		{
			m_z = m_weight * input;
			for (int i = 0; i < m_z.cols(); ++i)
				m_z.col(i) += m_bias;
		}
		void setActivations(const MatrixType& input) { m_a = input; }
		void computeActivations(const MatrixType& input) { m_a = input.unaryExpr(m_activation); }
		void computeActivationDerivatives(const MatrixType& input) { m_da = input.unaryExpr(m_activationDerivative); }

		const MatrixType& getWeightedSum() const { return m_z; }
		const MatrixType& getActivations() const { return m_a; }
		const MatrixType& getActivationDerivatives() const { return m_da; }
		MatrixType& getWeights() { return m_weight; }
		MatrixType& getBias() { return m_bias; }
		MatrixType& getNablaB() { return m_nabla_b; }
		MatrixType& getNablaW() { return m_nabla_w; }

		uint32_t UnitsInLayer() const { return m_unitsInLayer; }
		uint32_t UnitsInPreviousLayer() const { return m_unitsInPreviousLayer; }
	private:
		uint32_t m_unitsInLayer;
		uint32_t m_unitsInPreviousLayer;
		MatrixType m_z;
		MatrixType m_a;
		MatrixType m_da;

		MatrixType m_weight;
		MatrixType m_nabla_w;
		MatrixType m_bias;
		MatrixType m_nabla_b;
		ActivationFunction m_activation = nullptr;
		ActivationFunction m_activationDerivative = nullptr;
	};
#endif
}