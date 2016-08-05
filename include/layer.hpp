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

	// any layer has some amount of units and activation function
#if USE_EIGEN == 1
	class layerBase
	{
	public:
		using MatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
		layerBase(uint32_t unitsInLayer) : m_unitsInLayer(unitsInLayer) {}
		uint32_t UnitsInLayer() const { return m_unitsInLayer; }
		template<ActivationType activationType>
		void setActivationType()
		{
			m_activation = activation<activationType>;
			m_activationDerivative = activation_derivative<activationType>;
		}

		virtual void computeWeightedSum(const MatrixType& input) = 0;
		virtual void computeActivations(const MatrixType& input) { m_a = input.unaryExpr(m_activation); }
		virtual void computeActivationDerivatives(const MatrixType& input) { m_da = input.unaryExpr(m_activationDerivative); }
		const MatrixType& getWeightedSum() const { return m_z; }
		const MatrixType& getActivations() const { return m_a; }
		const MatrixType& getActivationDerivatives() const { return m_da; }
	protected:
		MatrixType m_a;
		MatrixType m_z;
		MatrixType m_da;

		uint32_t m_unitsInLayer;
		ActivationFunction m_activation = nullptr;
		ActivationFunction m_activationDerivative = nullptr;
	};

	template<LayerType type> class layer : public layerBase { };

	template<>
	class layer<LayerType::kEigenInput> : public layerBase
	{
	public:
		layer(uint32_t unitsInLayer) : layerBase(unitsInLayer) {}
		void computeWeightedSum(const MatrixType& input) { m_z = input; }
	};

	template<>
	class layer<LayerType::kEigenRegular> : public layerBase
	{
	public:
		using MatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;

		layer(uint32_t unitsInLayer, uint32_t unitsInPreviousLayer) : layerBase(unitsInLayer), m_unitsInPreviousLayer(unitsInPreviousLayer) {}

		template<WeightInitializationType weightInitializationType>
		void initializeWeights()
		{
			m_weight = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer()).unaryExpr(weight_initalization<weightInitializationType>());
			m_bias = MatrixType::Zero(UnitsInLayer(), 1).unaryExpr(weight_initalization<weightInitializationType>());
		}

		void computeWeightedSum(const MatrixType& input) { m_z = m_weight * input + m_bias; }

		uint32_t UnitsInPreviousLayer() const { return m_unitsInPreviousLayer; }
	private:
		uint32_t m_unitsInPreviousLayer;

		MatrixType m_weight;
		MatrixType m_bias;
	};
#endif
}