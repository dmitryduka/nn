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
		virtual void computeActivations(const MatrixType& input) = 0;
		virtual void computeActivationDerivatives(const MatrixType& input) = 0;

		const MatrixType& getWeightedSum() const { return m_z; }
		virtual const MatrixType& getActivations() const { return m_a; }
		virtual const MatrixType& getActivationDerivatives() const { return m_da; }

		virtual MatrixType& getWeights() { return m_dummy; }
		virtual MatrixType& getBias() { return m_dummy; }
		virtual MatrixType& getNablaB() { return m_dummy; }
		virtual MatrixType& getNablaW() { return m_dummy; }
	protected:
		MatrixType m_dummy;
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
		void computeWeightedSum(const MatrixType& input) { }
		void computeActivations(const MatrixType& input) { m_a = input; }
		void computeActivationDerivatives(const MatrixType& input) { }
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
			m_nabla_w = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer());
			m_bias = MatrixType::Zero(UnitsInLayer(), 1).unaryExpr(weight_initalization<weightInitializationType>());
			m_nabla_b = MatrixType::Zero(UnitsInLayer(), 1);
		}

		void computeWeightedSum(const MatrixType& input) { m_z = m_weight * input + m_bias; }
		void computeActivations(const MatrixType& input) { m_a = input.unaryExpr(m_activation); }
		void computeActivationDerivatives(const MatrixType& input) { m_da = input.unaryExpr(m_activationDerivative); }
		MatrixType& getWeights() { return m_weight; }
		MatrixType& getBias() { return m_bias; }
		MatrixType& getNablaB() { return m_nabla_b; }
		MatrixType& getNablaW() { return m_nabla_w; }
		uint32_t UnitsInPreviousLayer() const { return m_unitsInPreviousLayer; }
	private:
		uint32_t m_unitsInPreviousLayer;
		MatrixType m_nabla_b;
		MatrixType m_nabla_w;
		MatrixType m_weight;
		MatrixType m_bias;
	};
#endif
}