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
		kInput,
		kRegular,
		kSoftmax
	};

	// any layer has some amount of units and activation function
#if USE_EIGEN == 1
	class layer
	{
	public:
		using MatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;

		layer(LayerType type,
			uint32_t unitsInLayer,
			uint32_t unitsInPreviousLayer, 
			ActivationType activationType, 
			WeightInitializationType weightInitializationType) : 
			m_type(type),
			m_unitsInLayer(unitsInLayer), 
			m_unitsInPreviousLayer(unitsInPreviousLayer)
		{
			// set activation type
			if (activationType == ActivationType::kSigmoid)
			{
				m_activation = activation<ActivationType::kSigmoid>;
				m_activationDerivative = activation_derivative<ActivationType::kSigmoid>;
			}
			else if (activationType == ActivationType::kLinear)
			{
				m_activation = activation<ActivationType::kLinear>;
				m_activationDerivative = activation_derivative<ActivationType::kLinear>;
			}
			else if (activationType == ActivationType::kTanh)
			{
				m_activation = activation<ActivationType::kTanh>;
				m_activationDerivative = activation_derivative<ActivationType::kTanh>;
			}
			else if (activationType == ActivationType::kRelu)
			{
				m_activation = activation<ActivationType::kRelu>;
				m_activationDerivative = activation_derivative<ActivationType::kRelu>;
			}
			else if (activationType == ActivationType::kLRelu)
			{
				m_activation = activation<ActivationType::kLRelu>;
				m_activationDerivative = activation_derivative<ActivationType::kLRelu>;
			}

			if (type != LayerType::kInput)
			{
				if (weightInitializationType == WeightInitializationType::kGaussian)
				{
					m_weight = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer()).unaryExpr(weight_initalization<WeightInitializationType::kZeros>());
					m_nabla_w = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer());
					m_bias = MatrixType::Zero(UnitsInLayer(), 1).unaryExpr(weight_initalization<WeightInitializationType::kZeros>());
					m_nabla_b = MatrixType::Zero(UnitsInLayer(), 1);
				}
				else if (weightInitializationType == WeightInitializationType::kSequentialDebug)
				{
					m_weight = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer()).unaryExpr(weight_initalization<WeightInitializationType::kSequentialDebug>());
					m_nabla_w = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer());
					m_bias = MatrixType::Zero(UnitsInLayer(), 1).unaryExpr(weight_initalization<WeightInitializationType::kSequentialDebug>());
					m_nabla_b = MatrixType::Zero(UnitsInLayer(), 1);
				}
				else if (weightInitializationType == WeightInitializationType::kUniform)
				{
					m_weight = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer()).unaryExpr(weight_initalization<WeightInitializationType::kUniform>());
					m_nabla_w = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer());
					m_bias = MatrixType::Zero(UnitsInLayer(), 1).unaryExpr(weight_initalization<WeightInitializationType::kUniform>());
					m_nabla_b = MatrixType::Zero(UnitsInLayer(), 1);
				}
				else if (weightInitializationType == WeightInitializationType::kGaussian)
				{
					m_weight = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer()).unaryExpr(weight_initalization<WeightInitializationType::kGaussian>());
					m_nabla_w = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer());
					m_bias = MatrixType::Zero(UnitsInLayer(), 1).unaryExpr(weight_initalization<WeightInitializationType::kGaussian>());
					m_nabla_b = MatrixType::Zero(UnitsInLayer(), 1);
				}
				else if (weightInitializationType == WeightInitializationType::kWeightedGaussian)
				{
					m_weight = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer()).unaryExpr(weight_initalization<WeightInitializationType::kWeightedGaussian>(UnitsInLayer()));
					m_nabla_w = MatrixType::Zero(UnitsInLayer(), UnitsInPreviousLayer());
					m_bias = MatrixType::Zero(UnitsInLayer(), 1).unaryExpr(weight_initalization<WeightInitializationType::kWeightedGaussian>(UnitsInLayer()));
					m_nabla_b = MatrixType::Zero(UnitsInLayer(), 1);
				}
			}
		}

		~layer() {}

		void computeWeightedSum(const MatrixType& input) 
		{
			m_z.noalias() = m_weight * input;
			for (int i = 0; i < m_z.cols(); ++i)
				m_z.col(i).noalias() += m_bias;
		}
		void setActivations(const MatrixType& input) { m_a.noalias() = input; }
		void computeActivations(const MatrixType& input) 
		{ 
			if (m_type == LayerType::kRegular)
			{
				m_a.noalias() = input.unaryExpr(m_activation);
			}
			else if (m_type == LayerType::kSoftmax)
			{
				m_a = input;
				for (int i = 0; i < input.cols(); ++i)
				{
					m_a.col(i).noalias() = input.col(i).unaryExpr(&expf);
					m_a.col(i) /= m_a.col(i).sum();
				}
			}
		}
		void computeActivationDerivatives(const MatrixType& input) 
		{ 
			if (m_type == LayerType::kRegular)
			{
				m_da.noalias() = input.unaryExpr(m_activationDerivative);
			}
			else if (m_type == LayerType::kSoftmax)
			{
				m_da = m_a;
				auto softmax_derivative = [](real x) { return x * (1 - x); };
				m_da = m_da.unaryExpr(softmax_derivative);
			}
		}

		const MatrixType& getWeightedSum() const { return m_z; }
		const MatrixType& getActivations() const { return m_a; }
		const MatrixType& getActivationDerivatives() const { return m_da; }
		const MatrixType& getWeights() const { return m_weight; }
		MatrixType& getWeights() { return m_weight; }
		MatrixType& getBias() { return m_bias; }
		MatrixType& getNablaB() { return m_nabla_b; }
		MatrixType& getNablaW() { return m_nabla_w; }

		uint32_t UnitsInLayer() const { return m_unitsInLayer; }
		uint32_t UnitsInPreviousLayer() const { return m_unitsInPreviousLayer; }
	private:
		LayerType m_type;
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