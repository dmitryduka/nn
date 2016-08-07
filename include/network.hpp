#pragma once
#include <vector>
#include <memory>
#include "cost.hpp"
#include "layer.hpp"
#include "activations.hpp"

namespace nn
{
	class network
	{
	public:
		using Layer = layer<LayerType::kEigenRegular>;

		template<ActivationType activationType, WeightInitializationType weightInitializationType>
		Layer& addRegularLayer(uint32_t units)
		{
			uint32_t unitsInPreviousLayer = 0;
			if (!m_layers.empty())
				unitsInPreviousLayer = m_layers.back().UnitsInLayer();
			auto curLayer = layer<LayerType::kEigenRegular>(units, unitsInPreviousLayer);
			curLayer.setActivationType<activationType>();
			if (!m_layers.empty())
				curLayer.initializeWeights<weightInitializationType>();
			m_layers.push_back(curLayer);
			return m_layers.back();
		}

		Layer::MatrixType feedforward(const Layer::MatrixType& input)
		{
			m_layers[0].setActivations(input);
			for (size_t i = 1; i < m_layers.size(); ++i)
			{
				auto& l = m_layers[i];
				l.computeWeightedSum(m_layers[i - 1].getActivations());
				l.computeActivations(l.getWeightedSum());
				l.computeActivationDerivatives(l.getWeightedSum());
			}
			return m_layers[m_layers.size() - 1].getActivations();
		}

		real evaluate(const std::vector<MatrixType>& inputs, const std::vector<uint8_t>& labels, size_t count = 0)
		{
			if (inputs.size() != labels.size())
				throw std::logic_error("Inputs and labels should be of the same size");

			size_t correct = 0;
			std::vector<uint8_t> outputs;
			const auto range = count != 0 ? count : inputs.size();
			for (size_t i = 0; i < range; ++i)
			{
				const auto output = feedforward(inputs[i]);
				uint8_t idx = 0;
				output.col(0).maxCoeff(&idx);
				outputs.push_back(idx);
				correct += idx == labels[i];
			}
			return real(correct) / real(range);
		}

		void backprop(const std::vector<uint8_t>& label_batch)
		{
			// make one-hot label out of single uint8_t
			auto& outputLayer = m_layers.back();
			MatrixType labelOneHot = MatrixType::Zero(outputLayer.UnitsInLayer(), outputLayer.getActivations().cols());
			for (int i = 0; i < labelOneHot.cols(); ++i)
				labelOneHot(label_batch[i], i) = real(1.0);
			// compute delta
			MatrixType delta = mse_derivative(outputLayer.getActivations(), labelOneHot).array() * outputLayer.getActivationDerivatives().array();
			outputLayer.getNablaB() += delta;
			outputLayer.getNablaW() += delta * m_layers[m_layers.size() - 2].getActivations().transpose();

			for (size_t i = m_layers.size() - 2; i > 0; --i)
			{
				const auto& nextLayer = m_layers[i + 1];
				auto& layer = m_layers[i];
				const auto& prevLayer = m_layers[i - 1];
				auto w = nextLayer.getWeights().transpose();
				delta = (w * delta);
				delta = delta.array() * layer.getActivationDerivatives().array();
				// resize nabla-b to match the batch size
				if (layer.getNablaB().cols() != delta.cols())
					layer.getNablaB() = MatrixType::Zero(delta.rows(), delta.cols());
				layer.getNablaB() += delta;
				layer.getNablaW() += delta * prevLayer.getActivations().transpose();
			}
		}

		void update_weights(real eta, uint32_t batch_size)
		{
			for (size_t i = m_layers.size() - 1; i > 0; --i)
			{
				auto& layer = m_layers[i];
				layer.getWeights() -= (eta / real(batch_size)) * layer.getNablaW();
				for (int i = 0; i < layer.getNablaB().cols(); ++i)
					layer.getBias() -= (eta / real(batch_size)) * layer.getNablaB().col(i);
				layer.getNablaW().setConstant(0.0);
				layer.getNablaB().setConstant(0.0);
			}
		}
	private:
		std::vector<Layer> m_layers;
	};
}