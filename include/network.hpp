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
		template<ActivationType activationType>
		std::shared_ptr<layerBase> addInputLayer(uint32_t units)
		{
			if (!m_layers.empty())
				throw std::logic_error("Only one input layer supported");
			auto curLayer = std::make_shared<layer<LayerType::kEigenInput>>(units);
			curLayer->setActivationType<activationType>();
			m_layers.push_back(curLayer);
			return curLayer;
		}

		template<ActivationType activationType, WeightInitializationType weightInitializationType>
		std::shared_ptr<layerBase> addRegularLayer(uint32_t units)
		{
			if (m_layers.empty())
				throw std::logic_error("There should be at least one layer in the network before adding regular layer (impossible to deduce matrix dimensions)");
			const auto prevLayer = m_layers.back();
			auto curLayer = std::make_shared<layer<LayerType::kEigenRegular>>(units, prevLayer->UnitsInLayer());
			curLayer->setActivationType<activationType>();
			curLayer->initializeWeights<weightInitializationType>();
			m_layers.push_back(curLayer);
			return curLayer;
		}

		std::shared_ptr<layerBase> getLayer(size_t layerNumber) { return m_layers[layerNumber]; }

		layerBase::MatrixType feedforward(const layerBase::MatrixType& input)
		{
			m_layers[0]->computeWeightedSum(input);
			m_layers[0]->computeActivations(input);
			for (size_t i = 1; i < m_layers.size(); ++i)
			{
				auto l = m_layers[i];
				l->computeWeightedSum(m_layers[i - 1]->getActivations());
				l->computeActivations(l->getWeightedSum());
				l->computeActivationDerivatives(l->getWeightedSum());
			}
			return m_layers[m_layers.size() - 1]->getActivations();
		}

		real evaluate(const std::vector<MatrixType>& inputs, const std::vector<uint8_t>& labels)
		{
			if (inputs.size() != labels.size())
				throw std::logic_error("Inputs and labels should be of the same size");

			size_t correct = 0;
			std::vector<uint8_t> outputs;
			for (size_t i = 0; i < inputs.size(); ++i)
			{
				const auto output = feedforward(inputs[i]);
				uint8_t idx = 0;
				output.col(0).maxCoeff(&idx);
				outputs.push_back(idx);
				correct += idx == labels[i];
			}
			return real(correct) / real(inputs.size());
		}
	private:
		std::vector<std::shared_ptr<layerBase>> m_layers;
	};
}