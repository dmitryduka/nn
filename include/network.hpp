#pragma once
#include <vector>
#include <memory>
#include "cost.hpp"
#include "layer.hpp"
#include "activations.hpp"

namespace nn
{
	struct evaluate_results
	{
		real accuracy;
		real cost;
		std::vector<size_t> errors;
	};

	class network
	{
	public:
		using Layer = layer;

		Layer& addRegularLayer(LayerType type, uint32_t units, ActivationType activationType, WeightInitializationType weightInitializationType)
		{
			uint32_t unitsInPreviousLayer = 0;
			if (!m_layers.empty())
				unitsInPreviousLayer = m_layers.back().UnitsInLayer();
			auto curLayer = layer(type, units, unitsInPreviousLayer, activationType, weightInitializationType);
			m_layers.push_back(curLayer);
			return m_layers.back();
		}

		void setCostFunction(CostType type)
		{
			if (type == CostType::kQuadratic)
			{
				m_cost = cost<CostType::kQuadratic>;
				m_cost_derivative = cost_derivative<CostType::kQuadratic>;
			}
			else if (type == CostType::kCrossEntropy)
			{
				m_cost = cost<CostType::kCrossEntropy>;
				m_cost_derivative = cost_derivative<CostType::kCrossEntropy>;
			}
		}

		// singlethread version
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

		// multithread version
		void feedforward(const Layer::MatrixType& input, 
			std::vector<Layer::MatrixType>& activations, 
			std::vector<Layer::MatrixType>& activationDerivatives)
		{
			MatrixType in = input;
			activations.push_back(in);
			activationDerivatives.push_back(MatrixType());
			for (size_t i = 1; i < m_layers.size(); ++i)
			{
				auto& l = m_layers[i];
				in = l.computeWeightedSumExplicit(in);
				in = l.computeActivationsExplicit(in);
				activations.push_back(in);
				activationDerivatives.push_back(l.computeActivationDerivativesExplicit(in));
			}
		}

		// singlethread version
		void backprop(const std::vector<uint8_t>& label_batch)
		{
			// make one-hot label out of single uint8_t
			auto& outputLayer = m_layers.back();
			MatrixType labelOneHot = MatrixType::Zero(outputLayer.UnitsInLayer(), outputLayer.getActivations().cols());
			for (int i = 0; i < labelOneHot.cols(); ++i)
				labelOneHot(label_batch[i], i) = real(1.0);
			// compute delta
			MatrixType delta = m_cost_derivative(outputLayer.getActivations(), labelOneHot).array() * outputLayer.getActivationDerivatives().array();
			if (outputLayer.getNablaB().cols() != delta.cols())
				outputLayer.getNablaB() = MatrixType::Zero(delta.rows(), delta.cols());
			outputLayer.getNablaB().noalias() += delta;
			outputLayer.getNablaW().noalias() += delta * m_layers[m_layers.size() - 2].getActivations().transpose();

			for (size_t i = m_layers.size() - 2; i > 0; --i)
			{
				const auto& nextLayer = m_layers[i + 1];
				auto& layer = m_layers[i];
				const auto& prevLayer = m_layers[i - 1];
				delta = nextLayer.getWeights().transpose() * delta;
				delta = delta.array() * layer.getActivationDerivatives().array();
				// resize nabla-b to match the batch size
				if (layer.getNablaB().cols() != delta.cols())
					layer.getNablaB() = MatrixType::Zero(delta.rows(), delta.cols());
				layer.getNablaB() += delta;
				layer.getNablaW() += delta * prevLayer.getActivations().transpose();
			}
		}

		// multithread version
		void backprop(const std::vector<uint8_t>& label_batch,
			const std::vector<Layer::MatrixType>& activations,
			const std::vector<Layer::MatrixType>& activationDerivatives,
			std::vector<Layer::MatrixType>& nabla_w,
			std::vector<Layer::MatrixType>& nabla_b)
		{
			// make one-hot label out of single uint8_t
			auto& outputLayer = m_layers.back();
			MatrixType labelOneHot = MatrixType::Zero(outputLayer.UnitsInLayer(), activations.back().cols());
			for (int i = 0; i < labelOneHot.cols(); ++i)
				labelOneHot(label_batch[i], i) = real(1.0);
			// compute delta
			MatrixType delta = m_cost_derivative(activations.back(), labelOneHot).array() * activationDerivatives.back().array();
			nabla_b.push_back(delta);
			nabla_w.push_back(delta * activations[activations.size() - 2].transpose());

			for (size_t i = m_layers.size() - 2; i > 0; --i)
			{
				const auto& nextLayer = m_layers[i + 1];
				auto& layer = m_layers[i];
				const auto& prevLayer = m_layers[i - 1];
				delta = nextLayer.getWeights().transpose() * delta;
				delta = delta.array() * activationDerivatives[i].array();
				nabla_b.push_back(delta);
				nabla_w.push_back(delta * activations[i - 1].transpose());
			}
		}

		void derive_backprop(uint8_t image_label, Layer::MatrixType& image_grad)
		{
			// make one-hot label out of single uint8_t
			auto& outputLayer = m_layers.back();
			MatrixType labelOneHot = MatrixType::Zero(outputLayer.UnitsInLayer(), 1);
			labelOneHot(image_label, 0) = real(1.0);
			// compute delta
			MatrixType input = MatrixType::Zero(28 * 28, 1);
			for (int i = 0; i < 1; ++i)
			{
				MatrixType output = feedforward(input);
				MatrixType delta = labelOneHot - output;

				for (size_t i = m_layers.size() - 1; i > 0; --i)
				{
					auto& prevLayer = m_layers[i - 1];
					auto& layer = m_layers[i];
					auto& arr = delta.array();
					delta = layer.getWeights().transpose() * delta;
				}
				input += delta;
			}
			image_grad = input;
		}

		// singlethread version
		void update_weights(real eta, real lambda, uint32_t batch_size)
		{
			for (size_t i = m_layers.size() - 1; i > 0; --i)
			{
				auto& layer = m_layers[i];
				// regularization
				if (lambda != real(0.0))
					layer.getWeights() *= (1.0 - eta * lambda / real(batch_size));
				layer.getWeights() -= (eta / real(batch_size)) * layer.getNablaW();
				for (int i = 0; i < layer.getNablaB().cols(); ++i)
					layer.getBias() -= (eta / real(batch_size)) * layer.getNablaB().col(i);
				layer.getNablaW().setConstant(0.0);
				layer.getNablaB().setConstant(0.0);
			}
		}

		// multithread version
		void update_weights(real eta, real lambda, uint32_t batch_size,
			const std::vector<Layer::MatrixType>& nabla_w,
			const std::vector<Layer::MatrixType>& nabla_b)
		{
			for (size_t i = m_layers.size() - 1; i > 0; --i)
			{
				auto& layer = m_layers[i];
				// regularization
				if (lambda != real(0.0))
					layer.getWeights() *= (1.0 - eta * lambda / real(batch_size));
				layer.getWeights() -= (eta / real(batch_size)) * nabla_w[nabla_w.size() - i];
				for (int k = 0; k < nabla_b[nabla_b.size() - i].cols(); ++k)
					layer.getBias() -= (eta / real(batch_size)) * nabla_b[nabla_b.size() - i].col(k);
			}
		}

		void sgd(uint32_t img_width, uint32_t img_height,
			uint32_t batches, uint32_t batch_size,
			real eta, real lambda,
			const std::vector<MatrixType>& training_set,
			const std::vector<uint8_t>& training_labels)
		{
			MatrixType image_batch = MatrixType::Zero(img_width * img_height, batch_size);
			std::vector<uint8_t> label_batch(batch_size);
			for (size_t k = 0u; k < batches; k++)
			{
				const size_t batch_start = k * batch_size;
				const size_t batch_end = (k + 1) * batch_size;

				for (size_t i = batch_start; i < batch_end; ++i)
				{
					image_batch.col(i - batch_start) = training_set[i];
					label_batch[i - batch_start] = training_labels[i];
				}
				feedforward(image_batch);
				backprop(label_batch);
				update_weights(eta, lambda, batch_size);
			}
		}

		void psgd(uint32_t img_width, uint32_t img_height,
			uint32_t batches, uint32_t batch_size,
			real eta, real lambda,
			const std::vector<MatrixType>& training_set,
			const std::vector<uint8_t>& training_labels,
			bool useLock)
		{
			std::mutex weights_mutex;
			const auto worker_count = std::thread::hardware_concurrency();
			auto sgd_thread_func = [&](uint32_t threadNo)
			{
				MatrixType image_batch = MatrixType::Zero(img_width * img_height, batch_size);
				std::vector<uint8_t> label_batch(batch_size);
				const auto batches_per_thread = batches / worker_count;
				const auto batches_start = threadNo * batches_per_thread;
				const auto batches_end = std::min(batches_start + batches_per_thread, batches);
				for (size_t k = batches_start; k < batches_end; k++)
				{
					const size_t batch_start = k * batch_size;
					const size_t batch_end = (k + 1) * batch_size;

					for (size_t i = batch_start; i < batch_end; ++i)
					{
						image_batch.col(i - batch_start) = training_set[i];
						label_batch[i - batch_start] = training_labels[i];
					}
					std::vector<MatrixType> activations, activationDerivatives, nablaW, nablaB;
					feedforward(image_batch, activations, activationDerivatives);
					backprop(label_batch, activations, activationDerivatives, nablaW, nablaB);
					if (useLock)
					{
						std::lock_guard<std::mutex> lock(weights_mutex);
						update_weights(eta, lambda, batch_size, nablaW, nablaB);
					}
					else
					{
						update_weights(eta, lambda, batch_size, nablaW, nablaB);
					}
				}
			};

			std::vector<std::thread> workers;

			for (size_t i = 0u; i < worker_count; ++i)
				workers.push_back(std::thread(sgd_thread_func, i));
			for (size_t i = 0u; i < worker_count; ++i)
				workers[i].join();	
		}


		evaluate_results evaluate(const std::vector<MatrixType>& inputs, const std::vector<uint8_t>& labels, size_t count = 0)
		{
			if (inputs.size() != labels.size())
				throw std::logic_error("Inputs and labels should be of the same size");

			size_t correct = 0;
			real cost = 0.0;
			std::vector<uint8_t> outputs;
			std::vector<size_t> errors;
			const auto range = count != 0 ? count : inputs.size();
			for (size_t i = 0; i < range; ++i)
			{
				const auto output = feedforward(inputs[i]);
				uint8_t idx = 0;
				output.col(0).maxCoeff(&idx);
				outputs.push_back(idx);
				const auto& outputLayer = m_layers.back();
				MatrixType labelOneHot = MatrixType::Zero(outputLayer.UnitsInLayer(), 1);
				labelOneHot(labels[i], 0) = real(1.0);
				cost += m_cost(output, labelOneHot);
				correct += idx == labels[i];
				if (idx != labels[i])
					errors.push_back(i);
			}

			return evaluate_results{ real(correct) / real(range), cost / real(range), errors };
		}
	public:
		CostFunction m_cost;
		CostDerivativeFunction m_cost_derivative;
		std::vector<Layer> m_layers;
	};
}