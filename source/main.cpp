#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <tuple>
#include <map>

#include "mnist.hpp"
#include "timing.hpp"
#include "network.hpp"
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif
#include "python/py_plot.h"

namespace
{
	struct PythonWrapper
	{
		std::mutex m_pythonLock;
		PythonWrapper()
		{
			Py_Initialize();
			initplot();
		}
		~PythonWrapper() 
		{ 
			Py_Finalize(); 
		}

		template<typename ... T>
		void plot(T... t)
		{
			std::lock_guard<std::mutex> lock(m_pythonLock);
			::plot(t...);
		}
		template<typename ... T>
		void save_plot(T... t)
		{
			std::lock_guard<std::mutex> lock(m_pythonLock);
			::save_plot(t...);
		}
	};

	template<typename T>
	std::string to_string(const T& in, uint32_t width = 0)
	{
		std::stringstream out;
		if (width)
			out << std::setfill('0') << std::setw(width);
		out << in;
		return out.str();
	}
}

PythonWrapper g_PythonWrapper;

int main()
{
	using namespace nn;
	const uint32_t epochs = 60;
	const uint32_t dataset_size = 60000;
	const uint32_t training_set_size = dataset_size * 0.9;
	std::vector<MatrixType> training_set, validation_set;
	std::vector<uint8_t> training_labels, validation_labels;
	{
		const auto images = loadMNISTImages("externals/mnist/train-images.idx3-ubyte", dataset_size);
		const auto labels = loadMNISTLabels("externals/mnist/train-labels.idx1-ubyte", dataset_size);
		training_set = std::vector<MatrixType>(images.cbegin(), images.cbegin() + training_set_size);
		validation_set = std::vector<MatrixType>(images.cbegin() + training_set_size, images.cend());
		training_labels = std::vector<uint8_t>(labels.cbegin(), labels.cbegin() + training_set_size);
		validation_labels = std::vector<uint8_t>(labels.cbegin() + training_set_size, labels.cend());
	}
	network original_net;
	original_net.addRegularLayer(LayerType::kInput, 28 * 28, ActivationType::kNone, WeightInitializationType::kWeightedGaussian);
	original_net.addRegularLayer(LayerType::kRegular, 64, ActivationType::kLRelu, WeightInitializationType::kWeightedGaussian);
	original_net.addRegularLayer(LayerType::kRegular, 64, ActivationType::kLRelu, WeightInitializationType::kWeightedGaussian);
	original_net.addRegularLayer(LayerType::kSoftmax, 10, ActivationType::kNone, WeightInitializationType::kWeightedGaussian);
	original_net.setCostFunction(CostType::kCrossEntropy);
	// RELU - tops at ~84%, eta = 0.05, batch_size = 30, 100 (seems to not matter much)
	// Sigmoid - tops at ~98%, eta = 2.0, 1.0, 0.5 (<0.96,<0.97,<1.0), batch_size = 10
	auto trainNet = [&](uint32_t netNo, real eta, real lambda, uint32_t batch_size)
	{
		std::vector<float> graph_epoch, graph_acc;
		network net = original_net;
		// SGD
		timing timer;
		const size_t batches = training_set.size() / batch_size;
		MatrixType image_batch = MatrixType::Zero(28 * 28, batch_size);
		std::vector<uint8_t> label_batch(batch_size);
		evaluate_results result;
		for (size_t epoch = 0u; epoch < epochs; ++epoch)
		{
			for (size_t k = 0u; k < batches; k++)
			{
				const size_t batch_start = k * batch_size;
				const size_t batch_end = (k + 1) * batch_size;

				for (size_t i = batch_start; i < batch_end; ++i)
				{
					image_batch.col(i - batch_start) = training_set[i];
					label_batch[i - batch_start] = training_labels[i];
				}
				net.feedforward(image_batch);
				net.backprop(label_batch);
				net.update_weights(eta, lambda, batch_size);
			}
			result = net.evaluate(validation_set, validation_labels);
			graph_epoch.push_back(epoch);
			graph_acc.push_back(result.accuracy);
			std::cout << "Epoch " << epoch << ", acc: " << result.accuracy * 100.0f << "%, cost = " << result.cost << " (" << timer.seconds() << " seconds passed)" << std::endl;
		}

		const bool dump_error_images = false;
		if (dump_error_images)
		{
			for (auto i : result.errors)
			{
				std::vector<float> img(validation_set[i].data(), validation_set[i].data() + 28 * 28);
				plot_image(img, "error" + to_string(i) + ".png");
			}
		}

		std::string plotLabel = "eta=" + to_string(eta) + ",lambda=" + to_string(lambda) + ", bs=" + to_string(batch_size);
		g_PythonWrapper.plot(netNo, graph_epoch, graph_acc, plotLabel);
	};

	std::vector<std::tuple<float, float>> params = { 
		std::make_tuple(0.1f, 0.0f),
	};

	std::vector<std::thread> workers;
	const uint32_t totalNets = uint32_t(params.size());
	::initialize_plot(totalNets);
	for (uint32_t k = 0; k < totalNets; k += std::thread::hardware_concurrency())
	{
		const auto threadLimit = std::min(totalNets - k, std::thread::hardware_concurrency());
		for (size_t i = 0u; i < threadLimit; ++i)
		{
			const auto param = params.back();
			const auto eta = std::get<0>(param);
			const auto lambda = std::get<1>(param);
			params.pop_back();
			workers.push_back(std::thread(trainNet, k + i, eta, lambda, 32));
		}
		for (size_t i = 0u; i < threadLimit; ++i)
			workers[i].join();
		workers.clear();
	}

	g_PythonWrapper.save_plot(0.0, epochs, 0.0, 1.0, 1.0, 0.01, "Epochs", "Accuracy", "results.png");

	return 0;
}
