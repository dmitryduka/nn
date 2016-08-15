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
#include "convolution.hpp"
#if USE_PYTHON == 1
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif
#include "python/py_plot.h"
#endif

namespace
{
	struct PythonWrapper
	{
		std::mutex m_pythonLock;
#if USE_PYTHON == 1
		PythonWrapper()
		{
			Py_Initialize();
			initplot();
		}
		~PythonWrapper() 
		{ 
			Py_Finalize();
		}
#endif

		template<typename ... T>
		void plot(T... t)
		{
#if USE_PYTHON == 1
			std::lock_guard<std::mutex> lock(m_pythonLock);
			::plot(t...);
#endif
		}
		template<typename ... T>
		void save_plot(T... t)
		{
#if USE_PYTHON == 1
			std::lock_guard<std::mutex> lock(m_pythonLock);
			::save_plot(t...);
#endif
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
	const uint32_t epochs = 3;
	const uint32_t dataset_size = 60000;
	const uint32_t training_set_size = 55000;
	std::vector<MatrixType> training_set, validation_set;
	std::vector<uint8_t> training_labels, validation_labels;
	{
		const auto images = loadMNISTImages("externals/mnist/train-images.idx3-ubyte", LoadSettings(kNormalize | kVectorize), dataset_size);
		const auto labels = loadMNISTLabels("externals/mnist/train-labels.idx1-ubyte", dataset_size);
		training_set = std::vector<MatrixType>(images.cbegin(), images.cbegin() + training_set_size);
		validation_set = std::vector<MatrixType>(images.cbegin() + training_set_size, images.cend());
		training_labels = std::vector<uint8_t>(labels.cbegin(), labels.cbegin() + training_set_size);
		validation_labels = std::vector<uint8_t>(labels.cbegin() + training_set_size, labels.cend());
	}

	network original_net;
	original_net.addLayer(LayerType::kInput, 28 * 28, ActivationType::kNone, WeightInitializationType::kNone);
	original_net.addLayer(LayerType::kFC, 256, ActivationType::kLRelu, WeightInitializationType::kWeightedGaussian);
	original_net.addLayer(LayerType::kFC, 256, ActivationType::kLRelu, WeightInitializationType::kWeightedGaussian);
	original_net.addLayer(LayerType::kSoftmax, 10, ActivationType::kNone, WeightInitializationType::kWeightedGaussian);
	original_net.setCostFunction(CostType::kCrossEntropy);
	auto train_net = [&](uint32_t netNo, real eta, real lambda, uint32_t batch_size)
	{
		std::vector<float> graph_epoch, graph_acc;
		network net = original_net;
		// SGD
		timing timer;
		const uint32_t batches = uint32_t(training_set.size() / batch_size);
		evaluate_results result;
		for (size_t epoch = 0u; epoch < epochs; ++epoch)
		{
			net.psgd(28, 28, batches, batch_size, eta, lambda, training_set, training_labels, false);
			result = net.evaluate(validation_set, validation_labels);
			graph_epoch.push_back(epoch);
			graph_acc.push_back(result.accuracy);
			std::cout << "acc: " << result.accuracy * 100.0f << "%, cost = " << result.cost << " (" << timer.seconds() << " seconds passed)" << std::endl;
			std::cout << "Epoch " << epoch << " (" << timer.seconds() << " seconds passed)" << std::endl;
		}

		const bool dump_error_images = false;
		if (dump_error_images)
		{
			for (auto i : result.errors)
			{
				std::vector<float> img(validation_set[i].data(), validation_set[i].data() + 28 * 28);
#if USE_PYTHON == 1
				plot_image(img, "error" + to_string(i) + ".png");
#endif
			}
		}

		std::string plotLabel = "[" + to_string(netNo) + "] eta=" + to_string(eta) + ",lambda=" + to_string(lambda) + ", bs=" + to_string(batch_size);
		g_PythonWrapper.plot(netNo, graph_epoch, graph_acc, plotLabel);
	};

#if USE_PYTHON == 1
	::initialize_plot(1);
#endif
	train_net(0, 0.01, 0.0, 32);
	g_PythonWrapper.save_plot(0.0, epochs - 1, 0.0, 1.0, 1.0, 0.01, "Epochs", "Accuracy", "results.png");

	return 0;
}
