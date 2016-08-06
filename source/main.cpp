#include <iostream>
#include "mnist.hpp"
#include "timing.hpp"
#include "network.hpp"

int main()
{
	// RELU - tops at ~84%, eta = 0.05, batch_size = 30, 100 (seems to not matter much)
	// Sigmoid - tops at ~98%, eta = 2.0, 1.0, 0.5 (<0.96,<0.97,<1.0), batch_size = 10
	{
		using namespace nn;
		network net;
		constexpr ActivationType type = ActivationType::kSigmoid;

		net.addInputLayer<type>(28 * 28);
		net.addRegularLayer<type, WeightInitializationType::kWeightedGaussian>(50);
		net.addRegularLayer<type, WeightInitializationType::kWeightedGaussian>(10);
		
		std::vector<MatrixType> training_set, validation_set;
		std::vector<uint8_t> training_labels, validation_labels;
		const size_t dataset_size = 60000;
		{
			const auto images = loadMNISTImages("externals/mnist/train-images.idx3-ubyte", dataset_size);
			const auto labels = loadMNISTLabels("externals/mnist/train-labels.idx1-ubyte", dataset_size);
			training_set = std::vector<MatrixType>(images.cbegin(), images.cbegin() + 50000);
			validation_set = std::vector<MatrixType>(images.cbegin() + 50000, images.cend());
			training_labels = std::vector<uint8_t>(labels.cbegin(), labels.cbegin() + 50000);
			validation_labels = std::vector<uint8_t>(labels.cbegin() + 50000, labels.cend());
		}
		const size_t epochs = 30;
		const size_t batch_size = 10;
		real eta = 2.0f;
		// SGD
		timing timer;
		for (size_t epoch = 0u; epoch < epochs; ++epoch)
		{
			const size_t batches = training_set.size() / batch_size;
			MatrixType image_batch = MatrixType::Zero(28 * 28, batch_size);
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
				net.feedforward(image_batch);
				net.backprop(label_batch);
				net.update_weights(eta, batch_size);
			}
			real correct = net.evaluate(validation_set, validation_labels);
			// learning rate slow down at peak accuracy
			if (correct > 0.96) eta = 1.0f;
			if (correct > 0.97) eta = 0.5f;
			std::cout << "Epoch " << epoch << ", acc: " << correct * 100.0f << "% (" << timer.seconds() << " seconds passed)" << std::endl;
		}
	}
	int x;
	std::cin >> x;
	return 0;
}