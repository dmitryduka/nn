#include <iostream>
#include "mnist.hpp"
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
		net.addRegularLayer<type, WeightInitializationType::kWeightedGaussian>(30);
		net.addRegularLayer<type, WeightInitializationType::kWeightedGaussian>(10);

		auto images = loadMNISTImages("externals/mnist/train-images.idx3-ubyte");
		auto labels = loadMNISTLabels("externals/mnist/train-labels.idx1-ubyte");
		const size_t epochs = 30;
		const size_t batch_size = 10;
		const size_t test_set_size = 10000; // saves time to evaluate only a part of the set		
		real eta = 2.0f;
		// SGD
		for (size_t epoch = 0u; epoch < epochs; ++epoch)
		{
			const size_t batches = images.size() / batch_size;
			for (size_t k = 0u; k < batches; k++)
			{
				for (size_t i = k * batch_size; i < (k + 1) * batch_size; ++i)
				{
					auto correct = net.feedforward(images[i]);
					net.backprop(labels[i]);
				}
				net.update_weights(eta, batch_size);
			}
			real correct = net.evaluate(images, labels, test_set_size);
			if (correct > 0.96) eta = 1.0f;
			if (correct > 0.97) eta = 0.5f;
			std::cout << "Epoch " << epoch << ", acc: " << correct * 100.0f << "%" << std::endl;
		}
	}
	int x;
	std::cin >> x;
	return 0;
}