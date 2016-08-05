#include <iostream>
#include "mnist.hpp"
#include "network.hpp"

int main()
{
	{
		using namespace nn;
		network net;
		net.addInputLayer<ActivationType::kSigmoid>(28 * 28);
		net.addRegularLayer<ActivationType::kSigmoid, WeightInitializationType::kGaussian>(30);
		net.addRegularLayer<ActivationType::kSigmoid, WeightInitializationType::kGaussian>(10);

		auto images = loadMNISTImages("externals/mnist/train-images.idx3-ubyte", 60000);
		auto labels = loadMNISTLabels("externals/mnist/train-labels.idx1-ubyte", 60000);
		const size_t epochs = 30;
		for (size_t epoch = 0u; epoch < epochs; ++epoch)
		{
			for (size_t k = 0u; k < 6000u; k++)
			{
				const size_t batch_size = 10;
				for (size_t i = k * batch_size; i < (k + 1) * batch_size; ++i)
				{
					auto correct = net.feedforward(images[i]);
					net.backprop(labels[i]);
				}
				net.update_weights(2.0, batch_size);
			}
			real correct = net.evaluate(images, labels, 10000);
			std::cout << "Epoch " << epoch << ", acc: " << correct * 100.0f << "%" << std::endl;
		}
	}
	int x;
	std::cin >> x;
	return 0;
}