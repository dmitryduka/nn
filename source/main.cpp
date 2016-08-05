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
		for (int k = 0; k < 1000; k++)
		{
			const size_t batch_size = 20;
			for (size_t i = k * batch_size; i < (k + 1) * batch_size; ++i)
			{
				auto correct = net.feedforward(images[i]);
				net.backprop(labels[i]);
			}
			net.update_weights(3.0, batch_size);
			real correct = net.evaluate(images, labels);
			std::cout << "Correct: " << correct << std::endl;
		}
	}

	return 0;
}