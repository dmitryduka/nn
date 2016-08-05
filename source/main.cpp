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

		auto images = loadMNISTImages("externals/mnist/train-images.idx3-ubyte", 1);
		auto labels = loadMNISTLabels("externals/mnist/train-labels.idx1-ubyte", 1);
		auto correct = net.evaluate(images, labels);
		net.backprop(labels[0]);
		std::cout << "Correct: " << correct << std::endl;
	}

	return 0;
}