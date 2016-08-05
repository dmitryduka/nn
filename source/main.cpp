#include <iostream>

#include "network.hpp"

int main()
{
	{
		using namespace nn;
		network net;
		net.addInputLayer<ActivationType::kSigmoid>(28 * 28);
		net.addRegularLayer<ActivationType::kSigmoid, WeightInitializationType::kGaussian>(30);
		net.addRegularLayer<ActivationType::kSigmoid, WeightInitializationType::kGaussian>(10);

		using MatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
		const MatrixType input = MatrixType::Random(28 * 28, 1);
		const MatrixType output = net.feedforward(input);
	}

	return 0;
}