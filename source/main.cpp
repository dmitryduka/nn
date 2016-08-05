#include <iostream>
#include "cost.hpp"
#include "layer.hpp"
#include "activations.hpp"

int main()
{
	{
		using namespace nn;
		using InputLayer = layer<LayerType::kEigenInput, ActivationType::kSigmoid, WeightInitializationType::kGaussian, 784>;
		using FirstLayer = layer<LayerType::kEigenRegular, ActivationType::kSigmoid, WeightInitializationType::kGaussian, 30, 784>;
		using SecondLayer = layer<LayerType::kEigenRegular, ActivationType::kSigmoid, WeightInitializationType::kGaussian, 10, 30>;
		using InputType = InputLayer::MatrixType;

		FirstLayer l1;
		SecondLayer l2;
		for (int i = 0; i < 500; ++i)
		{
			InputType in = InputType::Random(InputLayer::UnitsInLayer, 1);
			l1.computeWeightedSum(InputLayer::computeActivations(in));
			l1.computeActivations();
			l2.computeWeightedSum(l1.getOutput());
			l2.computeActivations();
		}
	}

	int x;
	std::cin >> x;
	return 0;
}