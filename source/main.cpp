#include <iostream>

#include "cost.hpp"
#include "timing.hpp"
#include "layer.hpp"
#include "activations.hpp"

int main()
{
	{
		using namespace nn;
		using InputLayer = layer<LayerType::kEigenInput, ActivationType::kSigmoid, 256>;
		using FirstLayer = layer<LayerType::kEigenRegular, ActivationType::kSigmoid, 30, 256>;
		using SecondLayer = layer<LayerType::kEigenRegular, ActivationType::kSigmoid, 10, 30>;
		using InputType = InputLayer::MatrixType;

		timing timer;
		FirstLayer l1;
		SecondLayer l2;
		InputType in = InputType::Random(InputLayer::UnitsInLayer, 1);
		l1.computeWeightedSum(InputLayer::computeActivations(in));
		l1.computeActivations();
		l2.computeWeightedSum(l1.getOutput());
		l2.computeActivations();
		timer.printDuration();
	}

	int x;
	std::cin >> x;
	return 0;
}