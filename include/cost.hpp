#pragma once
#include "settings.hpp"
#if USE_EIGEN == 1
#include <Eigen/Dense>
#endif

namespace nn
{
	enum class CostType
	{
		kQuadratic,
		kCrossEntropy
	};

#if USE_EIGEN == 1
	using MatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
	using CostFunction = real(*)(const MatrixType& output, const MatrixType& truth);
	using CostDerivativeFunction = MatrixType(*)(const MatrixType& output, const MatrixType& truth);

	template<CostType t> real cost(const MatrixType& output, const MatrixType& truth);
	template<CostType t> MatrixType cost_derivative(const MatrixType& output, const MatrixType& truth);

	// mse
	template<>
	real cost<CostType::kQuadratic>(const MatrixType& output, const MatrixType& truth)
	{
		if (output.rows() != truth.rows() || output.cols() != truth.cols())
			throw std::logic_error("Cost functions require equally sized matrices");
		const real c = real((truth - output).squaredNorm());
		return real(0.5) * c;
	}
	template<>
	MatrixType cost_derivative<CostType::kQuadratic>(const MatrixType& output, const MatrixType& truth)
	{
		if (output.rows() != truth.rows() || output.cols() != truth.cols())
			throw std::logic_error("Cost functions require equally sized matrices");
		return output - truth;
	}

	// cross-entropy
	template<>
	real cost<CostType::kCrossEntropy>(const MatrixType& output, const MatrixType& truth)
	{
		if (output.rows() != truth.rows() || output.cols() != truth.cols())
			throw std::logic_error("Cost functions require equally sized matrices");
		const MatrixType one = MatrixType::Ones(truth.rows(), truth.cols());
		const MatrixType tmp = -truth * output.unaryExpr(&logf) - (one - truth) * ((one - output).unaryExpr(&logf));
		if (tmp.hasNaN())
			throw std::runtime_error("NaNs detected");
		return tmp.sum();
	}
	template<>
	MatrixType cost_derivative<CostType::kCrossEntropy>(const MatrixType& output, const MatrixType& truth)
	{
		if (output.rows() != truth.rows() || output.cols() != truth.cols())
			throw std::logic_error("Cost functions require equally sized matrices");
		return output - truth;
	}
#endif
}