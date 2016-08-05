#pragma once
#include "settings.hpp"
#if USE_EIGEN == 1
#include <Eigen/Dense>
#endif

namespace nn
{
#if USE_EIGEN == 1
	using MatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
	real mse(const MatrixType& output, const MatrixType& truth)
	{
		if (output.rows() != truth.rows() || output.cols() != truth.cols())
			throw std::logic_error("Cost functions require equally sized matrices");
		return real((truth - output).squaredNorm());
	}
#endif
}