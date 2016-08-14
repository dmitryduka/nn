#pragma once

#include "layer.hpp"
#include <Eigen/Dense>

namespace
{
	using namespace nn;

	uint32_t conv_size(uint32_t input, uint32_t kernel_width, uint32_t padding, uint32_t stride)
	{
		return (input - kernel_width + 2 * padding) / stride + 1;
	}

	layer::MatrixType conv_helper(const layer::MatrixType& input,
		const layer::MatrixType& kernel,
		uint32_t start_row, uint32_t start_col,
		uint32_t end_row, uint32_t end_col,
		uint32_t stride,
		bool normalize)
	{
		layer::MatrixType output = layer::MatrixType::Zero(end_row - start_row, end_col - start_col);
		const auto KSizeX = kernel.rows();
		const auto KSizeY = kernel.cols();

		for (auto row = start_row; row < end_row; row += stride)
			for (auto col = start_col; col < end_col; col += stride)
				output(row - start_row, col - start_col) = input.block(row - KSizeX / 2, col - KSizeY / 2, KSizeX, KSizeY).cwiseProduct(kernel).sum();

		if (normalize)
			return output / kernel.sum();
		else
			return output;
	}

	layer::MatrixType zero_pad(const layer::MatrixType& input, uint32_t size)
	{
		layer::MatrixType result = layer::MatrixType::Zero(input.rows() + 2 * size, input.cols() + 2 * size);
		result.block(size, size, input.rows(), input.cols()) = input;
		return result;
	}
}

namespace nn
{
	layer::MatrixType conv(const layer::MatrixType& input, const layer::MatrixType& kernel, uint32_t stride, bool zeroPad = false, bool normalize = false)
	{
		const auto kernel_half_width = kernel.cols() / 2;
		const auto kernel_half_height = kernel.rows() / 2;
		if (!zeroPad)
		{
			const auto result_width = conv_size(input.cols(), kernel.cols(), 0, stride);
			const auto result_height = conv_size(input.rows(), kernel.rows(), 0, stride);
			const auto conv_start_col = (input.cols() - result_width) / 2;
			const auto conv_start_row = (input.rows() - result_height) / 2;
			const auto conv_end_col = conv_start_col + result_width;
			const auto conv_end_row = conv_start_row + result_height;
			return conv_helper(input, kernel, conv_start_row, conv_start_col, conv_end_row, conv_end_col, stride, normalize);
		}
		else
		{
			return conv_helper(zero_pad(input, kernel_half_width), 
				kernel, kernel_half_width, kernel_half_height, 
				kernel_half_width + input.cols(),
				kernel_half_height + input.rows(),
				stride,
				normalize);
		}
	}

}