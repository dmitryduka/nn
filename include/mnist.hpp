#pragma once
#include <vector>
#include <fstream>
#include "settings.hpp"
#if USE_EIGEN == 1
#include <Eigen/Dense>
#endif

namespace nn
{
	using MatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
	std::vector<MatrixType> loadMNISTImages(const std::string& filename, uint32_t maxImages = 0xFFFFFFFF)
	{
		std::ifstream in(filename, std::ifstream::binary);
		if (in)
		{
			in.seekg(0, in.end);
			size_t fileLen = in.tellg();
			in.seekg(0, in.beg);
			std::vector<unsigned char> images(fileLen);
			in.read(reinterpret_cast<char*>(images.data()), images.size());
			if (images.size() < 16 + 28 * 28)
				throw std::runtime_error("No images in the dataset");
			for (size_t i = 0; i < 16; i += 4)
			{
				std::swap(images[i], images[i + 3]);
				std::swap(images[i + 1], images[i + 2]);
			}
			uint32_t magic, datasetSize, imageWidth, imageHeight;
			magic = *reinterpret_cast<uint32_t*>(images.data());
			if (magic != 0x00000803)
				throw std::runtime_error("Bad magic number");
			datasetSize = *reinterpret_cast<uint32_t*>(images.data() + 4);
			if (datasetSize > maxImages)
				datasetSize = maxImages;
			imageWidth = *reinterpret_cast<uint32_t*>(images.data() + 8);
			imageHeight = *reinterpret_cast<uint32_t*>(images.data() + 12);
			std::vector<MatrixType> result;
			size_t position = 16;
			for (uint32_t i = 0u; i < datasetSize; ++i)
			{
				const auto vectorizedSize = imageWidth * imageHeight;
				MatrixType image(vectorizedSize, 1);
					for (size_t j = 0; j < vectorizedSize; ++j)
						image(j, 0) = real(images[position++]);
				image /= 255.0;
				result.push_back(image);
			}			
			return result;
		}
		else
			throw std::runtime_error("Can't open MNIST images dataset");
	}

	std::vector<uint8_t> loadMNISTLabels(const std::string& filename, uint32_t maxLabels = 0xFFFFFFFF)
	{
		std::ifstream in(filename, std::ifstream::binary);
		if (in)
		{
			in.seekg(0, in.end);
			size_t fileLen = in.tellg();
			in.seekg(0, in.beg);
			std::vector<unsigned char> images(fileLen);
			in.read(reinterpret_cast<char*>(images.data()), images.size());
			if (images.size() < 16 + 28 * 28)
				throw std::runtime_error("No images in the dataset");
			for (size_t i = 0; i < 8; i += 4)
			{
				std::swap(images[i], images[i + 3]);
				std::swap(images[i + 1], images[i + 2]);
			}
			uint32_t magic, datasetSize;
			magic = *reinterpret_cast<uint32_t*>(images.data());
			if (magic != 0x00000801)
				throw std::runtime_error("Bad magic number");
			datasetSize = *reinterpret_cast<uint32_t*>(images.data() + 4);
			if (datasetSize > maxLabels)
				datasetSize = maxLabels;
			std::vector<uint8_t> result(images.cbegin() + 8, images.cbegin() + 8 + datasetSize);
			return result;
		}
		else
			throw std::runtime_error("Can't open MNIST images dataset");
	}
}
