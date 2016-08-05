#pragma once

#include <chrono>
#include <iostream>

namespace nn
{
	class timing
	{
	public:
		timing() { start(); }
		void start() { m_start = std::chrono::high_resolution_clock::now(); }
		float ms() { return std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - m_start).count(); }
		void printDuration() { std::cout << "Time: " << ms() << " s" << std::endl; }
	private:
		std::chrono::time_point<std::chrono::steady_clock> m_start;
	};
}
