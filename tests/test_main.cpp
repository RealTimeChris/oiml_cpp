// oiml.cpp : Defines the entry point for the application.
//
#include <iostream>
#include <oiml/Index.hpp>

int32_t main() {
	
	oiml::vector<char, 32> vector{};
	vector.resize(2323);
	std::cout << "oiml::cpu_arch: " << static_cast<int>(oiml::cpu_arch) << std::endl;
	oiml::block_q8_0<32> testValues[128]{};
	alignas(32) float testFloats01[32]{};
	alignas(32) float testFloats02[32]{};
	oiml::oiml_vec_dot_q8_0_f32(128, testFloats01, testValues, testFloats02);
	std::cout << "Function -DONE 0101" << std::endl;
	std::cout << "Function -DONE 0202" << std::endl;
	//oiml_vec_dot_q8_0_f32_function();
	std::cout << "Function -DONE 0303" << std::endl;
	return 0;
}
