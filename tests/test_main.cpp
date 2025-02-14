// oiml.cpp : Defines the entry point for the application.
//
#include <iostream>
#include <oiml/Index.hpp>
using namespace std;

int32_t main()
{
	oiml::block_q8_0<32> testValues[128]{};
	alignas(32) float testFloats01[32]{};
	alignas(32) float testFloats02[32]{};
	oiml::oiml_vec_dot_q8_0_f32_function(256, testFloats01, testValues, testFloats02);
	cout << "Hello CMake." << endl;
	return 0;
}
