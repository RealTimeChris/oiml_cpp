#pragma once

#include <oiml/oiml_cpu/arm.hpp>
#include <oiml/oiml_cpu/avx.hpp>

namespace oiml {

	OIML_INLINE void oiml_vec_dot_q8_0_f32(int32_t n, float* s, const void* x, const float* vy) {
		const auto function_ptr = oiml_vec_dot_q8_0_f32_function;
		function_ptr(n, s, x, vy);
	}

}