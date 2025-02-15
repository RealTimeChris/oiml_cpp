#pragma once

#include <oiml/oiml_cpu/util_functions.hpp>
#include <oiml/oiml_cpu/detect_isa.hpp>
#include <oiml/common/common.hpp>

namespace oiml {

	class impl {
	  public:
		virtual OIML_INLINE void oiml_vec_dot_q8_0_f32(int32_t n, float* s, const void* x, const float* vy) const = 0;
	};

}
