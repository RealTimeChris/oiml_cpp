#pragma once

#include <oiml/oiml_cpu/detect_isa.hpp>
#include <oiml/oiml_cpu/common.hpp>

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)

namespace oiml {	

	OIML_INLINE void oiml_vec_dot_q8_0_f32_arm(int32_t n, float*s, const block_q8_0*x, const float*y) {
		constexpr int32_t qk = QK8_0;
		const int32_t nb	 = n / qk;

		float sumf = 0;

		float32x4_t sumv0 = vdupq_n_f32(0.0f);
		float32x4_t sumv1 = vdupq_n_f32(0.0f);
		float32x4_t sumv2 = vdupq_n_f32(0.0f);
		float32x4_t sumv3 = vdupq_n_f32(0.0f);
		float32x4_t sumv4 = vdupq_n_f32(0.0f);
		float32x4_t sumv5 = vdupq_n_f32(0.0f);
		float32x4_t sumv6 = vdupq_n_f32(0.0f);
		float32x4_t sumv7 = vdupq_n_f32(0.0f);

		int ib = 0;
		for (; ib < nb; ib += 1) {
			const block_q8_0* restrict b = x++;
			const float d				 = GGML_FP16_TO_FP32(b->d);

			//        const uint8x16x2_t x0_0 = vld2q_s8(x0->qs);
			int8x16_t vec1 = vld1q_s8(b->qs);// load 16x int8_t
			int8x16_t vec2 = vld1q_s8(b->qs + 16);// load 16x int8_t
			int16x8_t x0_0 = vmovl_s8(vget_low_s8(vec1));// cast the first 8x int8_t to int16_t
			int16x8_t x0_1 = vmovl_s8(vget_high_s8(vec1));// cast the last 8x int8_t to int16_t
			int16x8_t x1_0 = vmovl_s8(vget_low_s8(vec2));// cast the first 8x int8_t to int16_t
			int16x8_t x1_1 = vmovl_s8(vget_high_s8(vec2));// cast the last 8x int8_t to int16_t
			//printf("0:%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", ((int8_t *)&vec)[0], ((int8_t *)&vec)[1], ((int8_t *)&vec)[2], ((int8_t *)&vec)[3], ((int8_t *)&vec)[4], ((int8_t *)&vec)[5], ((int8_t *)&vec)[6], ((int8_t *)&vec)[7], ((int8_t *)&vec)[8], ((int8_t *)&vec)[9], ((int8_t *)&vec)[10], ((int8_t *)&vec)[11], ((int8_t *)&vec)[12], ((int8_t *)&vec)[13], ((int8_t *)&vec)[14], ((int8_t *)&vec)[15]);
			//printf("2:%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", ((int32_t *)&x0_00)[0], ((int32_t *)&x0_00)[1], ((int32_t *)&x0_00)[2], ((int32_t *)&x0_00)[3], ((int32_t *)&x0_01)[0], ((int32_t *)&x0_01)[1], ((int32_t *)&x0_01)[2], ((int32_t *)&x0_01)[3], ((int32_t *)&x0_02)[0], ((int32_t *)&x0_02)[1], ((int32_t *)&x0_02)[2], ((int32_t *)&x0_02)[3], ((int32_t *)&x0_03)[0], ((int32_t *)&x0_03)[1], ((int32_t *)&x0_03)[2], ((int32_t *)&x0_03)[3]);
			//printf("4:%f %f %f %f %d %d %d %d %d %d %d %d %d %d %d %d\n", ((float *)&xx0)[0], ((float *)&xx0)[1], ((float *)&xx0)[2], ((float *)&xx0)[3], ((int32_t *)&x0_01)[0], ((int32_t *)&x0_01)[1], ((int32_t *)&x0_01)[2], ((int32_t *)&x0_01)[3], ((int32_t *)&x0_02)[0], ((int32_t *)&x0_02)[1], ((int32_t *)&x0_02)[2], ((int32_t *)&x0_02)[3], ((int32_t *)&x0_03)[0], ((int32_t *)&x0_03)[1], ((int32_t *)&x0_03)[2], ((int32_t *)&x0_03)[3]);

			const float32x4_t x0 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s8(x0_0))), d);
			const float32x4_t x1 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s8(x0_0))), d);
			const float32x4_t x2 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s8(x0_1))), d);
			const float32x4_t x3 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s8(x0_1))), d);
			const float32x4_t x4 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s8(x1_0))), d);
			const float32x4_t x5 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s8(x1_0))), d);
			const float32x4_t x6 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s8(x1_1))), d);
			const float32x4_t x7 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s8(x1_1))), d);

			sumv0 = vfmaq_f32(sumv0, x0, vld1q_f32(y));
			sumv1 = vfmaq_f32(sumv1, x1, vld1q_f32(y + 4));
			sumv2 = vfmaq_f32(sumv2, x2, vld1q_f32(y + 8));
			sumv3 = vfmaq_f32(sumv3, x3, vld1q_f32(y + 12));
			sumv4 = vfmaq_f32(sumv4, x4, vld1q_f32(y + 16));
			sumv5 = vfmaq_f32(sumv5, x5, vld1q_f32(y + 20));
			sumv6 = vfmaq_f32(sumv6, x6, vld1q_f32(y + 24));
			sumv7 = vfmaq_f32(sumv7, x7, vld1q_f32(y + 28));
			y += 32;
		}

		sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3) + vaddvq_f32(sumv4) + vaddvq_f32(sumv5) + vaddvq_f32(sumv6) + vaddvq_f32(sumv7);

		*s = sumf;
	}

	using oiml_vec_dot_q8_0_f32_type = decltype(&oiml_vec_dot_q8_0_f32_arm);

	static constexpr oiml_vec_dot_q8_0_f32_type oiml_vec_dot_q8_0_f32_type_funcs[] = { oiml_vec_dot_q8_0_f32_arm };
	static const oiml_vec_dot_q8_0_f32_type oiml_vec_dot_q8_0_f32_function{ get_work_func(oiml_vec_dot_q8_0_f32_type_funcs, internal::cpu_arch_index) };
}

#endif