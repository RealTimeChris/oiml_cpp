#pragma once

#include <oiml/oiml_cpu/util_functions.hpp>
#include <oiml/oiml_cpu/detect_isa.hpp>
#include <oiml/oiml_cpu/common.hpp>

#if (defined(__x86_64__) || defined(_M_AMD64)) && !defined(_M_ARM64EC)

namespace oiml {

	OIML_INLINE void oiml_vec_dot_q8_0_f32_avx512(int32_t n, float* s, const void* x, const float* vy) {
		std::cout << "WERE HERE 0000-01" << std::endl;
		constexpr int32_t qk = QK8_0;
		const int32_t nb	 = n / qk;

		float sumf = 0;

		*s = sumf;
	}

	OIML_INLINE void oiml_vec_dot_q8_0_f32_avx2(int32_t n, float* s, const void* x, const float* vy) {
		std::cout << "WERE HERE 0000" << std::endl;
		constexpr int32_t qk = QK8_0;
		const int32_t nb	 = n / qk;

		float sumf		 = 0;
		__m256 total_sum = _mm256_setzero_ps();
		const block_q8_0<32>* vx{ static_cast<const block_q8_0<32>*>(x) };
		std::cout << "WERE HERE 0000" << std::endl;

		for (int32_t ib = 0; ib < nb - 4; ib += 4) {
			__m256 local_sum = _mm256_setzero_ps();
			__m256 d_broad00 = _mm256_set1_ps(oiml_compute_fp16_to_fp32(vx->d));
			std::cout << "WERE HERE 0101" << std::endl;

			__m256i x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(vx->qs.data())));
			__m256i x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(vx->qs.data() + 16)));
			++vx;
			std::cout << "WERE HERE 0202" << std::endl;
			__m256 temp0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			__m256 y0	 = _mm256_loadu_ps(vy);
			local_sum	 = _mm256_fmadd_ps(temp0, y0, local_sum);
			std::cout << "WERE HERE 0303" << std::endl;
			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
			y0		  = _mm256_loadu_ps(vy + 8);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
			y0		  = _mm256_loadu_ps(vy + 16);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
			y0		  = _mm256_loadu_ps(vy + 24);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
			local_sum = _mm256_setzero_ps();
			std::cout << "WERE HERE 0404" << std::endl;
			x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(vx->qs.data())));
			x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(vx->qs.data() + 16)));

			d_broad00 = _mm256_set1_ps(oiml_compute_fp16_to_fp32(vx->d));
			vy += 32;
			++vx;

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			y0		  = _mm256_loadu_ps(vy);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
			y0		  = _mm256_loadu_ps(vy + 8);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
			y0		  = _mm256_loadu_ps(vy + 16);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
			y0		  = _mm256_loadu_ps(vy + 24);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
			local_sum = _mm256_setzero_ps();

			x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(vx->qs.data())));
			x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(vx->qs.data() + 16)));

			d_broad00 = _mm256_set1_ps(oiml_compute_fp16_to_fp32(vx->d));
			vy += 32;
			++vx;

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			y0		  = _mm256_loadu_ps(vy);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
			y0		  = _mm256_loadu_ps(vy + 8);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
			y0		  = _mm256_loadu_ps(vy + 16);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
			y0		  = _mm256_loadu_ps(vy + 24);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
			local_sum = _mm256_setzero_ps();

			x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(vx->qs.data())));
			x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(vx->qs.data() + 16)));

			d_broad00 = _mm256_set1_ps(oiml_compute_fp16_to_fp32(vx->d));
			vy += 32;
			++vx;

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			y0		  = _mm256_loadu_ps(vy);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
			y0		  = _mm256_loadu_ps(vy + 8);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
			y0		  = _mm256_loadu_ps(vy + 16);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
			y0		  = _mm256_loadu_ps(vy + 24);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);
			total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
			vy += 32;
		}

		__m128 sum = _mm_add_ps(_mm256_castps256_ps128(total_sum), _mm256_extractf128_ps(total_sum, 1));

		sum = _mm_hadd_ps(sum, sum);
		sum = _mm_hadd_ps(sum, sum);

		sumf = _mm_cvtss_f32(sum);

		*s = sumf;
	}

	OIML_INLINE void oiml_vec_dot_q8_0_f32_avx(int32_t n, float* s, const void* x, const float* vy) {
		std::cout << "WERE HERE 0000-03" << std::endl;
		constexpr int32_t qk = QK8_0;
		const int32_t nb	 = n / qk;

		float sumf = 0;
		*s		   = sumf;
	}

	using oiml_vec_dot_q8_0_f32_type = void (*)(int32_t , float* , const void* , const float* );

	static constexpr std::array<oiml_vec_dot_q8_0_f32_type, 3> oiml_vec_dot_q8_0_f32_type_funcs{ { oiml_vec_dot_q8_0_f32_avx, oiml_vec_dot_q8_0_f32_avx2,
		oiml_vec_dot_q8_0_f32_avx512 } };
	static const oiml_vec_dot_q8_0_f32_type oiml_vec_dot_q8_0_f32_function{ get_work_func(oiml_vec_dot_q8_0_f32_type_funcs, internal::cpu_arch_index) };
}

#endif