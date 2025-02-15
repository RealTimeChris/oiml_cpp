#pragma once

#include <oiml/oiml_cpu/util_functions.hpp>
#include <oiml/oiml_cpu/detect_isa.hpp>
#include <oiml/common/common.hpp>

#if defined(OIML_IS_X86_64)

namespace oiml {

	OIML_INLINE void oiml_vec_dot_q8_0_f32_avx512(int32_t n, float* s, const void* x_new, const float* vy) {
		constexpr int32_t qk = QK8_0;
		const int nb		 = n / qk;

		float sumf = 0;

	#if defined(__AVX512F__)

		const block_q8_0<64>* x = static_cast<const block_q8_0<64>*>(x_new);

		const __m512 zero = _mm512_setzero_ps();
		__m512 sumv0	  = _mm512_setzero_ps();
		__m512 sumv1	  = _mm512_setzero_ps();
		__m512 total_sum  = _mm512_setzero_ps();

		for (int32_t ib = 0; ib < nb - 4; ib += 4) {
			__m512 d_broad = _mm512_set1_ps(oiml_compute_fp16_to_fp32(x->d));

			__m256i qs = _mm256_loadu_si256(( __m256i* )(x->qs.data()));
			__m512i x0 = _mm512_cvtepi8_epi16(qs);

			sumv0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(x0))), _mm512_loadu_ps(vy), zero);
			sumv1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(x0, 1))), _mm512_loadu_ps(vy + 16), sumv0);

			total_sum = _mm512_fmadd_ps(sumv1, d_broad, total_sum);

			++x;
			vy += 32;

			d_broad = _mm512_set1_ps(oiml_compute_fp16_to_fp32(x->d));

			qs = _mm256_loadu_si256(( __m256i* )(x->qs.data()));
			x0 = _mm512_cvtepi8_epi16(qs);

			sumv0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(x0))), _mm512_loadu_ps(vy), zero);
			sumv1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(x0, 1))), _mm512_loadu_ps(vy + 16), sumv0);

			total_sum = _mm512_fmadd_ps(sumv1, d_broad, total_sum);

			++x;
			vy += 32;

			d_broad = _mm512_set1_ps(oiml_compute_fp16_to_fp32(x->d));

			qs = _mm256_loadu_si256(( __m256i* )(x->qs.data()));
			x0 = _mm512_cvtepi8_epi16(qs);

			sumv0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(x0))), _mm512_loadu_ps(vy), zero);
			sumv1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(x0, 1))), _mm512_loadu_ps(vy + 16), sumv0);

			total_sum = _mm512_fmadd_ps(sumv1, d_broad, total_sum);

			++x;
			vy += 32;

			d_broad = _mm512_set1_ps(oiml_compute_fp16_to_fp32(x->d));

			qs = _mm256_loadu_si256(( __m256i* )(x->qs.data()));
			x0 = _mm512_cvtepi8_epi16(qs);

			sumv0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(x0))), _mm512_loadu_ps(vy), zero);
			sumv1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(x0, 1))), _mm512_loadu_ps(vy + 16), sumv0);

			total_sum = _mm512_fmadd_ps(sumv1, d_broad, total_sum);

			++x;
			vy += 32;
		}

		__m256 sum = _mm256_add_ps(_mm512_castps512_ps256(total_sum), _mm512_extractf32x8_ps(total_sum, 1));

		{
			// hiQuad = ( x7, x6, x5, x4 )
			const __m128 hiQuad = _mm256_extractf128_ps(sum, 1);
			// loQuad = ( x3, x2, x1, x0 )
			const __m128 loQuad = _mm256_castps256_ps128(sum);
			// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
			const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
			// loDual = ( -, -, x1 + x5, x0 + x4 )
			const __m128 loDual = sumQuad;
			// hiDual = ( -, -, x3 + x7, x2 + x6 )
			const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
			// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
			const __m128 sumDual = _mm_add_ps(loDual, hiDual);
			// lo = ( -, -, -, x0 + x2 + x4 + x6 )
			const __m128 lo = sumDual;
			// hi = ( -, -, -, x1 + x3 + x5 + x7 )
			const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
			// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
			const __m128 sum = _mm_add_ss(lo, hi);
			sumf			 = _mm_cvtss_f32(sum);
		}
	#endif

		*s = sumf;
	}

	OIML_INLINE void oiml_vec_dot_q8_0_f32_avx2(int32_t n, float* s, const void* x_new, const float* vy) {
		constexpr int32_t qk = QK8_0;
		const int32_t nb	 = n / qk;

		float sumf = 0;
		
	#if defined(__AVX2__)

		const block_q8_0<32>* x = static_cast<const block_q8_0<32>*>(x_new);

		__m256 total_sum = _mm256_setzero_ps();

		for (int32_t ib = 0; ib < nb - 4; ib += 4) {
			__m256 local_sum = _mm256_setzero_ps();
			__m256 d_broad00 = _mm256_set1_ps(oiml_compute_fp16_to_fp32(x->d));

			__m256i x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs.data())));
			__m256i x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs.data() + 16)));
			++x;

			__m256 temp0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			__m256 y0	 = _mm256_loadu_ps(vy);
			local_sum	 = _mm256_fmadd_ps(temp0, y0, local_sum);

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

			x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs.data())));
			x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs.data() + 16)));

			d_broad00 = _mm256_set1_ps(oiml_compute_fp16_to_fp32(x->d));
			vy += 32;
			++x;

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

			x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs.data())));
			x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs.data() + 16)));

			d_broad00 = _mm256_set1_ps(oiml_compute_fp16_to_fp32(x->d));
			vy += 32;
			++x;

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

			x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs.data())));
			x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs.data() + 16)));

			d_broad00 = _mm256_set1_ps(oiml_compute_fp16_to_fp32(x->d));
			vy += 32;
			++x;

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

#endif

		*s = sumf;
	}

	OIML_INLINE void oiml_vec_dot_q8_0_f32_avx(int32_t n, float* s, const void* x, const float* vy) {
		std::cout << "WERE HERE 0000-03" << std::endl;
		constexpr int32_t qk = QK8_0;
		const int32_t nb	 = n / qk;

		float sumf = 0;
		*s		   = sumf;
	}

	using oiml_vec_dot_q8_0_f32_type = decltype(&oiml_vec_dot_q8_0_f32_avx);

	constexpr oiml_vec_dot_q8_0_f32_type func0 = &oiml_vec_dot_q8_0_f32_avx;
	constexpr oiml_vec_dot_q8_0_f32_type func1 = &oiml_vec_dot_q8_0_f32_avx2;
	constexpr oiml_vec_dot_q8_0_f32_type func2 = &oiml_vec_dot_q8_0_f32_avx512;

	inline static constexpr array<oiml_vec_dot_q8_0_f32_type, 3> oiml_vec_dot_q8_0_f32_type_funcs{ { func0, func1, func2 } };
	inline static const oiml_vec_dot_q8_0_f32_type oiml_vec_dot_q8_0_f32_function{ get_work_func(oiml_vec_dot_q8_0_f32_type_funcs, cpu_arch_index) };
}

#endif