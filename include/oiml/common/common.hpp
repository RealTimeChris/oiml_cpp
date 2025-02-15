/*
	MIT License

	Copyright (c) 2023 OpenInfer

	Permission is hereby granted, free of charge, to any person obtaining a copy of this
	software and associated documentation files (the "Software"), to deal in the Software
	without restriction, including without limitation the rights to use, copy, modify, merge,
	publish, distribute, sublicense, and/or sell copies of the Software, and to permit
	persons to whom the Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all copies or
	substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	DEALINGS IN THE SOFTWARE.
*/
/// https://github.com/OpenInfer/oiml
/// Updated: Sep 3, 2025

#pragma once

#include <oiml/common/config.hpp>
#include <oiml/oiml_cpu/detect_isa.hpp>
#include <oiml/common/allocator.hpp>
#include <oiml/common/file_loader.hpp>
#include <vector>
#include <array>

#if defined(OIML_IS_X86_64)

	#include <immintrin.h>

namespace oiml {

	using oiml_simd_int_128 = __m128i;
	using oiml_simd_int_256 = __m256i;
	using oiml_simd_int_512 = __m512i;

	constexpr array<uint64_t, 3> bitsPerStep{ { 128, 256, 512 } };

#elif defined(OIML_IS_ARM64)

	#include <arm_neon.h>

namespace oiml {

	using oiml_simd_int_128 = uint8x16_t;
	using oiml_simd_int_256 = uint32_t;
	using oiml_simd_int_512 = size_t;

#else

namespace oiml {

	using oiml_simd_int_128 = uint16_t;
	using oiml_simd_int_256 = uint32_t;
	using oiml_simd_int_512 = uint64_t;

#endif

	template<typename value_type>
	concept simd_int_512_type = std::is_same_v<oiml_simd_int_512, std::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept simd_int_256_type = std::is_same_v<oiml_simd_int_256, std::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept simd_int_128_type = std::is_same_v<oiml_simd_int_128, std::remove_cvref_t<value_type>>;

	static constexpr int32_t QK8_0{ 32 };

	using oiml_half = uint16_t;

	template<size_t alignment> struct block_q8_0 {
		OIML_ALIGN(alignment) array<int8_t, 32> qs {};
		oiml_half d{};
	};
}