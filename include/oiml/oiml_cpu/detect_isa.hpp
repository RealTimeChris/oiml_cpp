#if (defined(__x86_64__) || defined(_M_AMD64)) && !defined(_M_ARM64EC)
	#define OIML_IS_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
	#define OIML_IS_ARM64 1
#elif defined(__riscv) && __riscv_xlen == 64
	#define OIML_IS_RISCV64 1
#elif defined(__loongarch_lp64)
	#define OIML_IS_LOONGARCH64 1
#elif defined(__PPC64__) || defined(_M_PPC64)
	#define OIML_IS_PPC64 1
	#if defined(__ALTIVEC__)
		#define OIML_IS_PPC64_VMX 1
	#endif
#endif

/* From
https://github.com/endorno/pytorch/blob/master/torch/lib/TH/generic/simd/simd.h
Highly modified.

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research Institute
(Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert,
Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
America and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

enum class instruction_set {
	DEFAULT		= 0x0,
	NEON		= 0x1,
	AVX2		= 0x4,
	SSE42		= 0x8,
	PCLMULQDQ	= 0x10,
	BMI1		= 0x20,
	BMI2		= 0x40,
	ALTIVEC		= 0x80,
	AVX512F		= 0x100,
	AVX512DQ	= 0x200,
	AVX512IFMA	= 0x400,
	AVX512PF	= 0x800,
	AVX512ER	= 0x1000,
	AVX512CD	= 0x2000,
	AVX512BW	= 0x4000,
	AVX512VL	= 0x8000,
	AVX512VBMI2 = 0x10000,
	LSX			= 0x20000,
	LASX		= 0x40000,
};

#include <cstdint>
#include <array>
#include <cstdlib>
#if defined(_MSC_VER)
	#include <intrin.h>
#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
	#include <cpuid.h>
#endif

namespace oiml {

#if defined(__PPC64__)

	static inline instruction_set detect_supported_architectures() {
		return instruction_set::ALTIVEC;
	}

#elif defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)

	static inline instruction_set detect_supported_architectures() {
		return instruction_set::NEON;
	}

#elif defined(__x86_64__) || defined(_M_AMD64)// x64


	namespace {
		// Can be found on Intel ISA Reference for CPUID
		constexpr uint32_t cpuid_avx2_bit		 = 1 << 5;///< @private Bit 5 of EBX for EAX=0x7
		constexpr uint32_t cpuid_bmi1_bit		 = 1 << 3;///< @private bit 3 of EBX for EAX=0x7
		constexpr uint32_t cpuid_bmi2_bit		 = 1 << 8;///< @private bit 8 of EBX for EAX=0x7
		constexpr uint32_t cpuid_avx512f_bit	 = 1 << 16;///< @private bit 16 of EBX for EAX=0x7
		constexpr uint32_t cpuid_avx512dq_bit	 = 1 << 17;///< @private bit 17 of EBX for EAX=0x7
		constexpr uint32_t cpuid_avx512ifma_bit	 = 1 << 21;///< @private bit 21 of EBX for EAX=0x7
		constexpr uint32_t cpuid_avx512pf_bit	 = 1 << 26;///< @private bit 26 of EBX for EAX=0x7
		constexpr uint32_t cpuid_avx512er_bit	 = 1 << 27;///< @private bit 27 of EBX for EAX=0x7
		constexpr uint32_t cpuid_avx512cd_bit	 = 1 << 28;///< @private bit 28 of EBX for EAX=0x7
		constexpr uint32_t cpuid_avx512bw_bit	 = 1 << 30;///< @private bit 30 of EBX for EAX=0x7
		constexpr uint32_t cpuid_avx512vl_bit	 = 1U << 31;///< @private bit 31 of EBX for EAX=0x7
		constexpr uint32_t cpuid_avx512vbmi2_bit = 1 << 6;///< @private bit 6 of ECX for EAX=0x7
		constexpr uint64_t cpuid_avx256_saved	 = uint64_t(1) << 2;///< @private bit 2 = AVX
		constexpr uint64_t cpuid_avx512_saved	 = uint64_t(7) << 5;///< @private bits 5,6,7 = opmask, ZMM_hi256, hi16_ZMM
		constexpr uint32_t cpuid_sse42_bit		 = 1 << 20;///< @private bit 20 of ECX for EAX=0x1
		constexpr uint32_t cpuid_osxsave		 = (uint32_t(1) << 26) | (uint32_t(1) << 27);///< @private bits 26+27 of ECX for EAX=0x1
		constexpr uint32_t cpuid_pclmulqdq_bit	 = 1 << 1;///< @private bit  1 of ECX for EAX=0x1
	}

	static inline void cpuid(int32_t* eax, int32_t* ebx, int32_t* ecx, int32_t* edx) {
	#if defined(_MSC_VER)
		int32_t cpu_info[4];
		__cpuidex(cpu_info, *eax, *ecx);
		*eax = cpu_info[0];
		*ebx = cpu_info[1];
		*ecx = cpu_info[2];
		*edx = cpu_info[3];
	#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
		uint32_t level = *eax;
		__get_cpuid(level, eax, ebx, ecx, edx);
	#else
		uint32_t a = *eax, b, c = *ecx, d;
		asm volatile("cpuid\n\t" : "+a"(a), "=b"(b), "+c"(c), "=d"(d));
		*eax = a;
		*ebx = b;
		*ecx = c;
		*edx = d;
	#endif
	}

	static inline uint64_t xgetbv() {
	#if defined(_MSC_VER)
		return _xgetbv(0);
	#else
		uint32_t xcr0_lo, xcr0_hi;
		asm volatile("xgetbv\n\t" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
		return xcr0_lo | (uint64_t(xcr0_hi) << 32);
	#endif
	}

	static inline instruction_set detect_supported_architectures() {
		int32_t eax, ebx, ecx, edx;
		uint32_t host_isa = 0x0;

		// EBX for EAX=0x1
		eax = 0x1;
		ecx = 0x0;
		cpuid(&eax, &ebx, &ecx, &edx);

		if (ecx & cpuid_sse42_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::SSE42);
		} else {
			return static_cast<instruction_set>(host_isa);// everything after is redundant
		}

		if (ecx & cpuid_pclmulqdq_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::PCLMULQDQ);
		}


		if ((ecx & cpuid_osxsave) != cpuid_osxsave) {
			return static_cast<instruction_set>(host_isa);
		}

		// xgetbv for checking if the OS saves registers
		uint64_t xcr0 = xgetbv();

		if ((xcr0 & cpuid_avx256_saved) == 0) {
			return static_cast<instruction_set>(host_isa);
		}

		// ECX for EAX=0x7
		eax = 0x7;
		ecx = 0x0;
		cpuid(&eax, &ebx, &ecx, &edx);
		if (ebx & cpuid_avx2_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::AVX2);
		}
		if (ebx & cpuid_bmi1_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::BMI1);
		}

		if (ebx & cpuid_bmi2_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::BMI2);
		}

		if (!((xcr0 & cpuid_avx512_saved) == cpuid_avx512_saved)) {
			return static_cast<instruction_set>(host_isa);
		}

		if (ebx & cpuid_avx512f_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::AVX512F);
		}

		if (ebx & cpuid_avx512dq_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::AVX512DQ);
		}

		if (ebx & cpuid_avx512ifma_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::AVX512IFMA);
		}

		if (ebx & cpuid_avx512pf_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::AVX512PF);
		}

		if (ebx & cpuid_avx512er_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::AVX512ER);
		}

		if (ebx & cpuid_avx512cd_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::AVX512CD);
		}

		if (ebx & cpuid_avx512bw_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::AVX512BW);
		}

		if (ebx & cpuid_avx512vl_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::AVX512VL);
		}

		if (ecx & cpuid_avx512vbmi2_bit) {
			host_isa |= static_cast<uint32_t>(instruction_set::AVX512VBMI2);
		}

		return static_cast<instruction_set>(host_isa);
	}

#elif defined(__loongarch_sx) && !defined(__loongarch_asx)

	static inline uint32_t detect_supported_architectures() {
		return static_cast<uint32_t>(instruction_set::LSX);
	}

#elif defined(__loongarch_asx)

	static inline uint32_t detect_supported_architectures() {
		return static_cast<uint32_t>(instruction_set::LASX);
	}

#else

	static inline uint32_t detect_supported_architectures() {
		return static_cast<uint32_t>(instruction_set::DEFAULT);
	}

#endif

	inline static const instruction_set cpu_arch{ detect_supported_architectures() };

	uint64_t get_cpu_arch_index(instruction_set set) {
		if (static_cast<uint32_t>(set) & static_cast<uint32_t>(instruction_set::AVX512F)) {
			std::cout << "CURRENT SET-2: " << ( int32_t )set << std::endl;
			return 2;
		} else if (static_cast<uint32_t>(set) & static_cast<uint32_t>(instruction_set::AVX2)) {
			std::cout << "CURRENT SET-1: " << ( int32_t )set << std::endl;
			return 1;
		} else if (static_cast<uint32_t>(set) & static_cast<uint32_t>(instruction_set::SSE42)) {
			std::cout << "CURRENT SET-0: " << ( int32_t )set << std::endl;
			return 0;
		} else if (static_cast<uint32_t>(set) & static_cast<uint32_t>(instruction_set::NEON)) {
			std::cout << "CURRENT SET-NEON: " << ( int32_t )set << std::endl;
			return 0;
		} else {
			return 0;
		}
	}

	inline static const auto cpu_arch_index{ get_cpu_arch_index(cpu_arch) };

	template<typename function_type, size_t size> const function_type& get_work_func(const array<function_type, size>& functions, const size_t index) {
		std::cout << "CURRENT IMPL INDEX: " << index << std::endl;
		std::cout << "ADDRESS: " << functions[index] << std::endl;
		return functions[index];
	}

}