/*
	MIT License

	Copyright (c) 2024 RealTimeChris

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
/// https://github.com/RealTimeChris/jsonifier
/// Feb 3, 2023
#pragma once

#include <oiml/common/config.hpp>
#include <memory_resource>
#include <stdlib.h>

namespace oiml {

	template<auto multiple, typename value_type = decltype(multiple)> OIML_INLINE constexpr value_type roundUpToMultiple(value_type value) noexcept {
		if constexpr ((multiple & (multiple - 1)) == 0) {
			constexpr auto mulSub1{ multiple - 1 };
			constexpr auto notMulSub1{ ~mulSub1 };
			return (value + (mulSub1)) & notMulSub1;
		} else {
			const auto remainder = value % multiple;
			return remainder == 0 ? value : value + (multiple - remainder);
		}
	}

	template<typename value_type_new, size_t alignment> class alloc_wrapper {
	  public:
		using value_type	  = value_type_new;
		using pointer		  = value_type_new*;
		using const_pointer	  = const value_type_new*;
		using reference		  = value_type_new&;
		using const_reference = const value_type_new&;
		using size_type		  = std::size_t;
		using difference_type = std::ptrdiff_t;		
		using allocator_traits = std::allocator_traits<alloc_wrapper<value_type, alignment>>;

		template<typename U> struct rebind {
			using other = alloc_wrapper<U, alignment>;
		};

		OIML_INLINE alloc_wrapper() noexcept = default;

		template<typename U> alloc_wrapper(const alloc_wrapper<U, alignment>&) noexcept {
		}

		OIML_INLINE static pointer allocate(size_type count) noexcept {
			if OIML_UNLIKELY (count == 0) {
				return nullptr;
			}
#if defined(OIML_WIN) || defined(OIML_LINUX)
			return static_cast<value_type*>(_mm_malloc(roundUpToMultiple<alignment>(count * sizeof(value_type)), alignment));
#else
			return static_cast<value_type*>(aligned_alloc(alignment, roundUpToMultiple<alignment>(count * sizeof(value_type))));
#endif
		}

		OIML_INLINE void deallocate(pointer ptr, size_t = 0) noexcept {
			if OIML_LIKELY (ptr) {
#if defined(OIML_WIN) || defined(OIML_LINUX)
				_mm_free(ptr);
#else
				free(ptr);
#endif
			}
		}

		template<typename... arg_types> OIML_INLINE static void construct(pointer ptr, arg_types&&... args) noexcept {
			new (ptr) value_type(std::forward<arg_types>(args)...);
		}

		OIML_INLINE static size_type maxSize() noexcept {
			return allocator_traits::max_size(alloc_wrapper{});
		}

		OIML_INLINE static void destroy(pointer ptr) noexcept {
			ptr->~value_type();
		}
	};

}