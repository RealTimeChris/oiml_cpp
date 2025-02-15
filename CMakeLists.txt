#	MIT License
#
#	Copyright (c) 2025 OpenInfer
#
#	Permission is hereby granted, free of charge, to any person obtaining a copy of this 
#	software and associated documentation files (the "Software"), to deal in the Software 
#	without restriction, including without limitation the rights to use, copy, modify, merge, 
#	publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
#	persons to whom the Software is furnished to do so, subject to the following conditions:
#
#	The above copyright notice and this permission notice shall be included in all copies or 
#	substantial portions of the Software.
#
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
#	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
#	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
#	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
#	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
#	DEALINGS IN THE SOFTWARE.
#
# CMakeLists.txt - The CMake script for building this library.

cmake_minimum_required(VERSION 3.18)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/CMake;")

set(PROJECT_NAME "oiml")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(
  "${PROJECT_NAME}"
  VERSION "1.0.0"
  LANGUAGES CXX
)

add_library("${PROJECT_NAME}" INTERFACE)

add_library("${PROJECT_NAME}::${PROJECT_NAME}" ALIAS "${PROJECT_NAME}")

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/oiml/*.hpp")

 set_target_properties(
     "${PROJECT_NAME}" PROPERTIES
     OUTPUT_NAME "oiml"
     CXX_STANDARD_REQUIRED ON
     CXX_EXTENSIONS OFF
 )
 
 target_include_directories(
     "${PROJECT_NAME}" INTERFACE
     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:include/>"
 )

target_compile_features("${PROJECT_NAME}" INTERFACE "cxx_std_20")
 
target_compile_options(
	"${PROJECT_NAME}" INTERFACE
	"$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
	"$<$<CXX_COMPILER_ID:MSVC>:/Oi;/Ot;/GT;/GL;/fp:fast;/permissive->"
	"$<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>:$<$<CXX_COMPILER_ID:Clang>:-mavx2;-mavx;-mfma;-msse4.2>>"
	"$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>"
    "$<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>:$<$<CXX_COMPILER_ID:GNU>:-mavx2;-mavx;-mfma;-msse4.2>>"
)
 
target_link_options(
    "${PROJECT_NAME}" INTERFACE
    "$<$<AND:$<STREQUAL:$<UPPER_CASE:${UBSAN}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
    "$<$<AND:$<CXX_COMPILER_ID:CLANG>,$<STREQUAL:${ASAN_ENABLED},TRUE>>:-fsanitize=address>"
	"$<$<AND:$<CXX_COMPILER_ID:Clang>,$<STREQUAL:${ASAN_ENABLED},TRUE>>:-fsanitize=address>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:${ASAN_ENABLED},TRUE>>:-fsanitize=address>"
    "$<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO;/OPT:ICF;/OPT:REF;/LTCG>"
)

target_compile_definitions(
	"${PROJECT_NAME}" INTERFACE 
	"$<$<STREQUAL:${ASAN_ENABLED},TRUE>:ASAN_ENABLED>"
)

set(CONFIG_FILE_NAME "${PROJECT_NAME}Config.cmake")
set(EXPORTED_TARGETS_NAME "${PROJECT_NAME}Targets")
set(EXPORTED_TARGETS_FILE_NAME "${EXPORTED_TARGETS_NAME}.cmake")
set(EXPORTED_TARGETS_FILE_PATH "share/oiml/${EXPORTED_TARGETS_FILE_NAME}")

include(CMakePackageConfigHelpers)
configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/CMake/${CONFIG_FILE_NAME}.in"
	"${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
	INSTALL_DESTINATION "share/oiml"
	PATH_VARS
	EXPORTED_TARGETS_FILE_PATH
)

set(VERSION_FILE_NAME "${PROJECT_NAME}ConfigVersion.cmake")

write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
	VERSION "${PRODUCT_VERSION}"
	COMPATIBILITY AnyNewerVersion
)

install(
	FILES
	"${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
	"${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
	DESTINATION "share/oiml"
)

install(
	DIRECTORY
	"${CMAKE_CURRENT_SOURCE_DIR}/include/"
	DESTINATION "include"
)

install(
	TARGETS "${PROJECT_NAME}"
	EXPORT "${EXPORTED_TARGETS_NAME}"
)

install(
	EXPORT "${EXPORTED_TARGETS_NAME}"
	FILE "${EXPORTED_TARGETS_FILE_NAME}"
	NAMESPACE "${PROJECT_NAME}::"
	DESTINATION "share/oiml"
)

if("${OIML_CPP_TESTS}" STREQUAL "TRUE")
	add_subdirectory("./tests")
endif()
