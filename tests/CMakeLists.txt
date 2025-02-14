cmake_minimum_required(VERSION 3.18)

project(
  "oiml_test"
  VERSION "1.0.0"
  LANGUAGES CXX
)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/Install/")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_executable(
  "oiml_test" 
  "./test_main.cpp"
)

target_link_libraries(
	"oiml_test"
	PUBLIC 
	oiml::oiml
)

target_compile_options(
	"oiml_test" PUBLIC
	"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:/fsanitize=address>>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wnull-dereference>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wuninitialized>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wconversion>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wpedantic>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wshadow>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wextra>"
	"$<$<CXX_COMPILER_ID:CLANG>:-Wall>"

	
	"$<$<CXX_COMPILER_ID:GNU>:-Wnull-dereference>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wuninitialized>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wconversion>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wpedantic>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wshadow>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wextra>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wall>"

	"$<$<CXX_COMPILER_ID:MSVC>:/Wall>"
	"$<$<CXX_COMPILER_ID:MSVC>:/W4>"
)

target_link_options(
	"oiml_test" PUBLIC
	"$<$<CXX_COMPILER_ID:GNU>:$<$<STREQUAL:${ASAN_ENABLED},TRUE>:-fsanitize=address>>"
)

target_compile_definitions(
	"oiml_test" PUBLIC 
	"BASE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../\""
	"oiml_COMMIT=\"${oiml_COMMIT_HASH}\""
	"SIMDJSON_COMMIT=\"${SIMDJSON_COMMIT_HASH}\""
	"GLAZE_COMMIT=\"${GLAZE_COMMIT_HASH}\""
	"OPERATING_SYSTEM_NAME=\"${CMAKE_SYSTEM_NAME}\""
	"OPERATING_SYSTEM_VERSION=\"${CMAKE_SYSTEM_VERSION}\""
	"COMPILER_ID=\"${CMAKE_CXX_COMPILER_ID}\""
	"COMPILER_VERSION=\"${CMAKE_CXX_COMPILER_VERSION}\""
)

if (WIN32)
	install(
		FILES 
		"$<TARGET_PDB_FILE:oiml_test>"
		DESTINATION "bin"
		OPTIONAL
	)
endif()

install(
	FILES 
	"$<TARGET_FILE:oiml_test>"
	DESTINATION "bin"
	OPTIONAL
)
