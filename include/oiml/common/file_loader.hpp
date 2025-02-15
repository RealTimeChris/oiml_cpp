/*
	MIT License	

	Copyright (c) 2025 OpenInfer

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
/// Feb 3, 2023
#pragma once

#include <oiml/common/allocator.hpp>
#include <oiml/common/array.hpp>
#include <filesystem>
#include <fstream>

namespace oiml {

	template<typename value_type, size_t alignment> using vector = std::vector<value_type, alloc_wrapper<value_type, alignment>>;

	template<size_t alignment> class file_loader {
	  public:
		constexpr file_loader() {};

		static vector<char, alignment> loadFile(const std::string& filePath) {
			std::string directory{ filePath.substr(0, filePath.find_last_of("/") + 1) };
			if (!std::filesystem::exists(directory)) {
				std::filesystem::create_directories(directory);
			}

			if (!std::filesystem::exists(filePath)) {
				std::ofstream createFile{ filePath, std::ios::binary };
				createFile.close();
			}

			std::ifstream theStream{ filePath, std::ios::binary | std::ios::ate };
			if (!theStream) {
				throw std::runtime_error("Failed to open file.");
			}

			std::streamsize size = theStream.tellg();
			theStream.seekg(0, std::ios::beg);

			vector<char, alignment> buffer(size);
			if (!theStream.read(buffer.data(), size)) {
				throw std::runtime_error("Failed to read file.");
			}

			theStream.close();
			return buffer;
		}

		static void saveFile(const std::string& fileToSave, const std::string& filePath, bool retry = true) {
			std::ofstream theStream{ filePath.data(), std::ios::binary | std::ios::out | std::ios::trunc };
			theStream.write(fileToSave.data(), static_cast<int64_t>(fileToSave.size()));
			if (theStream.is_open()) {
				std::cout << "File succesfully written to: " << filePath << std::endl;
			} else {
				std::string directory{ filePath.substr(0, filePath.find_last_of("/") + 1) };
				if (!std::filesystem::exists(directory) && retry) {
					std::filesystem::create_directories(directory);
					return saveFile(fileToSave, filePath, false);
				}
				std::cerr << "File failed to be written to: " << filePath << std::endl;
			}
			theStream.close();
		}
	};

}