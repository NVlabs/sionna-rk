cmake_minimum_required(VERSION 3.18)

# Support multiple CUDA architectures: Orin (sm_87), Thor (sm_110), DGX Spark (sm_121)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "87;110;121")
endif()

project(ldpc_decoder_python LANGUAGES CXX CUDA)

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Create a patched copy with CUDA graphs disabled for testing
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/ldpc_decoder.cu DECODER_SOURCE)
string(REPLACE "#define USE_GRAPHS" "//#define USE_GRAPHS  // disabled for testing" DECODER_SOURCE "${DECODER_SOURCE}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/ldpc_decoder_test.cu "${DECODER_SOURCE}")

nanobind_add_module(ldpc_decoder ${CMAKE_CURRENT_BINARY_DIR}/ldpc_decoder_test.cu)
target_compile_definitions(ldpc_decoder PRIVATE ENABLE_NANOBIND)
target_include_directories(ldpc_decoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

