cmake_minimum_required(VERSION 3.18)

# Support multiple CUDA architectures: Orin (sm_87), Thor (sm_110), DGX Spark (sm_121)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "87;110;121")
endif()

project(chn_emu_python LANGUAGES CXX CUDA)

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDAToolkit REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Build the Python module
nanobind_add_module(chn_emu_cuda ${CMAKE_CURRENT_SOURCE_DIR}/../src/chn_emu_cuda.cu)
target_compile_definitions(chn_emu_cuda PRIVATE ENABLE_NANOBIND)
target_include_directories(chn_emu_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_include_directories(chn_emu_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Link cuBLAS
target_link_libraries(chn_emu_cuda PRIVATE CUDA::cublas CUDA::cudart)
