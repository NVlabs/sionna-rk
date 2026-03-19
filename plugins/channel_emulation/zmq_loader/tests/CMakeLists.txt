cmake_minimum_required(VERSION 3.18)

# Support multiple CUDA architectures: Orin (sm_87), Thor (sm_110), DGX Spark (sm_121)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "87;110;121")
endif()

project(cir_zmq_python LANGUAGES C CXX)

include(FetchContent)

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Find cJSON library - try pkg-config first, then fetch if not found
find_package(PkgConfig REQUIRED)
pkg_check_modules(cjson REQUIRED libcjson)

# Find ZMQ library
find_package(PkgConfig REQUIRED)
pkg_check_modules(zmq REQUIRED libzmq)

# Build the Python module
# Set language to CXX for cir_zmq.c since it contains C++ code when ENABLE_NANOBIND is defined
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../src/cir_zmq.c PROPERTIES LANGUAGE CXX)

nanobind_add_module(cir_zmq_py
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cir_zmq.c
)
target_compile_definitions(cir_zmq_py PRIVATE ENABLE_NANOBIND)
target_include_directories(cir_zmq_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${cjson_INCLUDE_DIRS}
    ${zmq_INCLUDE_DIRS}
)

target_link_libraries(cir_zmq_py PRIVATE ${cjson_LIBRARIES} ${zmq_LIBRARIES})
