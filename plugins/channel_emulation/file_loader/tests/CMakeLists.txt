cmake_minimum_required(VERSION 3.18)
project(cir_file_python LANGUAGES C CXX)

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
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(cjson QUIET libcjson)
endif()

if(NOT cjson_FOUND)
    message(STATUS "cJSON not found via pkg-config, fetching from GitHub...")
    FetchContent_Declare(
        cjson
        GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
        GIT_TAG v1.7.18
    )
    # Disable cJSON tests and examples
    set(ENABLE_CJSON_TEST OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_AND_STATIC_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(cjson)
    set(CJSON_TARGET cjson)
else()
    message(STATUS "Found cJSON via pkg-config")
    set(CJSON_TARGET ${cjson_LIBRARIES})
endif()

# Build the Python module
# Set language to CXX for cir_file.c since it contains C++ code when ENABLE_NANOBIND is defined
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../src/cir_file.c PROPERTIES LANGUAGE CXX)

nanobind_add_module(cir_file_py
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cir_file.c
)
target_compile_definitions(cir_file_py PRIVATE ENABLE_NANOBIND)
target_include_directories(cir_file_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${cjson_INCLUDE_DIRS}
)

if(cjson_FOUND)
    target_link_libraries(cir_file_py PRIVATE ${cjson_LIBRARIES})
else()
    target_link_libraries(cir_file_py PRIVATE cjson)
endif()

