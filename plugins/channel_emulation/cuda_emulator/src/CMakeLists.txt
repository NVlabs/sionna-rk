set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Add loader source to main executable
set(PLUGINS_SRC ${PLUGINS_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/chn_emu_load.c PARENT_SCOPE)

set(PHY_CHN_EMU_SRC
  chn_emu_cuda.cu
  chn_emu.c
)

# Use the modern CUDAToolkit finder instead of deprecated FindCUDA
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

if (ENABLE_DGX_OPTIMIZATIONS)
  add_compile_definitions(ENABLE_DGX_OPTIMIZATIONS)
endif ()

add_library(chn_emu MODULE ${PHY_CHN_EMU_SRC})
if (ENABLE_DGX_OPTIMIZATIONS)
  set_target_properties(chn_emu PROPERTIES CUDA_ARCHITECTURES "120")
else ()
  set_target_properties(chn_emu PROPERTIES CUDA_ARCHITECTURES "87")
endif ()
set_target_properties(chn_emu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Add CUDA libraries using modern targets
if(TARGET CUDAToolkit::cublas)
  target_link_libraries(chn_emu PRIVATE CUDAToolkit::cublas)
else()
  find_library(CUBLAS_LIBRARY cublas HINTS ${CUDAToolkit_LIBRARY_DIR})
  if(CUBLAS_LIBRARY)
    target_link_libraries(chn_emu PRIVATE ${CUBLAS_LIBRARY})
  endif()
endif()

add_dependencies(nr-softmodem chn_emu)
add_dependencies(chn_emu generate_T)
