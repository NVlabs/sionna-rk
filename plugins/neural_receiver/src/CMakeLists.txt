set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# find TensorRT package
find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h PATHS ${tensorrt_ROOT} /usr/include /usr/local/TensorRT PATH_SUFFIXES include)
find_library(TensorRT_LIBRARY NAMES nvinfer PATHS ${tensorrt_ROOT} /usr/lib /usr/local/TensorRT PATH_SUFFIXES lib)
#find_library(TensorRT_PARSER_LIBRARY NAMES nvparsers PATHS ${tensorrt_ROOT} /usr/lib /usr/local/TensorRT PATH_SUFFIXES lib)
find_library(TensorRT_ONNX_LIBRARY NAMES nvonnxparser PATHS ${tensorrt_ROOT} /usr/lib /usr/local/TensorRT PATH_SUFFIXES lib)
if (NOT TensorRT_INCLUDE_DIR OR NOT TensorRT_LIBRARY OR NOT TensorRT_ONNX_LIBRARY)
  message(FATAL_ERROR "TensorRT installation missing ${TensorRT_INCLUDE_DIR} ${TensorRT_LIBRARY} ${TensorRT_PARSER_LIBRARY} ${TensorRT_ONNX_LIBRARY}")
endif ()

# receiver libraries

set(PHY_RECEIVER_TRT_RUNTIME_SRC
  runtime/data_processing.cu
  runtime/trt_receiver.cpp
)

cuda_add_library( receiver_trt MODULE ${PHY_RECEIVER_TRT_RUNTIME_SRC}
  nr_receiver_trt.c
)
set_target_properties(receiver_trt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(receiver_trt PRIVATE ${TensorRT_INCLUDE_DIR})
target_compile_options(receiver_trt PRIVATE -ftls-model=global-dynamic)
target_compile_options(receiver_trt PRIVATE -Wno-deprecated-declarations -Wno-error=unused-variable -Wno-error=unused-function -Wno-error=unused-result)
target_link_options(receiver_trt PRIVATE -ftls-model=global-dynamic)
target_link_libraries(receiver_trt ${TensorRT_LIBRARY} pthread)

add_dependencies(nr-uesoftmodem receiver_trt)
add_dependencies(nr-softmodem receiver_trt)
add_dependencies(receiver_trt generate_T)

# capture
set(PHY_RECEIVER_CAPTURE_SRC
  nr_receiver_capture.c
)

add_library(receiver_capture MODULE ${PHY_RECEIVER_CAPTURE_SRC})
target_compile_options(receiver_capture PRIVATE -Wno-deprecated-declarations -Wno-error=unused-variable -Wno-error=unused-result)
target_link_libraries(receiver_capture PRIVATE pthread)

add_dependencies(nr-softmodem receiver_capture)
add_dependencies(nr-uesoftmodem receiver_capture)
add_dependencies(receiver_capture generate_T)
