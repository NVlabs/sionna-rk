set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Add loader source to main executable
set(PLUGINS_SRC ${PLUGINS_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/cir_file_load.c PARENT_SCOPE)

# CIR file plugin source files
set(CIR_FILE_SRC
  cir_file.c
)

add_library(cir_file MODULE ${CIR_FILE_SRC})

# Find required libraries (cJSON for parsing config)
pkg_check_modules(cjson REQUIRED libcjson)
target_link_libraries(cir_file PRIVATE ${cjson_LIBRARIES})
target_include_directories(cir_file PRIVATE ${cjson_INCLUDE_DIRS})

add_dependencies(nr-softmodem cir_file)
add_dependencies(cir_file generate_T)
