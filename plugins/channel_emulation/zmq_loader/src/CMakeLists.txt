set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Add loader source to main executable
set(PLUGINS_SRC ${PLUGINS_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/cir_zmq_load.c PARENT_SCOPE)

# original
set(PHY_CIR_ZMQ_SRC
  cir_zmq.c
)

add_library(cir_zmq MODULE ${PHY_CIR_ZMQ_SRC})

# Find required libraries
pkg_check_modules(cjson REQUIRED libcjson)
target_link_libraries(cir_zmq PRIVATE zmq ${cjson_LIBRARIES})
target_include_directories(cir_zmq PRIVATE ${cjson_INCLUDE_DIRS})

add_dependencies(nr-softmodem cir_zmq)
add_dependencies(cir_zmq generate_T)
