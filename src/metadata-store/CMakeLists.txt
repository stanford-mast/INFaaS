project(infaas-metadata)

# For libev and hiredis from redox
find_package(hiredis REQUIRED)
find_package(libev REQUIRED)

set(REDOX_LIB_DEPS
      ${HIREDIS_LIBRARIES}
      ${LIBEV_LIBRARIES}
      ${CMAKE_THREAD_LIBS_INIT})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(/usr/local/include)
link_directories(/usr/local/lib64)

set(redis-md_SOURCES redis_metadata.cc)
add_library(redis-md SHARED ${redis-md_SOURCES})
target_link_libraries(redis-md redox ${REDOX_LIB_DEPS})
set(redis-md_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} /usr/local/include)
target_include_directories(redis-md PRIVATE ${redis-md_INCLUDES})

add_executable(redis_md_test redis_md_test.cc redis_metadata.cc)
add_executable(redis_startup_helper redis_startup_helper.cc redis_metadata.cc)
target_link_libraries(redis_md_test redis-md)
target_link_libraries(redis_startup_helper redis-md)

set_target_properties(redis_md_test redis_startup_helper
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

