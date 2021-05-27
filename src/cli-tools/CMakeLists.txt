project(infaas-cli)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV 2.4 REQUIRED)

include_directories("${CMAKE_SOURCE_DIR}/src")

include_directories(/usr/local/include)
link_directories(/usr/local/lib /usr/local/lib64)
link_libraries(grpc++_unsecure grpc gpr)

add_executable(infaas_modarch infaas_modarch.cc)
add_executable(infaas_modinfo infaas_modinfo.cc)
add_executable(infaas_online_query infaas_online_query.cc)
add_executable(infaas_offline_query infaas_offline_query.cc)
add_executable(infaas_modelregistration infaas_modelregistration.cc)
target_link_libraries(infaas_modarch inf-master)
target_link_libraries(infaas_modinfo inf-master)
target_link_libraries(infaas_online_query inf-master ${OpenCV_LIBS})
target_link_libraries(infaas_offline_query inf-master)
target_link_libraries(infaas_modelregistration inf-master)

set_target_properties(infaas_modarch infaas_modinfo infaas_online_query
    infaas_offline_query infaas_modelregistration
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

