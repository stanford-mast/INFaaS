project(infaas-master)

find_package(Protobuf REQUIRED)

find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(AWSSDK REQUIRED COMPONENTS ec2 s3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories("${CMAKE_SOURCE_DIR}/src")

include_directories(/usr/local/include)
link_directories(/usr/local/lib /usr/local/lib64)

include_directories("${CMAKE_CURRENT_BINARY_DIR}/../../protos")

# Master library
set(inf-master_SOURCES queryfe_client.cc modelreg_client.cc)
add_library(inf-master SHARED ${inf-master_SOURCES})
target_link_libraries(inf-master redis-md infaas-protos
                      ${PROTOBUF_LIBRARY})
set(inf-master_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} /usr/local/include)
target_include_directories(inf-master PRIVATE ${inf-master_INCLUDES})

link_libraries(grpc++_unsecure grpc gpr)

add_executable(modelreg_server modelreg_server.cc)
add_executable(modelreg_heartbeat modelreg_heartbeat.cc)

add_executable(queryfe_server queryfe_server.cc)
add_executable(queryfe_heartbeat queryfe_heartbeat.cc)

add_executable(master_vm_daemon master_vm_daemon.cc)

target_link_libraries(modelreg_server inf-master ${AWSSDK_LINK_LIBRARIES})
target_link_libraries(modelreg_heartbeat inf-master)
target_link_libraries(queryfe_server inf-master inf-worker ${AWSSDK_LINK_LIBRARIES})
target_link_libraries(queryfe_heartbeat inf-master)

target_link_libraries(master_vm_daemon redis-md inf-worker ${AWSSDK_LINK_LIBRARIES})

set_target_properties(modelreg_server modelreg_heartbeat
    queryfe_server queryfe_heartbeat master_vm_daemon
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

