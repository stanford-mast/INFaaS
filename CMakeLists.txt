cmake_minimum_required(VERSION 3.7)

project(infaas-root)

# Only build worker
option(BUILD_ONLY_WORKER "Only build Worker" OFF)
option(BUILD_ONLY_MD "Only build Metadata Store" OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE   ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CMake path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Dependencies
# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf REQUIRED)

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
# find_package(gRPC REQUIRED)

set(_GRPC_GRPCPP_UNSECURE grpc++_unsecure)
set(_GRPC_CPP_PLUGIN_EXECUTABLE /usr/local/bin/grpc_cpp_plugin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Targets
add_subdirectory(protos)
add_subdirectory(src)

