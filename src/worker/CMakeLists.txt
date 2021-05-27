project(infaas-worker)

# Worker directory
find_package(Protobuf REQUIRED)

find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenCV 2.4 REQUIRED)
find_package(AWSSDK REQUIRED COMPONENTS s3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(PROTOS_PATH "${CMAKE_CURRENT_BINARY_DIR}/../../protos")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Test whether this worker has GPU or not
set(_has_nvidia OFF)
if (EXISTS "/proc/driver/nvidia/version")
  message(STATUS "Worker found GPU driver!")
  set(_has_nvidia ON)
  add_definitions(-DINFAAS_GPU_WORKER)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Test whether this worker has Inferentia or not
set(_has_inferentia OFF)
if (EXISTS "/run/infa/")
  message(STATUS "Worker found Inferentia Neuron driver!")
  set(_has_inferentia ON)
  add_definitions(-DINFAAS_NEURON_WORKER)
endif()

include_directories("${CMAKE_SOURCE_DIR}/src")
include_directories(${PROTOS_PATH})

include_directories(/usr/local/include)
link_directories(/usr/local/lib /usr/local/lib64)

# Library for Master
set(inf-worker_SOURCES query_client.cc)
add_library(inf-worker SHARED ${inf-worker_SOURCES})
target_link_libraries(inf-worker redis-md
                      infaas-protos-internal ${PROTOBUF_LIBRARY})
set(inf-worker_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} /usr/local/include)
target_include_directories(inf-worker PRIVATE ${inf-worker_INCLUDES})

# Library for TRTIS
add_library(trtis-request SHARED trtis_request.cc trtis_model_config.cc)
target_link_libraries(trtis-request nv-trtis-protos ${PROTOBUF_LIBRARY} curl grpc++_unsecure grpc gpr)
set(trtis-request_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} /usr/local/include
    ${PROTOS_PATH})
target_include_directories(trtis-request PRIVATE ${trt-request_INCLUDES})

# Library for worker internal use
set(worker-util_SOURCES common_model_util.cc autoscaler.cc)
# TODO: need to investicate why worker-util cannob be a SHARED lib.
# Now it would get link error if we set it to SHARED.
add_library(worker-util STATIC ${worker-util_SOURCES})
target_link_libraries(worker-util trtis-request redis-md infaas-protos-internal
                      inf-worker ${PROTOBUF_LIBRARY} ${AWSSDK_LINK_LIBRARIES})
set(worker-util_INCLUDES ${CMAKE_CURRENT_BINARY_DIR} /usr/local/include)
target_include_directories(worker-util PRIVATE ${worker-util_INCLUDES})

# Build targets and link libraries
link_libraries(grpc++_unsecure grpc gpr pthread)
# TRTIS related executables
add_executable(trtis_simple_client trtis_simple_client.cc)
add_executable(trtis_perf_client trtis_perf_client.cc)

target_link_libraries(trtis_simple_client trtis-request)
target_link_libraries(trtis_perf_client trtis-request)

# INFaaS related executables
add_executable(query_executor query_executor.cc)
add_executable(query_heartbeat query_heartbeat.cc)

# Only for GPU worker
if (_has_nvidia)
  cuda_add_executable(gpu_daemon gpu_daemon.cc OPTIONS -std=c++11)
  target_link_libraries(gpu_daemon redis-md trtis-request)
  set_target_properties(gpu_daemon
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

endif(_has_nvidia)

if(BUILD_ONLY_WORKER)
  add_executable(query_test query_test.cc)
  add_executable(gnmt_query_test gnmt_query_test.cc)
  target_link_libraries(gnmt_query_test inf-worker)
  target_link_libraries(query_test worker-util ${OpenCV_LIBS})
  set_target_properties(query_test gnmt_query_test
      PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
else(BUILD_ONLY_WORKER)
  message(STATUS "Skipping Worker Tests")
endif(BUILD_ONLY_WORKER)

# Only compile for Inferentia workers
if (_has_inferentia)
  add_executable(inferentia_query_test inferentia_query_test.cc)
  target_link_libraries(inferentia_query_test inf-worker ${OpenCV_LIBS})
  set_target_properties(inferentia_query_test
      PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

endif(_has_inferentia)

target_link_libraries(query_heartbeat inf-worker)
target_link_libraries(query_executor worker-util ${AWSSDK_LINK_LIBRARIES})

set_target_properties(trtis_simple_client trtis_perf_client query_executor query_heartbeat
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

