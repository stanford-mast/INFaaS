INCLUDE(GRPCGenerateCPP)

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
  query.proto
  sys_monitor.proto
  infaas_request_status.proto
)

PROTOBUF_GENERATE_GRPC_CPP(PROTO_GRPC_SRCS PROTO_GRPC_HDRS
  query.proto
  sys_monitor.proto
)

add_library(infaas-protos-internal
  ${PROTO_SRCS}
  ${PROTO_GRPC_SRCS}
)

target_include_directories(infaas-protos-internal PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
)
