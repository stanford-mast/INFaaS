INCLUDE(GRPCGenerateCPP)

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
  request_reply.proto
  modelreg.proto
  queryfe.proto
)

PROTOBUF_GENERATE_GRPC_CPP(PROTO_GRPC_SRCS PROTO_GRPC_HDRS
  modelreg.proto
  queryfe.proto
)

add_library(infaas-protos
  ${PROTO_SRCS}
  ${PROTO_GRPC_SRCS}
)

target_include_directories(infaas-protos PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
)

add_subdirectory(internal)
add_subdirectory(trtis)
