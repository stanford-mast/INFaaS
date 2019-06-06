INCLUDE(GRPCGenerateCPP)

###### For TensorRT clients
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
  "grpc_service.proto"
  "api.proto"
  "model_config.proto"
  "request_status.proto"
  "server_status.proto"
)

PROTOBUF_GENERATE_GRPC_CPP(PROTO_GRPC_SRCS PROTO_GRPC_HDRS
  "grpc_service.proto"
)

add_library(nv-trtis-protos
  ${PROTO_SRCS}
  ${PROTO_GRPC_SRCS}
)

target_include_directories(nv-trtis-protos PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)
