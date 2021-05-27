/*
 * Copyright 2018-2021 Board of Trustees of Stanford University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <sys/time.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "query_client.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

static const int MAX_GRPC_DEADLINE = 10000;  // In msec = 10 seconds.

namespace infaas {
namespace internal {
namespace {
// Timestamp to millisecond duration.
double ts_to_ms(const struct timeval& start, const struct timeval& end) {
  return (end.tv_sec - start.tv_sec) * 1000.0 +
         (end.tv_usec - start.tv_usec) / 1000.0;
}

void set_grpc_deadline(ClientContext* context,
                       int ddl_in_ms = MAX_GRPC_DEADLINE) {
  std::chrono::system_clock::time_point deadline =
      std::chrono::system_clock::now() + std::chrono::milliseconds(ddl_in_ms);
  context->set_deadline(deadline);
}

}  // namespace

// Translate the function call to gRPC call.
InfaasRequestStatus QueryClient::QueryOnline(
    const google::protobuf::RepeatedPtrField<std::string>& input,
    const std::vector<std::string>& model, const std::string submitter,
    google::protobuf::RepeatedPtrField<std::string>* output,
    const int64_t& latency, const double& minacc, const double& maxcost,
    const int grpc_deadline) {
  struct timeval time1, time2;
  gettimeofday(&time1, NULL);
  // Data we are sending to the server.
  QuerySLO query_slo;
  query_slo.set_latencyinusec(latency);
  query_slo.set_minaccuracy(minacc);
  query_slo.set_maxcost(maxcost);

  QueryOnlineRequest request;

  request.mutable_raw_input()->CopyFrom(input);
  for (auto m : model) { request.add_model(m); }
  request.mutable_slo()->CopyFrom(query_slo);
  request.set_submitter(submitter);

  QueryOnlineResponse reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;
  set_grpc_deadline(&context, grpc_deadline);

  gettimeofday(&time2, NULL);
  printf("[query_client.cc] Prepare QueryOnline request: %.4lf ms.\n",
         ts_to_ms(time1, time2));
  // The actual RPC.
  Status status = stub_->QueryOnline(&context, request, &reply);

  gettimeofday(&time1, NULL);
  printf("[query_client.cc] QueryOnline total: %.4lf ms.\n",
         ts_to_ms(time2, time1));

  // Act upon its status.
  InfaasRequestStatus request_status;
  if (status.ok() &&
      (reply.status().status() == InfaasRequestStatusEnum::SUCCESS)) {
    *output = reply.raw_output();

    gettimeofday(&time2, NULL);
    printf("[query_client.cc] get output: %.4lf ms.\n", ts_to_ms(time1, time2));
    fflush(stdout);
    return reply.status();
  } else if (status.error_code() == grpc::StatusCode::INVALID_ARGUMENT) {
    // Internal error.
    std::string errmsg = "INTERNAL FAILURE: " + reply.status().msg();
    std::cerr << errmsg << std::endl;
    request_status.set_status(InfaasRequestStatusEnum::INVALID);
    request_status.set_msg(errmsg);
    return request_status;
  } else {
    // RPC error
    std::string errmsg = "RPC FAILURE: " + status.error_message();
    std::cerr << errmsg << std::endl;
    request_status.set_status(InfaasRequestStatusEnum::UNAVAILABLE);
    request_status.set_msg(errmsg);
    return request_status;
  }
}

InfaasRequestStatus QueryClient::QueryOffline(
    const std::string& input_url, const std::vector<std::string>& model,
    const std::string submitter, const std::string& output_url,
    const double& maxcost, const int grpc_deadline) {
  // Data we are sending to the server.
  QuerySLO query_slo;
  // query_slo.set_latencyinusec(latency);
  // query_slo.set_minaccuracy(minacc);
  query_slo.set_maxcost(maxcost);

  QueryOfflineRequest request;
  request.set_input_url(input_url);
  for (auto m : model) { request.add_model(m); }
  request.set_output_url(output_url);
  request.mutable_slo()->CopyFrom(query_slo);
  request.set_submitter(submitter);

  QueryOfflineResponse reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;
  set_grpc_deadline(&context, grpc_deadline);

  // The actual RPC.
  Status status = stub_->QueryOffline(&context, request, &reply);

  // Act upon its status.
  InfaasRequestStatus request_status;
  if (status.ok() &&
      (reply.status().status() == InfaasRequestStatusEnum::SUCCESS)) {
    return reply.status();
  } else if (status.error_code() == grpc::StatusCode::INVALID_ARGUMENT) {
    // Internal error.
    std::string errmsg = "INTERNAL FAILURE: " + reply.status().msg();
    std::cerr << errmsg << std::endl;
    request_status.set_status(InfaasRequestStatusEnum::INVALID);
    request_status.set_msg(errmsg);
    return request_status;
  } else {
    // RPC error
    std::string errmsg = "RPC FAILURE: " + status.error_message();
    std::cerr << errmsg << std::endl;
    request_status.set_status(InfaasRequestStatusEnum::UNAVAILABLE);
    request_status.set_msg(errmsg);
    return request_status;
  }
}

InfaasRequestStatus QueryClient::Heartbeat() {
  HeartbeatRequest request;
  InfaasRequestStatus request_status;
  request_status.set_status(InfaasRequestStatusEnum::SUCCESS);
  request.mutable_status()->CopyFrom(request_status);

  HeartbeatResponse reply;

  ClientContext context;

  // The actual RPC.
  Status status = stub_->Heartbeat(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    std::cerr << "Failed to connect, error code ";
    std::cerr << status.error_code() << ": " << status.error_message()
              << std::endl;
    InfaasRequestStatus request_status;
    request_status.set_status(InfaasRequestStatusEnum::INVALID);
    request_status.set_msg(status.error_message());
    return request_status;
  }
}

}  // namespace internal
}  // namespace infaas
