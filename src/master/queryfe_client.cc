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
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "queryfe_client.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

static const int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;
static const int MAX_GRPC_DEADLINE = 10000;  // In msec = 10 seconds.

namespace infaaspublic {
namespace infaasqueryfe {
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

QueryFEClient::QueryFEClient(std::shared_ptr<Channel> channel)
    : stub_(Query::NewStub(channel)) {}

std::vector<std::string> QueryFEClient::QueryOnline(
    const std::vector<std::string>& input, const std::string& grandparent_model,
    const std::string& parent_model, const std::string& model_variant,
    const std::string& submitter, const int64_t& latency, const double& minacc,
    const double& maxcost) {
  struct timeval time1, time2;
  gettimeofday(&time1, NULL);
  // Data we are sending to the server.
  QuerySLO query_slo;
  query_slo.set_latencyinusec(latency);
  query_slo.set_minaccuracy(minacc);
  query_slo.set_maxcost(maxcost);

  QueryOnlineRequest request;
  for (auto inp : input) { request.add_raw_input(inp); }

  request.set_grandparent_model(grandparent_model);
  request.set_parent_model(parent_model);
  request.set_model_variant(model_variant);
  request.set_submitter(submitter);
  request.mutable_slo()->CopyFrom(query_slo);

  // Container for the data we expect from the server.
  QueryOnlineResponse reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;
  set_grpc_deadline(&context);

  gettimeofday(&time2, NULL);
  printf("[queryfe_client.cc] QueryOnline prepare data  %.4lf ms.\n",
         ts_to_ms(time1, time2));
  // The actual RPC.
  Status status = stub_->QueryOnline(&context, request, &reply);

  gettimeofday(&time1, NULL);
  printf("[queryfe_client.cc] QueryOnline master fe end-to-end  %.4lf ms.\n",
         ts_to_ms(time2, time1));
  // Act upon its status.
  if (status.ok()) {
    RequestReplyEnum reply_status = reply.status().status();
    if (reply_status != RequestReplyEnum::SUCCESS) {
      std::cout << "INFaaS reply error status: " << reply.status().msg()
                << std::endl;
      return {reply.status().msg()};
    }

    auto reply_output = reply.raw_output();
    std::vector<std::string> return_out(reply_output.begin(),
                                        reply_output.end());
    return return_out;
  } else {
    std::cout << "RPC error code: " << status.error_code() << ": "
              << status.error_message() << std::endl;

    // Return vector with error message to signal failure
    return {status.error_message()};
  }
}

RequestReply QueryFEClient::QueryOffline(const std::string& input_url,
                                         const std::string& model,
                                         const std::string& submitter,
                                         const std::string& output_url,
                                         const double& maxcost) {
  // Data we are sending to the server.
  QueryOfflineRequest request;
  request.set_input_url(input_url);
  request.set_model(model);
  request.set_submitter(submitter);
  request.set_output_url(output_url);
  request.set_maxcost(maxcost);

  // Container for the data we expect from the server.
  QueryOfflineResponse reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;
  set_grpc_deadline(&context);

  // The actual RPC.
  Status status = stub_->QueryOffline(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
    RequestReply rs;
    rs.set_status(RequestReplyEnum::INVALID);
    rs.set_msg(status.error_message());
    return rs;
  }
}

AllParReply QueryFEClient::AllParentInfo(const std::string& task,
                                         const std::string& dataset) {
  // Data we are sending to the server.
  AllParRequest request;
  request.set_task(task);
  request.set_dataset(dataset);

  // Container for the data we expect from the server.
  AllParResponse reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->AllParentInfo(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.reply();
  } else {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
    AllParReply failed_request;
    RequestReply rs;
    rs.set_status(RequestReplyEnum::INVALID);
    rs.set_msg(status.error_message());
    failed_request.mutable_status()->CopyFrom(rs);
    return failed_request;
  }
}

QueryModelReply QueryFEClient::QueryModelInfo(const std::string& model) {
  // Data we are sending to the server.
  QueryModelInfoRequest request;
  request.set_model(model);

  // Container for the data we expect from the server.
  QueryModelInfoResponse reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->QueryModelInfo(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.reply();
  } else {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
    QueryModelReply failed_request;
    RequestReply rs;
    rs.set_status(RequestReplyEnum::INVALID);
    rs.set_msg(status.error_message());
    failed_request.mutable_status()->CopyFrom(rs);
    return failed_request;
  }
}

RequestReply QueryFEClient::Heartbeat() {
  // Data we are sending to the server.
  HeartbeatRequest request;
  RequestReply rs;
  rs.set_status(RequestReplyEnum::SUCCESS);
  request.mutable_status()->CopyFrom(rs);

  // Container for the data we expect from the server.
  HeartbeatResponse reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->Heartbeat(&context, request, &reply);

  // Act upon its status.
  if (status.ok()) {
    return reply.status();
  } else {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
    rs.set_status(RequestReplyEnum::INVALID);
    rs.set_msg(status.error_message());
    return rs;
  }
}

}  // namespace infaasqueryfe
}  // namespace infaaspublic
