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

#include <iostream>
#include <memory>
#include <string>

#include "modelreg_client.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

namespace infaaspublic {
namespace infaasmodelreg {

ModelRegClient::ModelRegClient(std::shared_ptr<Channel> channel)
    : stub_(ModelReg::NewStub(channel)) {}

RequestReply ModelRegClient::RegisterModel(
    const std::string& submitter, const std::string& grandparent_model,
    const std::string& parent_model, const std::string& first_variant,
    const std::string& url, const std::string& dataset, const double& accuracy,
    const std::string& framework, const std::string& task,
    const std::string& config_name) {
  // Data we are sending to the server.
  ModelRegRequest request;
  request.set_submitter(submitter);
  request.set_grandparent_model(grandparent_model);
  request.set_parent_model(parent_model);
  request.set_first_variant(first_variant);
  request.set_url(url);
  request.set_dataset(dataset);
  request.set_accuracy(accuracy);
  request.set_framework(framework);
  request.set_task(task);
  request.set_config_name(config_name);

  // Container for the data we expect from the server.
  ModelRegResponse reply;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  ClientContext context;

  // The actual RPC.
  Status status = stub_->RegisterModel(&context, request, &reply);

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

RequestReply ModelRegClient::Heartbeat() {
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

}  // namespace infaasmodelreg
}  // namespace infaaspublic
