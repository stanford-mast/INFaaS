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

// This file should be used by the master to send queries to executors.
#ifndef QUERY_CLIENT_H
#define QUERY_CLIENT_H

#include <cstdint>
#include <string>
#include <vector>

#include <google/protobuf/repeated_field.h>
#include <grpcpp/grpcpp.h>
#include "internal/query.grpc.pb.h"

using grpc::Channel;

namespace infaas {
namespace internal {

class QueryClient {
public:
  QueryClient(std::shared_ptr<Channel> channel)
      : stub_(Query::NewStub(channel)) {}

  // QueryOnline request
  InfaasRequestStatus QueryOnline(
      const google::protobuf::RepeatedPtrField<std::string>& input,
      const std::vector<std::string>& model, const std::string submitter,
      google::protobuf::RepeatedPtrField<std::string>* output,
      const int64_t& latency = 0, const double& minacc = 0,
      const double& maxcost = 0, const int grpc_deadline = 10000);

  // QueryOffline request
  InfaasRequestStatus QueryOffline(const std::string& input_url,
                                   const std::vector<std::string>& model,
                                   const std::string submitter,
                                   const std::string& output_url,
                                   const double& maxcost = 0,
                                   const int grpc_deadline = 10000);

  // Heartbeat request
  InfaasRequestStatus Heartbeat();

private:
  std::unique_ptr<Query::Stub> stub_;
};

}  // namespace internal
}  // namespace infaas

#endif  // #ifndef QUERY_CLIENT_H
