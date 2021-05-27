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

#ifndef QUERYFE_H
#define QUERYFE_H

#include <cstdint>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "queryfe.grpc.pb.h"

using grpc::Channel;

namespace infaaspublic {
namespace infaasqueryfe {
class QueryFEClient {
public:
  QueryFEClient(std::shared_ptr<Channel> channel);

  // QueryOnline request
  std::vector<std::string> QueryOnline(
      const std::vector<std::string>& input,
      const std::string& grandparent_model = "",
      const std::string& parent_model = "",
      const std::string& model_variant = "", const std::string& submitter = "",
      const int64_t& latency = 0, const double& minacc = 0,
      const double& maxcost = 0);

  // QueryOffline request
  RequestReply QueryOffline(const std::string& input_url,
                            const std::string& model,
                            const std::string& submitter,
                            const std::string& output_url,
                            const double& maxcost = 0);

  // AllParentInfo request
  AllParReply AllParentInfo(const std::string& task,
                            const std::string& dataset);

  // QueryModelInfo request
  QueryModelReply QueryModelInfo(const std::string& model);

  // Heartbeat request
  RequestReply Heartbeat();

private:
  std::unique_ptr<Query::Stub> stub_;
};

}  // namespace infaasqueryfe
}  // namespace infaaspublic

#endif
