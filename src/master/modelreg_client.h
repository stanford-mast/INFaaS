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

#ifndef MODELREG_H
#define MODELREG_H

#include <string>

#include <grpcpp/grpcpp.h>
#include "modelreg.grpc.pb.h"

using grpc::Channel;

namespace infaaspublic {
namespace infaasmodelreg {

class ModelRegClient {
public:
  ModelRegClient(std::shared_ptr<Channel> channel);

  // RegisterModel request
  RequestReply RegisterModel(
      const std::string& submitter, const std::string& grandparent_model,
      const std::string& parent_model, const std::string& first_variant,
      const std::string& url, const std::string& dataset,
      const double& accuracy, const std::string& framework,
      const std::string& task, const std::string& config_name = "");

  // Heartbeat request
  RequestReply Heartbeat();

private:
  std::unique_ptr<ModelReg::Stub> stub_;
};

}  // namespace infaasmodelreg
}  // namespace infaaspublic

#endif
