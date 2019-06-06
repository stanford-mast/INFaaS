/*
 * Copyright 2018-2019 Board of Trustees of Stanford University
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
#include <string>

#include "include/constants.h"
#include "master/modelreg_client.h"
#include "master/queryfe_client.h"

static const int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: infaas_modinfo <model-architecture>" << std::endl;
    std::cout << "model-architecture: registered model architecture ";
    std::cout << "(options can be found using infaas_modarch)" << std::endl;
    return 1;
  }

  std::string parent_model(argv[1]);

  // Ask for model info
  grpc::ChannelArguments arguments;
  arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
  arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);
  infaaspublic::infaasqueryfe::QueryFEClient queryfe(grpc::CreateCustomChannel(
      "localhost:50052", grpc::InsecureChannelCredentials(), arguments));

  infaaspublic::QueryModelReply query_reply =
      queryfe.QueryModelInfo(parent_model);
  std::cout << "Info query for model architecture: " << parent_model << " ";
  if (query_reply.status().status() ==
      infaaspublic::RequestReplyEnum::SUCCESS) {
    std::cout << "SUCCEEDED" << std::endl;
    std::cout << "Input dimensions (if classification): "
              << query_reply.img_dim() << std::endl;
    std::cout << "Accuracy (% if classification, BLEU if translation): ";
    std::cout << query_reply.accuracy() << std::endl;
    std::cout << "Variants: " << std::endl;
    for (auto av : query_reply.all_models()) {
      std::cout << "\t" << av << std::endl;
    }
  } else {
    std::cout << "FAILED: " << query_reply.status().msg() << std::endl;
  }

  return 0;
}
