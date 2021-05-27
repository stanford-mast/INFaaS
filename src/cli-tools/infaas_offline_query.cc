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

#include <unistd.h>
#include <iostream>
#include <string>

#include "master/modelreg_client.h"
#include "master/queryfe_client.h"

static const int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "Usage: infaas_offline_query <input_url> <output_url> "
                 "<model-architecture>";
    std::cout << std::endl;
    std::cout << "Input_url/output_url: AWS bucket (do not include s3://)"
              << std::endl;
    return 1;
  }

  std::string input_url = std::string(argv[1]);
  std::string output_url = std::string(argv[2]);
  std::string model_architecture = std::string(argv[3]);

  // Submit the offline job
  grpc::ChannelArguments arguments;
  arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
  arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);
  infaaspublic::infaasqueryfe::QueryFEClient queryfe(grpc::CreateCustomChannel(
      "localhost:50052", grpc::InsecureChannelCredentials(), arguments));

  infaaspublic::RequestReply query_reply =
      queryfe.QueryOffline(input_url, model_architecture, "", output_url, 0.0);

  std::cout << "Offline query job submission ";
  if (query_reply.status() == infaaspublic::RequestReplyEnum::SUCCESS) {
    std::cout << "SUCCEEDED" << std::endl;
    std::cout << "You can now monitor " << output_url << " for job progress"
              << std::endl;
  } else {
    std::cout << "FAILED: " << query_reply.msg() << std::endl;
  }

  return 0;
}
