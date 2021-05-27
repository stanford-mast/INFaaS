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
#include <string>

#include "include/constants.h"
#include "master/modelreg_client.h"

static const int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Usage: ./infaas_modelregistration <config-name> "
                 "<model_var_dir_path>";
    std::cout << std::endl;
    // std::cout << "Submitter is assumed to be tester" << std::endl;
    std::cout
        << "model_var_dir_path: S3 path to your frozen model's root directory";
    std::cout << std::endl;
    std::cout << "Do not include 's3://' and ensure model_var_dir_path ends "
                 "with a '/'";
    std::cout << std::endl;
    return 1;
  }
  std::string submitter("tester");
  std::string config_name(argv[1]);
  std::string url(argv[2]);  // model_var_dir_path; Ending "/" is important!
  std::string grandparent("");
  std::string parent("");
  std::string dataset("");
  std::string framework("");
  std::string task("");
  double accuracy = 0.0;

  size_t ind = config_name.find(".");
  std::string first_variant = config_name.substr(0, ind);

  // First register the model
  infaaspublic::infaasmodelreg::ModelRegClient modelreg(grpc::CreateChannel(
      "localhost:50053", grpc::InsecureChannelCredentials()));

  infaaspublic::RequestReply reply =
      modelreg.RegisterModel(submitter, grandparent, parent, first_variant, url,
                             dataset, accuracy, framework, task, config_name);
  std::cout << "Model registration for " << first_variant << "... ";
  if (reply.status() == infaaspublic::RequestReplyEnum::SUCCESS) {
    std::cout << "SUCCEEDED" << std::endl;
  } else {
    std::cout << "FAILED: " << reply.msg() << std::endl;
  }

  return 0;
}
