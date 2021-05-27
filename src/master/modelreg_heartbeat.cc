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

int main(int argc, char** argv) {
  infaaspublic::infaasmodelreg::ModelRegClient modelreg(grpc::CreateChannel(
      "localhost:50053", grpc::InsecureChannelCredentials()));
  infaaspublic::RequestReply reply = modelreg.Heartbeat();
  std::cout << "Model registration startup: ";
  if (reply.status() == infaaspublic::RequestReplyEnum::SUCCESS) {
    std::cout << "SUCCEEDED" << std::endl;
  } else {
    std::cout << "FAILED" << std::endl;
  }

  return 0;
}
