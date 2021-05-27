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

// This file is used to send simple queries to Inferentia docker containers.
// Only used for debugging and testing.
#include <assert.h>
#include <unistd.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "include/constants.h"
#include "query_client.h"

#define FAIL(x) printf("[FAIL]: %s\n", x)
#define PASS(x) printf("[PASS]: %s\n", x)

namespace infaas {
namespace internal {
namespace {

bool test_heartbeat(QueryClient& query_client) {
  auto reply = query_client.Heartbeat();
  if (reply.status() == infaas::internal::InfaasRequestStatusEnum::SUCCESS) {
    PASS("Heartbeat");
  } else {
    std::string errmsg = "Heartbeat. Error msg: " + reply.msg();
    FAIL(errmsg.c_str());
    return false;
  }
  return true;
}

bool test_online_simple(QueryClient& query_client,
  const std::vector<std::string>& raw_text) {
  // Prepare input
  std::string inp_str = "The quick brown fox jumps over the lazy dog";
  QueryOnlineRequest request;
  request.add_raw_input(inp_str);
  if (!raw_text.empty()) {
    for (auto& raw : raw_text) {
      request.add_raw_input(raw);
    }
  }
  int batch_size = request.raw_input().size();
  std::cout << "input batch size" << batch_size << std::endl;

  QueryOnlineResponse resp;
  auto output = resp.mutable_raw_output();
  auto reply = query_client.QueryOnline(request.raw_input(), {"testmodel"},
    "Qian", output, 1000, 71, 200);
  if (reply.status() != InfaasRequestStatusEnum::SUCCESS) {
    std::string msg = "Query Online-testmodel, error msg: " + reply.msg();
    FAIL(msg.c_str());
    return false;
  }

  for (int i = 0; i < batch_size; ++i) {
    std::string out_str = output->Get(i);
    std::cout << "INFER OUTPUT: " << out_str << std::endl;
  }
  return true;
}

}  // namespace
}  // namespace internal
}  // namespace infaas

int main(int argc, char** argv) {
  // NOTE: need to launch Inferentia container/script before running this script.
  // The Inferentia container/script by default should run at port 9001
  std::string inf_addr = "localhost:9001";
  std::vector<std::string> raw_text;
  int add_inputs = 0;
  if (argc > 1) {
    inf_addr = "localhost:" + std::string(argv[1]);
    if (argc > 2) {
      add_inputs = atoi(argv[2]);
    }
  } else {
    std::cerr << "Usage: ./inferentia_query_test <port (default 9001)> <additional_input_sentences (default 0)>"
              << std::endl;
  }

  for (int i = 0; i < add_inputs; ++i) {
    std::cout << "Enter a sentence: ";
    std::string user_inp;
    std::getline(std::cin, user_inp);
    raw_text.push_back(user_inp);
  }

  // Connect to the container.
  infaas::internal::QueryClient query_client(
    grpc::CreateChannel(inf_addr, grpc::InsecureChannelCredentials()));

  // Test heatbeat.
  std::cout << "=== HEARTBEAT TESTS ===" << std::endl;
  assert(infaas::internal::test_heartbeat(query_client));

  // Test online query.
  std::cout << "=== QUERY ONLINE TESTS ===" << std::endl;
  assert(infaas::internal::test_online_simple(query_client, raw_text));

  return 0;
}
