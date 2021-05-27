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

#include "redis_metadata.h"

int main(int argc, char** argv) {
  // We don't need to check for a valid address since this has already been
  // checked by the startup script that calls this program
  if (argc < 8) {
    std::cerr << "Usage: ./redis_startup_helper <redis-ip> <redis-port> ";
    std::cerr << "<worker-name> <worker-ip> <worker-port> <worker-type> ";
    std::cerr << "<is-slack> [<instid>]" << std::endl;
    std::cerr << "worker-type => 0: CPU only, 1: Inferentia, 2: GPU";
    std::cerr << std::endl;
    std::cerr << "is-slack: 1 if is slack GPU" << std::endl;
    std::cerr << "instid is only required if running on AWS" << std::endl;
    return 1;
  }

  std::string redis_ip = argv[1];
  std::string redis_port = argv[2];
  std::string worker_name = argv[3];
  std::string worker_ip = argv[4];
  std::string worker_port = argv[5];
  int worker_type = std::stoi(argv[6]);
  int is_slack = std::stoi(argv[7]);

  RedisMetadata rmd({redis_ip, redis_port});

  const struct Address worker_addr = {worker_ip, worker_port};

  if (rmd.add_executor_addr(worker_name, worker_addr)) {
    std::cerr << "Failed to add worker in startup helper!" << std::endl;
    return 1;
  }

  // Set to CPU only or Inferentia if applicable
  if (worker_type == 0) {
    if (rmd.set_exec_onlycpu(worker_name)) {
      std::cerr << "Failed to set worker to CPU only!" << std::endl;
      return 1;
    }
  } else if (worker_type == 1) {
    if (rmd.set_exec_inferentia(worker_name)) {
      std::cerr << "Failed to set worker to support Inferentia!" << std::endl;
      return 1;
    }
  }

  // Set to is-slack only if applicable
  if (is_slack) {
    if (rmd.set_exec_slack(worker_name)) {
      std::cerr << "Failed to set worker as slack!" << std::endl;
      return 1;
    }
  }

  // Add instance-id as well
  if (argc == 9) {
    std::string instid = argv[8];
    if (rmd.add_executor_instid(worker_name, instid)) {
      std::cerr << "Failed to add worker's instance-id in startup helper!"
                << std::endl;
      return 1;
    }
  }

  return 0;
}
