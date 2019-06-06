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

#include "redis_metadata.h"

int main(int argc, char** argv) {
  // We don't need to check for a valid address since this has already been
  // checked by the startup script that calls this program
  if (argc < 7) {
    std::cerr << "Usage: ./redis_startup_helper <redis-ip> <redis-port> ";
    std::cerr << "<worker-name> <worker-ip> <worker-port> <is-cpu> [<instid>]"
              << std::endl;
    std::cerr << "is-cpu: 1 if is CPU only" << std::endl;
    std::cerr << "instid is only required if running on AWS" << std::endl;
    return 1;
  }

  std::string redis_ip = argv[1];
  std::string redis_port = argv[2];
  std::string worker_name = argv[3];
  std::string worker_ip = argv[4];
  std::string worker_port = argv[5];
  int is_cpu = std::stoi(argv[6]);

  RedisMetadata rmd({redis_ip, redis_port});

  const struct Address worker_addr = {worker_ip, worker_port};

  if (rmd.add_executor_addr(worker_name, worker_addr)) {
    std::cerr << "Failed to add worker in startup helper!" << std::endl;
    return 1;
  }

  // Set to CPU only if applicable
  if (is_cpu) {
    if (rmd.set_exec_onlycpu(worker_name)) {
      std::cerr << "Failed to set worker to CPU only!" << std::endl;
      return 1;
    }
  }

  // Add instance-id as well
  if (argc == 8) {
    std::string instid = argv[7];
    if (rmd.add_executor_instid(worker_name, instid)) {
      std::cerr << "Failed to add worker's instance-id in startup helper!"
                << std::endl;
      return 1;
    }
  }

  return 0;
}
