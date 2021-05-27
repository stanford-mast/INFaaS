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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <string>

#include "common_model_util.h"
#include "metadata-store/redis_metadata.h"
#include "trtis_request.h"

namespace trtis = nvidia::inferenceserver;
namespace trtisc = nvidia::inferenceserver::client;

static const std::string trtis_grpc_url = "localhost:8001";

// Timestamp to millisecond duration.
static double ts_to_ms(const struct timeval& start, const struct timeval& end) {
  return (end.tv_sec - start.tv_sec) * 1000.0 +
         (end.tv_usec - start.tv_usec) / 1000.0;
}

// Get timestamp in ms.
static double get_ts(const struct timeval& t) {
  return t.tv_sec * 1000.0 + t.tv_usec / 1000.0;
}

// Check trtserver status
static int8_t TrtserverReady() {
  std::unique_ptr<trtisc::ServerStatusContext> ctx;
  trtisc::Error err;
  err = trtisc::ServerStatusGrpcContext::Create(&ctx, trtis_grpc_url, false);
  if (err.IsOk()) {
    trtis::ServerStatus server_status;
    err = ctx->GetServerStatus(&server_status);
    if (err.IsOk()) {
      const auto& itr = server_status.ready_state();
      std::string status_str =
          trtis::ServerReadyState_descriptor()->FindValueByNumber(itr)->name();
      std::cout << "Current server status: " << status_str << std::endl;
      if (itr != trtis::SERVER_READY) {
        return 1;
      } else {
        return 0;
      }
    } else {
      std::cerr << "[TRTIS] Error: " << err.Message() << std::endl;
      return -1;
    }
  } else {
    std::cerr << "[TRTIS] Error: " << err.Message() << std::endl;
    return -1;
  }
  return 0;
}

// TODO: we use GPU memory usage as the GPU util, may need to change later.
void gpuMonitor(const std::string& worker_name,
                std::unique_ptr<RedisMetadata>& rmd) {
  struct timeval curr_time, prev_time;
  gettimeofday(&prev_time, NULL);
  int sleep_interval = 2500;  // Sleep 2.5 sec.
  while (true) {
    gettimeofday(&curr_time, NULL);
    double interval = ts_to_ms(prev_time, curr_time);
    // Skip the very first interval
    if (interval >= sleep_interval) {
      // Get GPU utilization
      uint64_t free_mem, total_mem;
      auto status = cudaMemGetInfo(&free_mem, &total_mem);
      if (status != cudaSuccess) {
        std::cerr << "Failed: cudaMemGetInfo, " << cudaGetErrorString(status)
                  << std::endl;
        continue;
      }
      double gpu_util = (double)(total_mem - free_mem) / (double)total_mem;
      gpu_util *= 100.0;  // percentage
      std::cout << std::fixed << get_ts(curr_time)
                << " [Interval = " << interval << " ] ";
      std::cout << "Worker: " << worker_name
                << " ; current GPU util: " << gpu_util << std::endl;
      int8_t rs = -1;
      // Only update to metadata if it's a reasonable point
      if (gpu_util > 0.0) {
        rs = rmd->update_gpu_util(worker_name, gpu_util);
        if (rs < 0) {
          std::cerr << "[gpuMonitor] Failed to update GPU util for worker: "
                    << worker_name << ". Status: " << int(rs) << std::endl;
        }
      }
      std::cout << "\n" << std::endl;
    }
    // Check trtserver health
    int8_t serverReady = TrtserverReady();
    std::cout << "[gpuMonitor] trtserver ready: " << serverReady << std::endl;
    if (serverReady < 0) {
      // Restart the trtserver
      std::string docker_cmd =
          "nvidia-docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit "
          "stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 "
          "-v/tmp/trtmodels:/models nvcr.io/nvidia/tensorrtserver:19.03-py3 "
          "trtserver --model-store=/models --exit-on-error=false "
          "--strict-model-config=false --tf-gpu-memory-fraction=0.5 "
          "--repository-poll-secs=5 > /opt/INFaaS/logs/worker/trtserver.log "
          "2>&1 &";
      int ready = system(docker_cmd.c_str());
      if (ready != 0) {
        std::cout << "Docker cmd " << docker_cmd << " failed." << std::endl;
      } else {
        std::cout << "Restarted trtserver!" << std::endl;
      }
    }
    prev_time = curr_time;
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
  }
  std::cout << "gpuMonitor shut down!" << std::endl;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: ./gpu_daemon <worker_name> <redis_ip> <redis_port>"
              << std::endl;
    exit(1);
  }
  std::string worker_name = argv[1];
  const struct Address redis_addr = {argv[2], argv[3]};
  if (RedisMetadata::is_empty_address(redis_addr)) {
    std::cerr << "Invalid redis server address: " << argv[2] << ":" << argv[3]
              << std::endl;
    exit(1);
  }
  std::cout << "Redis server address is: " << argv[2] << ":" << argv[3]
            << std::endl;

  std::unique_ptr<RedisMetadata> rmd =
      std::unique_ptr<RedisMetadata>(new RedisMetadata(redis_addr));

  gpuMonitor(worker_name, rmd);
  return 0;
}
