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

// This file contains utilities for autoscaler on the worker.
#ifndef AUTOSCALER_H
#define AUTOSCALER_H

#include <atomic>
#include <cstdint>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>

#include "metadata-store/redis_metadata.h"

namespace infaas {
namespace internal {
// We should have a sign for threads to stop. e.g., model_name =
// "infaas_stop", and count = "INT_MAX"
struct ScaleRequest {
  std::string model_name;
  int count;
  std::string down_var;
};

// None means do not autoscale; static means never scale down below the slack
// size; individual is scaling each model variant; and INFaaS is our per parent
// model algorithm.
enum AutoscalerType {
  AUTOSCALE_NONE = 0,
  AUTOSCALE_STATIC,
  AUTOSCALE_INDIVIDUAL,
  AUTOSCALE_INFAAS
};

// For auto scaling
class Autoscaler {
public:
  // Decide whether to scale up or down. The main entrance of autoscaler.
  static void AutoscalerArbiter(const std::string& worker_name,
                                const AutoscalerType& atype,
                                std::unique_ptr<RedisMetadata>& rmd);

  // Scale up/down on CPU/GPU. The daemon for actual execution.
  static void GpuAutoscalerDaemon(const std::string& worker_name,
                                  const AutoscalerType& atype,
                                  std::unique_ptr<RedisMetadata>& rmd,
                                  std::unique_ptr<Aws::S3::S3Client>& s3c);
  static void CpuAutoscalerDaemon(const std::string& worker_name,
                                  const AutoscalerType& atype,
                                  std::unique_ptr<RedisMetadata>& rmd,
                                  std::unique_ptr<Aws::S3::S3Client>& s3c);
  static void InfaAutoscalerDaemon(const std::string& worker_name,
                                   const AutoscalerType& atype,
                                   std::unique_ptr<RedisMetadata>& rmd,
                                   std::unique_ptr<Aws::S3::S3Client>& s3c);

  // If count > 0, load #count replicas, if count < 0, unload #count replicas.
  // Add one scaling request to the queue if the model_name has not existed.
  static int8_t setScaleRequestGpu(const std::string& model_name, int count,
                                   const std::string trt_down_var = "");
  static int8_t setScaleRequestCpu(const std::string& model_name, int count);
  static int8_t setScaleRequestInfa(const std::string& model_name, int count,
                                    const std::string infa_down_var = "");

  // Pop one scale request from the queue.
  // Return 0 means success, return -1 means the queue is empty or error
  // happened.
  static int8_t popScaleRequestGpu(ScaleRequest* reqs);
  static int8_t popScaleRequestCpu(ScaleRequest* reqs);
  static int8_t popScaleRequestInfa(ScaleRequest* reqs);
  static int getAvgBatch(const std::string& model_name);
  static void setAvgBatch(const std::string& model_name, int batch);

private:
  static std::mutex gpu_mutex_;
  static std::mutex cpu_mutex_;
  static std::mutex infa_mutex_;
  static std::deque<ScaleRequest> gpu_scale_reqs_;
  static std::deque<ScaleRequest> cpu_scale_reqs_;
  static std::deque<ScaleRequest> infa_scale_reqs_;
  // If true, the model is available to scale, otherwise, the model is blocked
  // and no more actions can be done.
  static std::map<std::string, std::atomic<bool>> model_available_;
  // Record how many backed up scale down requests for a model variant.
  static std::map<std::string, std::atomic<int>> model_num_scaledown_;
  static std::map<std::string, int> model_avg_batch_;
};

}  // namespace internal
}  // namespace infaas

#endif  // AUTOSCALER_H
