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

// This file contains some common utility functions to add/remove a model
// instance on the worker node. It also has functions to query the model.
#ifndef COMMON_MODEL_UTIL_H
#define COMMON_MODEL_UTIL_H

#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <google/protobuf/repeated_field.h>

#include "internal/query.grpc.pb.h"
#include "metadata-store/redis_metadata.h"

namespace infaas {
namespace internal {

class CommonModelUtil {
public:
  static bool HasBlacklisted();
  static void SetBlacklisted(bool flag);

private:
  // This is used to decide whether we need to stop all offline processing.
  static bool has_blacklisted_;
  static std::mutex mutex_;  // Protect the update of map.
};

// Convert timespec to uint64_t timestamp in usec.
inline uint64_t timespec_to_us(const struct timespec& a) {
  return (uint64_t)a.tv_sec * 1000000 + (uint64_t)a.tv_nsec / 1000;
}

// Get timestamp in usec.
inline uint64_t get_curr_timestamp() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return timespec_to_us(ts);
}

// Compute the duration between two timestamps (in usec), in milliseconds.
inline double get_duration_ms(const uint64_t& a, const uint64_t& b) {
  return (double)std::abs(b - a) / 1000.0;
}

int8_t list_s3_path(const std::string& bucket_name, const std::string& obj_name,
                    std::unique_ptr<Aws::S3::S3Client>& s3c,
                    std::vector<std::string>* key_names);

int8_t download_s3_local(const std::string& src_bucket,
                         const std::string& prefix,
                         const std::vector<std::string>& key_names,
                         const std::string& local_base_dir,
                         std::unique_ptr<Aws::S3::S3Client>& s3c);

int8_t upload_local_s3(const std::string& local_base_dir,
                       const std::vector<std::string>& file_names,
                       const std::string& dst_bucket, const std::string& prefix,
                       std::unique_ptr<Aws::S3::S3Client>& s3c);

// Parse and the "count: x" from the pbtxt file.
size_t get_count_from_pbtxt(const std::string& file_path);

// Parse and set the "count: x" to the pbtxt file.
// Return 0 means success, -1 means error happened.
int8_t set_count_from_pbtxt(const std::string& file_path, size_t count);

// Choose hardware based on the model framework.
std::string ChooseHardware(const std::string& model_name,
                           const std::unique_ptr<RedisMetadata>& rmd,
                           const std::string framework = "");

// For GPU executor
class GpuModelManager {
public:
  GpuModelManager(std::string worker_name) : worker_name_(worker_name) {}

  // Load a model to GPU
  // src_url: model source URL, the format should align with TRTIS model
  // repository.
  // model_name: model name, it might different from src_url object name
  //
  // Return 0 means success, return -1 means error.
  int8_t LoadModel(const std::string src_url, const std::string model_name,
                   std::unique_ptr<RedisMetadata>& rmd,
                   std::unique_ptr<Aws::S3::S3Client>& s3c);

  // Unload a model on GPU
  // model_name: model name to unload
  //
  // Return 0 means success, return -1 means error.
  int8_t UnloadModel(const std::string& model_namei,
                     std::unique_ptr<RedisMetadata>& rmd);

  // Query a model for online request.
  // This is the entrance of all GPU model query.
  int8_t QueryModelOnline(const std::string& model_name,
                          const QueryOnlineRequest* request,
                          QueryOnlineResponse* reply,
                          std::unique_ptr<RedisMetadata>& rmd,
                          std::unique_ptr<Aws::S3::S3Client>& s3c);

  // Query an Image Classification Model
  // model_name: the name of the model
  // topk: the number of classes returned. (Default: 1)
  //
  // Return 0 means success, return -1 means error.
  int8_t QueryImageClassModel(
      const std::string& model_name,
      const google::protobuf::RepeatedPtrField<std::string>& raw_input,
      google::protobuf::RepeatedPtrField<std::string>* raw_output,
      size_t topk = 1);

  // Query General models
  int8_t QueryGeneralModel(
      const std::string& model_name,
      const google::protobuf::RepeatedPtrField<std::string>& raw_input,
      google::protobuf::RepeatedPtrField<std::string>* raw_output);

  // For autoscaling
  static int numReplicas(const std::string& model_name);
  static int8_t changeNumReplicas(const std::string& model_name,
                                  const size_t& count);

private:
  std::string worker_name_;
  static std::map<std::string, size_t> model_to_num_reps_;  // model name to
                                                            // number of
                                                            // replicas. -1
                                                            // means the model
                                                            // is being loaded.
  static std::mutex update_mutex_;  // Protect the update of map.
};

// For CPU executor
class CpuModelManager {
public:
  CpuModelManager(std::string worker_name) : worker_name_(worker_name) {}

  // Load a model to CPU
  // src_url: model source URL.
  // model_name: model name, it might different from src_url object name
  // container_name: the name for that container, by default the same as
  // model_name.
  // for_online: true means container is for Online query, otherwise, for
  // Offline.
  //
  // Return 0 means success, return -1 means error.
  int8_t LoadModel(const std::string src_url, const std::string model_name,
                   std::unique_ptr<RedisMetadata>& rmd,
                   std::unique_ptr<Aws::S3::S3Client>& s3c,
                   const std::string container_name = "",
                   bool for_online = true);

  // Unload a model instance on CPU, pick the instance from the back of the
  // queue. model_name: model name to unload container_name: specify a container
  // name to unload. for_online: true means container is for Online query,
  // otherwise, for Offline.
  //
  // Return 0 means success, return -1 means error.
  int8_t UnloadModel(const std::string& model_name,
                     std::unique_ptr<RedisMetadata>& rmd,
                     const std::string container_name = "",
                     bool for_online = true);

  // Query a model for online request.
  // This is the entrance of all CPU model query.
  int8_t QueryModelOnline(const std::string& model_name,
                          const QueryOnlineRequest* request,
                          QueryOnlineResponse* reply,
                          std::unique_ptr<RedisMetadata>& rmd,
                          std::unique_ptr<Aws::S3::S3Client>& s3c);

  // Process an offline request.
  // This is the entrance of all CPU model query.
  int8_t QueryModelOffline(const std::string& model_name,
                           const QueryOfflineRequest& request,
                           std::unique_ptr<RedisMetadata>& rmd,
                           std::unique_ptr<Aws::S3::S3Client>& s3c);

  // Query an Image Classification Model
  // model_name: the name of the model
  // topk: the number of classes returned. (Default: 1)
  //
  // Return 0 means success, return -1 means error.
  int8_t QueryImageClassModel(
      const std::string& model_name,
      const google::protobuf::RepeatedPtrField<std::string>& raw_input,
      google::protobuf::RepeatedPtrField<std::string>* raw_output,
      size_t topk = 1);

  static size_t numReplicas(const std::string& model_name);

private:
  std::string worker_name_;
  // Map model to container names.
  // The queue is used to do round-robin scheduling among replicas, where it
  // stores the names of containers for that model.
  static std::map<std::string, std::deque<std::string>> model_to_names_online_;
  static std::map<std::string, std::deque<std::string>> model_to_names_offline_;
  static std::map<std::string, int>
      name_to_port_;  // Map container name to port.
  static std::set<int, std::greater<int>>
      used_ports_;                  // Record the unavailable port numbers.
  static std::mutex update_mutex_;  // TODO: Only 1 model can be updated to
                                    // above vectors at the same time.
};

}  // namespace internal
}  // namespace infaas

#endif  // #ifndef COMMON_MODEL_UTIL_H
