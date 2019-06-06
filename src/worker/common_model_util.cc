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

#include <curl/curl.h>
#include <dirent.h>
#include <grpcpp/grpcpp.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/DeleteObjectsRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/PutObjectRequest.h>

#include "common_model_util.h"
#include "include/base64.h"
#include "include/constants.h"
#include "include/json.hpp"
#include "internal/query.grpc.pb.h"
#include "query_client.h"
#include "trtis_request.h"

using Aws::S3::S3Client;
using grpc::ClientContext;
using grpc::Status;
using json = nlohmann::json;

// Constants and global variables
static const std::string trtis_grpc_url = "localhost:8001";
static const std::string local_model_dir = "/tmp/models/";
static const std::string local_trt_model_dir = "/tmp/trtmodels/";
static const std::string local_input_dir = "/tmp/infaas_input/";
static const std::string local_output_dir = "/tmp/infaas_output/";
static const std::string local_input_leaf_dir = "infer/";
static const std::string trt_pbtxt_file = "config.pbtxt";
static const std::string batching_parameters_file =
    infaas_path + "/src/worker/batching_parameters.txt";
static const int first_cpu_port = 9001;  // The first port used by CPU container
static const int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;
static const int offline_batch = 4;  // This might change.
static const int offline_cpu_thresh =
    40;  // Offline can run iff CPU util <= 40%, might change.
static const int sleep_interval = 1000;  // Offline poll per 1 sec.
static const int DOCKER_STOP_TIME =
    1;  // Wait 1 sec before killing a container, to allow it finish remaining
        // requests. TODO: need to stop a container gracefully.

static const bool CPU_ADAPTIVE_BATCHING = false;
static const bool OFFLINE_CONTROL =
    true;  // if true, run to avoid interference. Otherwise, run offline anyway.

namespace trtis = nvidia::inferenceserver;
namespace trtisc = nvidia::inferenceserver::client;

namespace infaas {
namespace internal {
namespace {

inline bool file_exist(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

// Create a directory recursively
inline int create_dir(const std::string& inp_path) {
  std::cout << "creating " << inp_path << std::endl;
  std::string path = inp_path;
  if (path.back() == '/') { path.pop_back(); }
  std::string parent_dir = path.substr(0, path.rfind("/"));
  // If parent directory doesn't exist, create one.
  int res = 0;
  if (!file_exist(parent_dir)) { res = create_dir(parent_dir); }
  if (res) {
    std::cerr << "failed to create parent_dir " << parent_dir << std::endl;
    return res;
  }
  return mkdir(path.c_str(), S_IRWXU | S_IRWXG);
}

// List all files inside a directory (not recursively!)
// Borrow the idea from: https://stackoverflow.com/a/612176
int8_t list_local_path(const std::string& local_path,
                       std::vector<std::string>* file_names) {
  DIR* dir;
  struct dirent* ent;
  if ((dir = opendir(local_path.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      std::string fname(ent->d_name);
      // Filter out the current/parent directory "." and "..".
      if (fname.back() != '.') { file_names->push_back(fname); }
    }
    closedir(dir);
  } else {
    // Could not open the directory.
    std::cerr << "Failed to open directory: " << local_path << std::endl;
    return -1;
  }
  return 0;
}

// Execute a command and get stdout string.
// Modified from: https://stackoverflow.com/a/478960
std::string execcmd(const char* cmd) {
  std::cout << "Execcmd: " << cmd << std::endl;
  char buffer[200];
  std::string result = "";
  FILE* pipe = popen(cmd, "r");
  if (!pipe) throw std::runtime_error("popen() failed!");
  try {
    while (fgets(buffer, sizeof buffer, pipe) != NULL) { result += buffer; }
  } catch (...) {
    pclose(pipe);
    throw;
  }
  pclose(pipe);
  return result;
}

void parse_s3_url(const std::string& src_url, std::string* src_bucket,
                  std::string* obj_name) {
  *obj_name = src_url.substr(bucket_prefix.size());
  auto pre_ind = obj_name->find("/");  // exclude bucket name
  *src_bucket = obj_name->substr(0, pre_ind);
  *obj_name = obj_name->substr(pre_ind + 1);
}

trtis::ModelReadyState GpuModelState(const std::string& model_name) {
  uint64_t time1, time2;
  time1 = get_curr_timestamp();

  trtis::ModelReadyState state = trtis::MODEL_UNAVAILABLE;
  std::unique_ptr<trtisc::ServerStatusContext> ctx;
  trtisc::Error err;
  err = trtisc::ServerStatusGrpcContext::Create(&ctx, trtis_grpc_url,
                                                model_name, false);
  if (err.IsOk()) {
    trtis::ServerStatus server_status;
    err = ctx->GetServerStatus(&server_status);
    if (err.IsOk()) {
      const auto& itr = server_status.model_status().find(model_name);
      if (itr == server_status.model_status().end()) {
        std::cerr << "[TRTIS] Couldn't find the model " << model_name
                  << std::endl;
        return state;
      } else {
        if (!itr->second.version_status().size()) {
          std::cerr << "[TRTIS] Couldn't find version status for model "
                    << model_name << std::endl;
          return state;
        }
        auto curr_state =
            itr->second.version_status().begin()->second.ready_state();
        std::string curr_state_str = trtis::ModelReadyState_descriptor()
                                         ->FindValueByNumber(curr_state)
                                         ->name();
        std::cerr << "[TRTIS] Current model state: " << curr_state_str
                  << std::endl;
        state = curr_state;
      }
    } else {
      std::cerr << "[TRTIS] Error: " << err.Message() << std::endl;
      return state;
    }
  } else {
    std::cerr << "[TRTIS] Error: " << err.Message() << std::endl;
    return state;
  }
  time2 = get_curr_timestamp();
  printf("[common_model_util.cc] GpuModelState  %.4lf ms.\n",
         get_duration_ms(time1, time2));
  return state;
}

// Wait until a GPU model reaches the desired state.
// Return 0 means success. -1 means error.
int8_t WaitGpuModelState(const std::string& model_name,
                         trtis::ModelReadyState model_state, unsigned interval,
                         int maxiter = 60) {
#ifdef INFAAS_GPU_WORKER
  uint64_t time1, time2;
  time1 = get_curr_timestamp();

  std::unique_ptr<trtisc::ServerStatusContext> ctx;
  trtisc::Error err;
  err = trtisc::ServerStatusGrpcContext::Create(&ctx, trtis_grpc_url,
                                                model_name, false);
  if (err.IsOk()) {
    int cnt = 0, max_try = 60;
    trtis::ServerStatus server_status;
    while (cnt < max_try) {
      cnt++;
      std::cout << "Try: # " << cnt << std::endl;
      err = ctx->GetServerStatus(&server_status);
      if (err.IsOk()) {
        const auto& itr = server_status.model_status().find(model_name);
        if (itr == server_status.model_status().end()) {
          std::cerr << "[TRTIS] Couldn't find the model " << model_name
                    << std::endl;
          continue;
        } else {
          if (!itr->second.version_status().size()) {
            std::cerr << "[TRTIS] Couldn't find version status for model "
                      << model_name << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(interval));
            continue;  // No available status.
          }
          auto curr_state =
              itr->second.version_status().begin()->second.ready_state();
          std::string curr_state_str = trtis::ModelReadyState_descriptor()
                                           ->FindValueByNumber(curr_state)
                                           ->name();
          std::cerr << "[TRTIS] Current model state: " << curr_state_str
                    << std::endl;

          if (curr_state == model_state) {
            time2 = get_curr_timestamp();
            printf("[common_model_util.cc] WaitGpuModelState  %.4lf ms.\n",
                   get_duration_ms(time1, time2));
            std::cout << "Final model state: " << curr_state_str << std::endl;
            return 0;
          }
        }
      } else {
        std::cerr << "[TRTIS] Error: " << err.Message() << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    }

    if (cnt == max_try) {
      std::cerr << "[TRTIS] WaitGpuModelState TIMED OUT!" << std::endl;
      return -1;
    }

  } else {
    std::cerr << "[TRTIS] Failed to check TRTIS server status" << std::endl;
    return -1;
  }
  time2 = get_curr_timestamp();
  printf("[common_model_util.cc] WaitGpuModelState  %.4lf ms.\n",
         get_duration_ms(time1, time2));
#endif  // #ifdef INFAAS_GPU_WORKER
  return 0;
}

// Borrowed the idea from https://gist.github.com/alghanmi/c5d7b761b2c9ab199157
static size_t CurlWriteCallBack(void* contents, size_t sz, size_t nmemb,
                                void* buff) {
  ((std::string*)buff)->append((char*)contents, sz * nmemb);
  return sz * nmemb;
}
// Wait until CPU container is ready to serve
bool WaitCpuModelReady(const std::string& model_name,
                       const std::string& framework, int portnum,
                       unsigned interval) {
  int cnt = 0, max_try = 50;  // Wait 50 times
  if (framework == "pytorch") {
    struct Address dest_addr = {"localhost", std::to_string(portnum)};
    QueryClient query_client(
        grpc::CreateChannel(RedisMetadata::Address_to_str(dest_addr),
                            grpc::InsecureChannelCredentials()));
    while (cnt < max_try) {
      auto reply = query_client.Heartbeat();
      if (reply.status() == InfaasRequestStatusEnum::SUCCESS) { break; }
      cnt++;
      std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    }
  } else if (framework == "tensorflow-cpu") {
    std::string tf_url = "http://localhost:" + std::to_string(portnum) +
                         "/v1/models/" + model_name;
    std::cout << "TF url: " << tf_url << std::endl;
    CURL* curl;
    CURLcode res;
    curl = curl_easy_init();
    if (!curl) { cnt = max_try; }
    while (cnt < max_try) {
      std::string readbuff;
      curl_easy_setopt(curl, CURLOPT_URL, tf_url.c_str());
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallBack);
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readbuff);
      res = curl_easy_perform(curl);
      if (readbuff.find("\"state\": \"AVAILABLE\"") != std::string::npos) {
        break;
      }
      cnt++;
      std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    }
    curl_easy_cleanup(curl);
  } else {
    // Unsupported framework.
    cnt = max_try;
  }
  if (cnt == max_try) { return false; }
  return true;
}

// Wait until the instance is not available.
bool WaitCpuInstanceStop(const std::string& instance_name) {
  char docker_cmd[512];
  sprintf(docker_cmd, "docker ps -aqf \"name=%s\"", instance_name.c_str());
  int cnt = 0, max_try = 10;  // wait 10sec
  int interval = 1000;        // 1 sec
  while (cnt < max_try) {
    std::string retstr = execcmd(docker_cmd);
    if (!retstr.empty()) {
      std::cout << "Instance " << instance_name << " still exists!"
                << std::endl;
    } else {
      break;
    }
    cnt++;
    std::this_thread::sleep_for(std::chrono::milliseconds(interval));
  }
  if (cnt == max_try) { return false; }
  return true;
}

}  // namespace

////////////////////////////// For Common Model Util /////////////////////////
bool CommonModelUtil::has_blacklisted_ = false;
std::mutex CommonModelUtil::mutex_;  // Protect the update of map.

bool CommonModelUtil::HasBlacklisted() {
  std::lock_guard<std::mutex> lock(CommonModelUtil::mutex_);
  return has_blacklisted_;
}

void CommonModelUtil::SetBlacklisted(bool flag) {
  std::lock_guard<std::mutex> lock(CommonModelUtil::mutex_);
  has_blacklisted_ = flag;
}

std::string ChooseHardware(const std::string& model_name,
                           const std::unique_ptr<RedisMetadata>& rmd,
                           const std::string givenFramework) {
  // We can select hardware type based on the framework.
  std::string framework;
  if (!givenFramework.empty()) {
    framework = givenFramework;
  } else {
    framework = rmd->get_model_info(model_name, "framework");
  }
  // std::cout << "ChooseHardware: framework: " << framework << std::endl;
  if (framework == "pytorch") {
    return "CPU";
  } else if (framework == "tensorflow-cpu") {
    return "CPU";
  }
#ifdef INFAAS_GPU_WORKER
  if (framework == "tensorrt") {
    return "GPU";
  } else if (framework == "tensorflow-gpu") {
    return "GPU";
  }
#endif  // #ifdef INFAAS_GPU_WORKER
  std::cout << "Framework unsupported: " << framework << std::endl;
  return "UNSUPPORTED";
}

// Borrow the idea from
// https://github.com/scanner-research/storehouse/s3/s3_storage.cpp#L285-L318
int8_t list_s3_path(const std::string& bucket_name, const std::string& obj_name,
                    std::unique_ptr<S3Client>& s3c,
                    std::vector<std::string>* key_names) {
  std::string cont_token("");
  std::string dir_name = obj_name;
  // if (dir_name.back() != '/') { dir_name += "/"; }
  while (1) {
    Aws::S3::Model::ListObjectsV2Request list_request;
    list_request.SetBucket(Aws::String(bucket_name.c_str()));
    list_request.SetPrefix(Aws::String(dir_name.c_str()));
    if (cont_token != "") {
      list_request.SetContinuationToken(Aws::String(cont_token.c_str()));
    }
    auto list_objects_outcome = s3c->ListObjectsV2(list_request);
    if (list_objects_outcome.IsSuccess()) {
      auto res = list_objects_outcome.GetResult();
      for (const auto& obj : res.GetContents()) {
        auto objkey = obj.GetKey();
        std::string keyname = std::string(objkey.c_str(), objkey.size());
        if (keyname.back() != '/') {
          // Skip the folder name that ends with "/".
          key_names->push_back(keyname);
        }
      }
      // Are there more objects to fetch?
      if (!res.GetIsTruncated()) {
        // No, so finish the loop
        break;
      } else {
        auto awscont = res.GetContinuationToken();
        cont_token = std::string(awscont.c_str(), awscont.size());
      }
    } else {
      if (list_objects_outcome.GetError().ShouldRetry()) {
        return 1;  // TODO: need retry logic.
      } else {
        auto error = list_objects_outcome.GetError();
        std::cerr << "list_s3_path got error: " << error.GetMessage();
        return -1;
      }
    }
  }
  return 0;
}

int8_t download_s3_local(const std::string& src_bucket,
                         const std::string& prefix,
                         const std::vector<std::string>& key_names,
                         const std::string& local_base_dir,
                         std::unique_ptr<S3Client>& s3c) {
  // Download each individual file.
  for (auto& key_name : key_names) {
    Aws::S3::Model::GetObjectRequest object_request;
    object_request.WithBucket(Aws::String(src_bucket.c_str()))
        .WithKey(Aws::String(key_name.c_str()));
    auto get_object_outcome = s3c->GetObject(object_request);
    if (get_object_outcome.IsSuccess()) {
      std::string local_file_path;
      Aws::OFStream local_file;
      // double check this is indeed the prefix.
      if (!prefix.empty() && (key_name.find(prefix) == 0)) {
        // Skip the prefix.
        local_file_path = local_base_dir + key_name.substr(prefix.size());
      } else {
        local_file_path = local_base_dir + key_name;
      }
      std::string parent_dir =
          local_file_path.substr(0, local_file_path.rfind("/"));

      std::cout << "Downloading key_name " << key_name << " to local path "
                << local_file_path << std::endl;
      // If parent directory doesn't exist, create one.
      if (!file_exist(parent_dir)) {
        if (create_dir(parent_dir)) {
          std::cerr << "Failed to create dir " << parent_dir << std::endl;
          return -1;
        }
      }
      local_file.open(local_file_path.c_str(),
                      std::ios::out | std::ios::binary);
      local_file << get_object_outcome.GetResult().GetBody().rdbuf();
      local_file.close();
    } else {
      std::cerr << "GetObject error: "
                << get_object_outcome.GetError().GetExceptionName() << " "
                << get_object_outcome.GetError().GetMessage() << std::endl;
      return -1;
    }
  }
  return 0;
}

int8_t upload_local_s3(const std::string& local_base_dir,
                       const std::vector<std::string>& file_names,
                       const std::string& dst_bucket, const std::string& prefix,
                       std::unique_ptr<Aws::S3::S3Client>& s3c) {
  // Local each individual file.
  for (auto& file_name : file_names) {
    std::string key_name = prefix + file_name;
    std::string local_path = local_base_dir + file_name;
    Aws::S3::Model::PutObjectRequest object_request;
    object_request.WithBucket(Aws::String(dst_bucket.c_str()))
        .WithKey(Aws::String(key_name.c_str()));
    auto input_data = Aws::MakeShared<Aws::FStream>(
        "PutObjectInputStream", local_path.c_str(),
        std::ios_base::in | std::ios_base::binary);
    object_request.SetBody(input_data);
    std::cout << "Uploading " << local_path << " to " << dst_bucket + key_name
              << std::endl;
    auto put_object_outcome = s3c->PutObject(object_request);
    if (!put_object_outcome.IsSuccess()) {
      std::cerr << "PutObject error: "
                << put_object_outcome.GetError().GetExceptionName() << " "
                << put_object_outcome.GetError().GetMessage() << std::endl;
      return -1;
    }
  }
  return 0;
}

size_t get_count_from_pbtxt(const std::string& file_path) {
  std::ifstream pbtxtfile(file_path);
  std::string line;
  size_t count = 0;
  const std::string prefix = "count: ";
  while (std::getline(pbtxtfile, line)) {
    // Read the number of replicas from file.
    auto ind = line.find(prefix);
    if (ind != std::string::npos) {
      line = line.substr(ind + prefix.size());
      count = std::stoul(line);
      // Assume there is only one instance of "count".
      break;
    }
  }
  pbtxtfile.close();
  return count;
}

int8_t set_count_from_pbtxt(const std::string& file_path, size_t count) {
  std::ifstream pbtxtfile(file_path);
  std::string line;
  std::string tmpname = file_path + ".tmp";
  std::ofstream tmpfile(tmpname);
  const std::string prefix = "count: ";
  while (std::getline(pbtxtfile, line)) {
    // Read the number of replicas from file.
    auto ind = line.find(prefix);
    if (ind != std::string::npos) {
      line = line.substr(0, ind + prefix.size());
      tmpfile << line << count << std::endl;
    } else {
      tmpfile << line << std::endl;
    }
  }
  pbtxtfile.close();
  tmpfile.close();
  auto res = rename(tmpname.c_str(), file_path.c_str());
  if (res != 0) {
    std::cerr << "Failed to rename " << tmpname << " to " << file_path
              << std::endl;
    return res;
  }

  return 0;
}

////////////////////////////////For GPU///////////////////////////////////////
std::map<std::string, size_t> GpuModelManager::model_to_num_reps_;
std::mutex GpuModelManager::update_mutex_;

int GpuModelManager::numReplicas(const std::string& model_name) {
  return model_to_num_reps_[model_name];
}

int8_t GpuModelManager::changeNumReplicas(const std::string& model_name,
                                          const size_t& count) {
  auto curr_count = numReplicas(model_name);
  if (curr_count == count) {
    // No change!
    return 0;
  } else if (count <= 0) {
    std::cerr << "count should larger than 0! Consider using UnloadModel"
              << std::endl;
    return -1;
  }
  // Write the new file.
  std::string pbtxt_path =
      local_trt_model_dir + model_name + "/" + trt_pbtxt_file;
  auto res = set_count_from_pbtxt(pbtxt_path, count);
  if (res < 0) { return res; }

  // Update number of replicas for the model.
  {
    std::lock_guard<std::mutex> lock(GpuModelManager::update_mutex_);
    model_to_num_reps_[model_name] = count;
  }

  // Make sure the model is ready to serve, because the model can be
  // temporarily unavailable.
  res = WaitGpuModelState(model_name, trtis::MODEL_READY, 1000);
  if (res != 0) {
    // Retry with larger interval.
    std::cout << "Retry..." << std::endl;
    // Wait every 5 sec.
    res = WaitGpuModelState(model_name, trtis::MODEL_READY, 5000);
    if (res != 0) { return res; }
  }
  return 0;
}

// For GPU executor, load a model means to copy the model to the correct worker
// directory on Google Storage
int8_t GpuModelManager::LoadModel(const std::string src_url,
                                  const std::string model_name,
                                  std::unique_ptr<RedisMetadata>& rmd,
                                  std::unique_ptr<S3Client>& s3c) {
#ifdef INFAAS_GPU_WORKER
  if (src_url.empty()) {
    std::cerr << "[GPU Manager] Empty source URL!" << std::endl;
    return -1;
  }
  if (model_name.empty()) {
    std::cerr << "[GPU Manager] Empty model name!" << std::endl;
    return -1;
  }
  uint64_t time1, time2;
  time1 = get_curr_timestamp();

  // Similar to CPU path, we use model_to_num_reps_ to prevent multiple threads
  // loading the same model.
  // If model_to_num_reps_ = -1, it means the model is currently being loaded
  // by another thread.
  {
    std::lock_guard<std::mutex> lock(GpuModelManager::update_mutex_);
    if ((model_to_num_reps_.find(model_name) != model_to_num_reps_.end() &&
         (model_to_num_reps_[model_name] != 0))) {
      std::cerr << "Model " << model_name << " is being loaded." << std::endl;
      return 1;
    } else {
      model_to_num_reps_[model_name] = -1;
    }
  }

  int8_t is_running = -1;
  is_running = rmd->is_model_running(model_name, worker_name_);
  if (is_running > 0) {
    std::cerr << "[GPU Manager] Inconsistency found: model " << model_name
              << " is already running!" << std::endl;
    return -1;
  } else if (is_running < 0) {
    std::cerr << "[GPU Manager] model " << model_name << " encountered error."
              << std::endl;
    return -1;
  }

  // Parse source bucket name and object name.
  size_t pre_ind = src_url.find(bucket_prefix);
  if ((pre_ind == std::string::npos) || (pre_ind != 0)) {
    std::cerr << "[GPU Manager] Not a valid bucket source: " << src_url
              << std::endl;
    // remember to clean up the map!
    {
      std::lock_guard<std::mutex> lock(GpuModelManager::update_mutex_);
      model_to_num_reps_[model_name] = 0;
    }
    return -1;
  }
  std::string obj_name, src_bucket;
  parse_s3_url(src_url, &src_bucket, &obj_name);
  std::cout << "[GPU Load model] src bucket name: " << src_bucket << "; object "
            << obj_name << std::endl;

  // Need to add a slash at the end to indicate directory (tensorrt models are
  // all in it's own directory).
  std::string trt_dst_url = local_trt_model_dir + model_name;
  std::string local_file_url = local_model_dir + model_name;
  if (trt_dst_url.back() != '/') { trt_dst_url += "/"; }
  if (local_file_url.back() != '/') { local_file_url += "/"; }
  if (!file_exist(local_file_url)) {
    uint64_t time3, time4;
    time3 = get_curr_timestamp();

    // List all objects with prefix = obj_name.
    // We assume that trt model requires a directory.
    if (obj_name.back() != '/') { obj_name += "/"; }
    std::vector<std::string> key_names;
    if (list_s3_path(src_bucket, obj_name, s3c, &key_names) != 0) {
      std::cerr << "[GPU Manager] failed to list the files" << std::endl;
      // remember to clean up the map!
      {
        std::lock_guard<std::mutex> lock(GpuModelManager::update_mutex_);
        model_to_num_reps_[model_name] = 0;
      }
      return -1;
    }
    if (key_names.size() <= 0) {
      std::cerr << "[CPU Manager] key_names size <= 0: " << obj_name
                << std::endl;
      return -1;
    }

    // Download file
    if (download_s3_local(src_bucket, obj_name, key_names, local_file_url,
                          s3c) != 0) {
      std::cerr << "Failed to download" << std::endl;
      // remember to clean up the map!
      {
        std::lock_guard<std::mutex> lock(GpuModelManager::update_mutex_);
        model_to_num_reps_[model_name] = 0;
      }
      return -1;
    }

    time4 = get_curr_timestamp();
    printf("[common_model_util.cc] GPU LoadModel - download files  %.4lf ms.\n",
           get_duration_ms(time3, time4));
  }

  // Copy files for TRT to serve.
  std::string command = "cp -r " + local_file_url + " " + local_trt_model_dir;
  std::cout << command << std::endl;
  if (system(command.c_str()) == -1) {
    std::cerr << "[GPU Manager] Failed to load model: " << model_name
              << " to worker " << worker_name_ << std::endl;
    {
      std::lock_guard<std::mutex> lock(GpuModelManager::update_mutex_);
      model_to_num_reps_[model_name] = 0;
    }
    return -1;
  }

  time2 = get_curr_timestamp();
  printf("[common_model_util.cc] LoadModel - total copy files  %.4lf ms.\n",
         get_duration_ms(time1, time2));

  // Wait until the model is ready to serve, wait interval 1sec.
  auto res = WaitGpuModelState(model_name, trtis::MODEL_READY, 1000);
  if (res != 0) {
    // Retry with larger interval.
    std::cout << "GPU Load model failed, retry..." << std::endl;
    // Wait every 5 sec.
    res = WaitGpuModelState(model_name, trtis::MODEL_READY, 5000);
    if (res != 0) {
      {
        std::lock_guard<std::mutex> lock(GpuModelManager::update_mutex_);
        model_to_num_reps_[model_name] = 0;
      }
      return res;
    }
  }

  // Update number of replicas for this model by checking the config.pbtxt file
  // to verify that.
  auto count = get_count_from_pbtxt(trt_dst_url + trt_pbtxt_file);
  {
    std::lock_guard<std::mutex> lock(GpuModelManager::update_mutex_);
    model_to_num_reps_[model_name] = count;
    rmd->add_running_model(worker_name_, model_name);
  }

#endif  // #ifdef INFAAS_GPU_WORKER
  return 0;
}

int8_t GpuModelManager::UnloadModel(const std::string& model_name,
                                    std::unique_ptr<RedisMetadata>& rmd) {
#ifdef INFAAS_GPU_WORKER
  if (model_name.empty()) {
    std::cerr << "[GPU Manager] Empty model name!" << std::endl;
    return -1;
  }

  // We assume the model is not available to serve queries once we started
  // unloading it.
  int8_t is_running = -1;
  int8_t res;
  is_running = rmd->is_model_running(model_name, worker_name_);
  if (is_running <= 0) {
    std::cerr << "[GPU Manager] model " << model_name << " is not running!"
              << std::endl;
    return 0;
  } else if (is_running < 0) {
    std::cerr << "[GPU Manager] model " << model_name << " encountered error."
              << std::endl;
    return -1;
  }
  std::string url = local_trt_model_dir + model_name;
  {
    std::lock_guard<std::mutex> lock(GpuModelManager::update_mutex_);
    // If the model is being loaded, don't remove.
    if (model_to_num_reps_[model_name] <= 0) {
      std::cerr << "[GPU Manager] model " << model_name
                << "is being loaded or removed. Don't unload again. #reps = "
                << model_to_num_reps_[model_name] << std::endl;
      return -1;
    }
    // Set model load/unload flag to prevent master sending future requests to
    // it.
    res = rmd->set_model_load_unload(model_name);
    if (res != 0) {
      std::cerr << "[GPU Manager] Failed to set model_load_unload flag for "
                << model_name << std::endl;
      // TODO: do we need handle the error here?
    }
  }
  // TODO: sometimes after removing the file trtserver can get into an
  // infinite loop. We suspect we need to wait until the model gets READY state.
  res = WaitGpuModelState(model_name, trtis::MODEL_READY, 500, 5);
  if (res != 0) {
    std::cerr << "[GPU Manager] Unload model failed to wait MODEL_READY for "
              << model_name << std::endl;
    return -1;
  }

  // If we unload the model from GPU, then the entire group will be unloaded.
  std::string command = "rm -rf " + url;
  std::cout << command << std::endl;
  if (system(command.c_str()) == -1) {
    std::cerr << "[GPU Manager] Failed to remove model directory: "
              << model_name << " from worker " << worker_name_ << std::endl;
    // TODO: we should probably not add back the model as running, because it
    // means the model is going wrong.
    return -1;
  } else {
    std::cerr << "Successfully removed " << url << std::endl;
  }
  // Wait until the model is unloaded, wait interval 500msec, at most 2sec.
  res = WaitGpuModelState(model_name, trtis::MODEL_UNAVAILABLE, 500, 4);
  if (res != 0) {
    // Add back model directory.
    std::cerr << "Failed to unload mdoel, add back " << model_name << std::endl;
    std::string local_file_url = local_model_dir + model_name;
    command = "cp -r " + local_file_url + " " + local_trt_model_dir;
    std::cout << command << std::endl;
    if (system(command.c_str()) == -1) {
      std::cerr << "Fatal error: failed to add back model " << model_name
                << std::endl;
    }
    return -1;
  }

  {
    std::lock_guard<std::mutex> lock(GpuModelManager::update_mutex_);
    rmd->remove_running_model(worker_name_, model_name);
    model_to_num_reps_[model_name] = 0;
    res = rmd->unset_model_load_unload(model_name);
    if (res != 0) {
      std::cerr << "[GPU Manager] Failed to unset model_load_unload flag for "
                << model_name << std::endl;
      // TODO: do we need to handle error here?
    }
  }

  std::cerr << "Successfully unload model " << model_name << std::endl;
#endif  // #ifdef INFAAS_GPU_WORKER
  return 0;
}

int8_t GpuModelManager::QueryModelOnline(const std::string& model_name,
                                         const QueryOnlineRequest* request,
                                         QueryOnlineResponse* reply,
                                         std::unique_ptr<RedisMetadata>& rmd,
                                         std::unique_ptr<S3Client>& s3c) {
#ifdef INFAAS_GPU_WORKER
  uint64_t time3, time4;
  time3 = get_curr_timestamp();
  if (model_name.empty()) {
    std::cerr << "[GPU Manager] Empty model name!" << std::endl;
    return -1;
  }
  // printf("[common_model_util.cc] manager QueryModelOnline start time:
  // %lu.\n",
  //        time3);

  int8_t res = 0;
  std::cout << "Getting GPU model state: " << model_name << std::endl;
  auto model_state = GpuModelState(model_name);
  if (model_state == trtis::MODEL_UNAVAILABLE) {
    std::cerr << "[GPU Manager] model " << model_name << " not loaded."
              << std::endl;
    std::string model_url = bucket_prefix + infaas_bucket + "/" + model_name;
    res = LoadModel(model_url, model_name, rmd, s3c);
    if (res < 0) {
      std::cerr << "[GPU Manager] Failed to load model: " << model_name
                << std::endl;
      return -1;
    }
  }
  // The model may be loaded by another thread and is unavailable right now.
  if (model_state != trtis::MODEL_READY) {
    res = WaitGpuModelState(model_name, trtis::MODEL_READY, 1000);
    if (res != 0) {
      // Retry with larger interval.
      std::cerr << "GPU Online query model not ready, retry..." << std::endl;
      // Wait every 5 sec.
      res = WaitGpuModelState(model_name, trtis::MODEL_READY, 5000);
      if (res != 0) { return res; }
    }
  }
  time4 = get_curr_timestamp();
  printf(
      "[common_model_util.cc] QueryModelOnline - check GpuModelState  %.4lf "
      "ms.\n",
      get_duration_ms(time3, time4));

  uint64_t time1, time2;
  time1 = get_curr_timestamp();
  // Check the model type.
  auto task = rmd->get_model_info(model_name, "task");
  // std::cout << "Model task is: " << task << std::endl;
  if (task.find("classification") != std::string::npos) {
    res = QueryGeneralModel(model_name, request->raw_input(),
                            reply->mutable_raw_output());
    time2 = get_curr_timestamp();
    printf("[common_model_util.cc] QueryModelOnline - infer  %.4lf ms.\n",
           get_duration_ms(time1, time2));

    return res;
  } else {
    std::cerr << "[GPU Manager] Could not support task: " << task << std::endl;
    return -1;
  }
#endif  // #ifdef INFAAS_GPU_WORKER
  return 0;
}

int8_t GpuModelManager::QueryGeneralModel(
    const std::string& model_name,
    const google::protobuf::RepeatedPtrField<std::string>& raw_input,
    google::protobuf::RepeatedPtrField<std::string>* raw_output) {
#ifdef INFAAS_GPU_WORKER
  if (model_name.empty()) {
    std::cerr << "[GPU Manager] Empty model name!" << std::endl;
    return -1;
  }

  uint64_t time1, time2;
  time1 = get_curr_timestamp();
  // Create the context for inference of the TRTIS model. Then use it to
  // validate the input.
  // Borrow ideas from:
  // https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/c++/image_client.cc
  std::unique_ptr<trtisc::InferContext> ctx;
  trtisc::Error err;
  err = trtisc::InferGrpcContext::Create(&ctx, trtis_grpc_url, model_name,
                                         /*model_version=latest*/ -1, false);
  if (!err.IsOk()) {
    std::cerr << "[GPU Manager] error: unable to create inference context: "
              << err << std::endl;
    return -1;
  }

  // Configure context for batch size and topk
  size_t batch_size = raw_input.size();
  std::cout << "batch size: " << batch_size << std::endl;
  std::unique_ptr<trtisc::InferContext::Options> options;
  err = trtisc::InferContext::Options::Create(&options);
  if (!err.IsOk()) {
    std::cerr << "[GPU Manager] failed initializing infer options: " << err
              << std::endl;
    return -1;
  }

  options->SetBatchSize(batch_size);
  // Add all output of this infer
  for (const auto& output : ctx->Outputs()) { options->AddRawResult(output); }
  err = ctx->SetRunOptions(*options);
  if (!err.IsOk()) {
    std::cerr << "[GPU Manager] Failed initializing batch size: " << err
              << std::endl;
    return -1;
  }
  time2 = get_curr_timestamp();
  printf(
      "[common_model_util.cc] QueryGeneralModel - prepare context  %.4lf ms.\n",
      get_duration_ms(time1, time2));
  // Actually send request of 'batch_size' images.
  const uint8_t* trtis_input = nullptr;
  std::map<std::string, std::unique_ptr<trtisc::InferContext::Result>> results;
  {
    // Assume there is only 1 input soruce.
    // TODO: some sequence models may require multiple input source, how to
    // handle that?
    const auto input = ctx->Inputs()[0];
    err = input->Reset();
    if (!err.IsOk()) {
      std::cerr << "[GPU Manager] Failed resetting input: " << err << std::endl;
      return -1;
    }

    // Set input to be the 'batch_size' images.
    for (size_t idx = 0; idx < batch_size; ++idx) {
      // TODO: need to find a better way to convert between raw input and TRTIS
      // input. This part has a potential of memory leak and stack overflow
      int raw_size = raw_input.Get(idx).size();
      trtis_input = reinterpret_cast<const uint8_t*>(&raw_input.Get(idx)[0]);
      err = input->SetRaw(trtis_input, raw_size);
      if (!err.IsOk()) {
        std::cerr << "[GPU Manager] Failed setting input: " << err << std::endl;
        return -1;
      }
      // std::cout << "Setting TRTIS input for idx=" << idx << std::endl;
    }
    time1 = get_curr_timestamp();
    printf(
        "[common_model_util.cc] QueryGeneralModel - prepare input  %.4lf ms.\n",
        get_duration_ms(time2, time1));
    // Send request synchronously.
    // TODO: may need to add async choice for offline path.
    std::cout << "Running TRTIS infer" << std::endl;
    err = ctx->Run(&results);
    if (!err.IsOk()) {
      std::cerr
          << "[GPU Manager] Failed sending synchronous TRTIS infer request: "
          << err << std::endl;
      return -1;
    }
  }

  time2 = get_curr_timestamp();
  printf("[common_model_util.cc] QueryGeneralModel - inference  %.4lf ms.\n",
         get_duration_ms(time1, time2));
  // std::cout << "Post process TRTIS result." << std::endl;
  // Post-process result.
  // TODO: we now assume there is only 1 output source. But some models could
  // have multiple outputs.
  if (results.size() != 1) {
    std::cerr << "[GPU Manager] Expected 1 result, got " << results.size()
              << std::endl;
    return -1;
  }

  const auto& result = results.begin()->second;
  const std::vector<uint8_t>* buf;
  for (size_t b = 0; b < batch_size; ++b) {
    err = result->GetRaw(b, &buf);
    if (!err.IsOk()) {
      std::cerr << "[GPU Manager] Failed get raw for batch " << b << ": " << err
                << std::endl;
      return -1;
    }
    // std::cout << "batch = " << b << ", buffer size is: " << buf->size() <<
    // std::endl;
    // TODO: this requires to copy the output. Think of a way to reduce that.
    raw_output->Add(std::string(buf->begin(), buf->end()));
  }

  time1 = get_curr_timestamp();
  printf("[common_model_util.cc] QueryGeneralModel - postprocess  %.4lf ms.\n",
         get_duration_ms(time2, time1));
#endif  // #ifdef INFAAS_GPU_WORKER
  return 0;
}

///////////////////////////////For CPU/////////////////////////////////////////
std::map<std::string, std::deque<std::string>>
    CpuModelManager::model_to_names_online_;
std::map<std::string, std::deque<std::string>>
    CpuModelManager::model_to_names_offline_;
std::map<std::string, int> CpuModelManager::name_to_port_;
std::set<int, std::greater<int>> CpuModelManager::used_ports_;
std::mutex CpuModelManager::update_mutex_;

size_t CpuModelManager::numReplicas(const std::string& model_name) {
  return model_to_names_online_[model_name].size();
}

// Load model on CPU means starting a container.
int8_t CpuModelManager::LoadModel(const std::string src_url,
                                  const std::string model_name,
                                  std::unique_ptr<RedisMetadata>& rmd,
                                  std::unique_ptr<S3Client>& s3c,
                                  const std::string container_name,
                                  bool for_online) {
  uint64_t time1, time2;
  time1 = get_curr_timestamp();
  std::map<std::string, std::deque<std::string>>& model_to_names =
      for_online ? model_to_names_online_ : model_to_names_offline_;

  if (src_url.empty()) {
    std::cerr << "[CPU Manager] Empty source URL!" << std::endl;
    return -1;
  }
  if (model_name.empty()) {
    std::cerr << "[CPU Manager] Empty model name!" << std::endl;
    return -1;
  }

  std::string instance_name =
      container_name.empty() ? model_name : container_name;
  std::cout << "Loading model " << model_name
            << " on CPU, instance/container name: " << instance_name
            << std::endl;

  int portnum;
  {
    std::lock_guard<std::mutex> lock(CpuModelManager::update_mutex_);
    if (name_to_port_.find(instance_name) != name_to_port_.end()) {
      // It is possible that the instance is found in name_to_port_ but not in
      // model_to_names, which means the model is being loaded but may not be
      // ready to serve.
      std::cout << "Instance " << instance_name << " is being loaded."
                << std::endl;
      return 1;
    }

    // If not added, then allocate a port number to the model.
    if (used_ports_.size() == 0) {
      portnum = first_cpu_port;
    } else {
      portnum = *(used_ports_.begin()) + 1;  // last port + 1
    }
    used_ports_.insert(portnum);
    name_to_port_[instance_name] = portnum;
  }

  // Download file if doesn't exist.
  std::string dst_url = local_model_dir + model_name;
  if (dst_url.back() != '/') { dst_url += "/"; }
  if (!file_exist(dst_url)) {
    uint64_t time3, time4;
    time3 = get_curr_timestamp();
    // Parse source bucket name and object name.
    size_t pre_ind = src_url.find(bucket_prefix);
    if ((pre_ind == std::string::npos) || (pre_ind != 0)) {
      std::cerr << "[CPU Manager] Not a valid bucket source: " << src_url
                << std::endl;
      // TODO: error handling here, should cleanup or retry?
      return -1;
    }
    std::string obj_name, src_bucket;
    parse_s3_url(src_url, &src_bucket, &obj_name);
    std::cout << "[CPU Load model] src bucket name: " << src_bucket
              << "; object " << obj_name << std::endl;

    // List all names and download
    // We assume that a model is within a directory.
    if (obj_name.back() != '/') { obj_name += "/"; }
    std::vector<std::string> key_names;
    if (list_s3_path(src_bucket, obj_name, s3c, &key_names) != 0) {
      std::cerr << "[CPU Manager] failed to list the files" << std::endl;
      // TODO: error handling here, should cleanup or retry?
      return -1;
    }
    if (key_names.size() <= 0) {
      std::cerr << "[CPU Manager] key_names size <= 0: " << obj_name
                << std::endl;
      return -1;
    }
    if (download_s3_local(src_bucket, obj_name, key_names, dst_url, s3c) != 0) {
      std::cerr << "[CPU Manager] Failed to load model: " << model_name
                << " to worker " << worker_name_ << std::endl;
      // TODO: error handling here, should cleanup or retry?
      return -1;
    }
    time4 = get_curr_timestamp();
    printf("[common_model_util.cc] CPU LoadModel - copy files  %.4lf ms.\n",
           get_duration_ms(time3, time4));
  }
  // Start the container
  auto framework = rmd->get_model_info(model_name, "framework");
  int input_dim = std::stoi(rmd->get_model_info(model_name, "img_dim"));
  int ready = -1;
  char docker_cmd[512];
  // Limit the cpu usage to cpu_per_container.
  int cpu_per_container = 1;
  // Get the number of CPUs requirement from the suffix of the name
  size_t rpos = model_name.rfind("_");
  if (rpos != std::string::npos) {
    std::string numstr = model_name.substr(rpos + 1);
    if (!numstr.empty() &&
        numstr.find_first_not_of("0123456789") == std::string::npos) {
      cpu_per_container = std::stoi(numstr);
    }
  }
  std::cout << "Num of cpu for container: " << cpu_per_container << std::endl;
  // Make sure the instance name is available.
  // A corner case here: the model is just being unloaded, and we need to load
  // it back at the same time.
  // Check the container name is valid to use.
  if (!WaitCpuInstanceStop(instance_name)) {
    std::cerr << "Instance " << instance_name << "is not stopped!" << std::endl;
    {
      std::lock_guard<std::mutex> lock(CpuModelManager::update_mutex_);
      auto it = name_to_port_.find(instance_name);
      if (it == name_to_port_.end()) {
        std::cout << "Model " << model_name << " has already unloaded."
                  << std::endl;
        return -1;
      }
      portnum = name_to_port_[instance_name];
      name_to_port_.erase(it);
      used_ports_.erase(portnum);
    }
    return -1;
  }

  if (framework == "pytorch") {
    std::string offline_nice("OFF");
    if (OFFLINE_CONTROL) { offline_nice = "ON"; }
    sprintf(docker_cmd,
            "docker run --rm -it -d -p%d:%d --cpus=%d --name=%s --ipc=host "
            "--cap-add=sys_nice -e OFFLINE_NICE=%s -v%s:/tmp/model "
            "-v%s:/tmp/infaas_input -v%s:/tmp/infaas_output "
            "qianl15/infaaspytorch:latest /workspace/container_start.sh "
            "pytorch_container.py %d %s %d",
            portnum, portnum, cpu_per_container, instance_name.c_str(),
            offline_nice.c_str(), local_model_dir.c_str(),
            local_input_dir.c_str(), local_output_dir.c_str(), input_dim,
            model_name.c_str(), portnum);
    ready = system(docker_cmd);
  } else if (framework == "tensorflow-cpu") {
    if (CPU_ADAPTIVE_BATCHING) {
      sprintf(docker_cmd,
              "docker run --rm -it -d -p%d:8501 --cpus=%d --name=%s --ipc=host "
              "--cap-add=sys_nice -v%s/%s:/models/%s -e MODEL_NAME=%s "
              "-v%s:/models/batching_parameters.txt -t tensorflow/serving "
              "--enable_batching=true "
              "--batching_parameters_file=/models/batching_parameters.txt",
              portnum, cpu_per_container, instance_name.c_str(),
              local_model_dir.c_str(), model_name.c_str(), model_name.c_str(),
              model_name.c_str(), batching_parameters_file.c_str());
    } else {
      sprintf(docker_cmd,
              "docker run --rm -it -d -p%d:8501 --cpus=%d --name=%s --ipc=host "
              "--cap-add=sys_nice -v%s:/models -e MODEL_NAME=%s "
              "tensorflow/serving",
              portnum, cpu_per_container, instance_name.c_str(),
              local_model_dir.c_str(), model_name.c_str());
    }
    ready = system(docker_cmd);
  } else {
    std::cerr << "Unsupport framework: " << framework << std::endl;
    ready = -1;
  }
  if (ready == -1) {
    std::cerr << "[CPU Manager] Docker cmd failed or unsupported: "
              << std::endl;
    {
      std::lock_guard<std::mutex> lock(CpuModelManager::update_mutex_);
      auto it = name_to_port_.find(instance_name);
      if (it == name_to_port_.end()) {
        std::cout << "Model " << model_name << " has already unloaded."
                  << std::endl;
        return -1;
      }
      portnum = name_to_port_[instance_name];
      name_to_port_.erase(it);
      used_ports_.erase(portnum);
    }
    return -1;
  }

  // Check the container is running
  sprintf(docker_cmd, "docker ps -aqf \"name=%s\"", instance_name.c_str());
  std::string retstr = execcmd(docker_cmd);
  // Should return the docker ID.
  if (retstr.empty()) {
    std::cerr << "[CPU Manager] Failed to start model instance: "
              << instance_name << " to worker " << worker_name_ << std::endl;
    {
      std::lock_guard<std::mutex> lock(CpuModelManager::update_mutex_);
      auto it = name_to_port_.find(instance_name);
      if (it == name_to_port_.end()) {
        std::cout << "Model " << model_name << " has already unloaded."
                  << std::endl;
        return -1;
      }

      portnum = name_to_port_[instance_name];
      name_to_port_.erase(it);
      used_ports_.erase(portnum);
    }

    return -1;
  }

  // Attention!! the gRPC can fail if we try to query right after the
  // container is started. Need a way to make sure the container is ready to
  // serve.

  // First, set the interval to 100ms
  if (!WaitCpuModelReady(model_name, framework, portnum, 100)) {
    std::cerr << "[CPU Manager] Heartbeat failed! Retry..." << std::endl;
    // Retry with interval 1s.
    if (!WaitCpuModelReady(model_name, framework, portnum, 1000)) {
      std::lock_guard<std::mutex> lock(CpuModelManager::update_mutex_);
      // Check whether the model has been loaded by other threads.
      auto it = name_to_port_.find(instance_name);
      if (it == name_to_port_.end()) {
        std::cout << "Model " << model_name << " has already unloaded."
                  << std::endl;
        return -1;
      }
      portnum = name_to_port_[instance_name];
      name_to_port_.erase(it);
      used_ports_.erase(portnum);
      return -1;
    }
  }
  time2 = get_curr_timestamp();
  printf("[common_model_util.cc] CPU LoadModel - total time  %.4lf ms.\n",
         get_duration_ms(time1, time2));

  // Update instance name to the map. Push to the queue only if everything is
  // fine.
  {
    std::lock_guard<std::mutex> lock(CpuModelManager::update_mutex_);
    // If it is the first instance of that model variant and it is for online
    // query, then update to metadata store.
    if (model_to_names[model_name].empty() && for_online) {
      rmd->add_running_model(worker_name_, model_name);
      std::string parent_mod = rmd->get_parent_model(model_name);
      rmd->unset_parent_scaledown(worker_name_, parent_mod);
    }
    model_to_names[model_name].push_back(instance_name);
  }
  std::cout << "Model is ready to serve with docker: " << retstr << std::endl;

  return 0;
}

// Unload one model instance on CPU means stopping a container.
int8_t CpuModelManager::UnloadModel(const std::string& model_name,
                                    std::unique_ptr<RedisMetadata>& rmd,
                                    const std::string container_name,
                                    bool for_online) {
  int portnum;
  if (model_name.empty()) {
    std::cerr << "[CPU Manager] Empty model name!" << std::endl;
    return -1;
  }
  std::cout << "Unload model instance of " << model_name << " from CPU."
            << std::endl;
  std::map<std::string, std::deque<std::string>>& model_to_names =
      for_online ? model_to_names_online_ : model_to_names_offline_;
  std::string instance_name;
  {
    std::lock_guard<std::mutex> lock(CpuModelManager::update_mutex_);
    // We assume the model is not available to serve queries once we started
    // unloading it.
    auto it = model_to_names.find(model_name);
    std::deque<std::string>& names_queue = it->second;
    if ((it == model_to_names.end()) || names_queue.empty()) {
      std::cout << "Model " << model_name << " has already unloaded."
                << std::endl;
      return 0;
    }

    // Remove the instance from back if unspecified name.
    instance_name =
        container_name.empty() ? names_queue.back() : container_name;
    std::cout << "Removing instance " << instance_name << std::endl;
    auto deq_it =
        std::find(names_queue.begin(), names_queue.end(), instance_name);
    auto ntp_it = name_to_port_.find(instance_name);
    if (ntp_it == name_to_port_.end()) {
      if (deq_it == names_queue.end()) {
        std::cout << "Instance " << instance_name << " is unloaded / not found"
                  << std::endl;
        return 0;
      } else {
        std::cerr << "Inconsistency found! instance name is not in "
                     "name_to_port_ but appears in model_to_names. \n";
        return -1;
      }
    }
    names_queue.erase(deq_it);
    portnum = ntp_it->second;
    name_to_port_.erase(ntp_it);
    used_ports_.erase(portnum);

    int8_t is_running = -1;
    is_running = rmd->is_model_running(model_name, worker_name_);
    if (!is_running && for_online) {
      std::cerr << "Model is not running! inconsistency found." << std::endl;
      return -1;
    }
    if (names_queue.empty() && for_online) {
      // Set model load/unload flag.
      int8_t res = rmd->set_model_load_unload(model_name);
      if (res != 0) {
        std::cerr << "Failed to set model_load_unload flag for " << model_name
                  << std::endl;
      }
      res = rmd->remove_running_model(worker_name_, model_name);
      if (res != 0) {
        std::cerr << "Failed to remove running model: " << int(res)
                  << std::endl;
        return -1;
      }
    }
  }
  char docker_cmd[200];
  // TODO: need to find a way to gracefully exit the container. Probably the
  // container should wait until all requests being serviced.
  sprintf(docker_cmd, "docker stop -t %d %s", DOCKER_STOP_TIME,
          instance_name.c_str());
  std::string retstr = execcmd(docker_cmd);
  if (retstr.find(instance_name) == std::string::npos) {
    std::cerr << "[CPU Manager] Docker cmd failed: " << retstr
              << "expected: " << instance_name << std::endl;
    std::cerr << "Find at pos " << retstr.find(instance_name) << std::endl;
    return -1;
  }
  if (for_online) {
    int8_t res = rmd->unset_model_load_unload(model_name);
    if (res != 0) {
      std::cerr << "Failed to unset model_load_unload flag for " << model_name
                << std::endl;
    }
  }
  return 0;
}

int8_t CpuModelManager::QueryModelOnline(const std::string& model_name,
                                         const QueryOnlineRequest* request,
                                         QueryOnlineResponse* reply,
                                         std::unique_ptr<RedisMetadata>& rmd,
                                         std::unique_ptr<S3Client>& s3c) {
  if (model_name.empty()) {
    std::cerr << "[CPU Manager] model name is empty!" << std::endl;
    return -1;
  }

  // Need to wait until model is loaded.
  int numtry = 0, maxtry = 60;
  int sleep_interval = 500;  // Sleep 500 ms.
  while (numtry < maxtry) {
    // It is possible that the model is being loaded but not ready to serve. So
    // model_to_names_online_ will not have the instance available. Therefore,
    // we need to wait.
    if ((model_to_names_online_.find(model_name) ==
         model_to_names_online_.end()) ||
        (model_to_names_online_[model_name].size() <= 0)) {
      // Load the model
      std::string src_url = bucket_prefix + infaas_bucket + "/" + model_name;
      auto res = LoadModel(src_url, model_name, rmd, s3c,
                           model_name + "_online_0", true);
      if (res < 0) {
        std::cerr << "[CPU Manager] QueryModelOnline - Failed to load model: "
                  << model_name << std::endl;
        return res;
      } else if (res > 0) {
        // If res > 0, means the model is being loaded by others, still need to
        // wait.
        std::cout << "Model " << model_name
                  << " is being loaded by others but not available \n";
      } else {
        // If res == 0, it means the model is loaded by this thread.
        break;
      }
    } else {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
    numtry++;
    std::cout << "Try #" << numtry << " failed, try again." << std::endl;
  }
  if (numtry == maxtry) {
    std::cerr << "[CPU Manager] Timed out waiting for model loading..."
              << std::endl;
    return -1;
  }

  // Schedule in a round robin way - get one from the front and push to the
  // back.
  // TODO: this may need change. We need a way to avoid changing the queue.
  std::string instance_name;
  {
    std::lock_guard<std::mutex> lock(CpuModelManager::update_mutex_);
    instance_name = model_to_names_online_[model_name].front();
    model_to_names_online_[model_name].pop_front();
    model_to_names_online_[model_name].push_back(instance_name);
  }

  int portnum = name_to_port_[instance_name];
  std::cout << "Serve with instance: " << instance_name
            << "; port number: " << std::to_string(portnum) << std::endl;
  auto framework = rmd->get_model_info(model_name, "framework");

  if (framework == "pytorch") {
    struct Address dest_addr = {"localhost", std::to_string(portnum)};
    grpc::ChannelArguments arguments;
    arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
    arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);
    std::unique_ptr<Query::Stub> stub_(Query::NewStub(grpc::CreateCustomChannel(
        RedisMetadata::Address_to_str(dest_addr),
        grpc::InsecureChannelCredentials(), arguments)));

    ClientContext context;

    uint64_t time1, time2;
    time1 = get_curr_timestamp();
    // Actual inference
    Status status = stub_->QueryOnline(&context, *request, reply);
    time2 = get_curr_timestamp();
    printf("[common_model_util.cc] Pytorch inference time  %.4lf ms.\n",
           get_duration_ms(time1, time2));
  } else if (framework == "tensorflow-cpu") {
    const google::protobuf::RepeatedPtrField<std::string>& raw_input =
        request->raw_input();
    size_t batch_size = raw_input.size();
    CURL* curl;
    CURLcode curl_res;
    curl = curl_easy_init();
    if (!curl) {
      std::cerr << "failed to post request to model " << model_name
                << std::endl;
    }
    uint64_t time1, time2;
    time1 = get_curr_timestamp();

    // Encode each input as base64
    std::string curl_reqs = "{ \"instances\": [";
    for (size_t idx = 0; idx < batch_size; ++idx) {
      unsigned int raw_size = raw_input.Get(idx).size();
      unsigned char const* bytes_to_encode =
          reinterpret_cast<unsigned char const*>(&raw_input.Get(idx)[0]);
      std::string base64_str = base64_encode(bytes_to_encode, raw_size);
      // std::replace(base64_str.begin(), base64_str.end(), '+', '-');
      // std::replace(base64_str.begin(), base64_str.end(), '/', '_');
      if (idx > 0) { curl_reqs += ","; }
      curl_reqs += "{\"b64\": \"" + base64_str + "\" }";
    }
    curl_reqs += "] }";
    // std::cout << "curl_reqs string size: " << curl_reqs.size() << std::endl;
    std::string readbuff;
    std::string tf_url = "http://localhost:" + std::to_string(portnum) +
                         "/v1/models/" + model_name + ":predict";

    // Now post the request
    struct curl_slist* curl_list = NULL;
    curl_list = curl_slist_append(curl_list, "Content-Type: application/json");
    curl_list = curl_slist_append(curl_list, "charsets: utf-8");
    curl_easy_setopt(curl, CURLOPT_URL, tf_url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_list);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, curl_reqs.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, curl_reqs.length());
    curl_easy_setopt(curl, CURLOPT_POST, 1);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlWriteCallBack);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readbuff);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_res = curl_easy_perform(curl);

    // Process output
    if (readbuff.size() < 1000) {
      std::cout << "Readbuff: " << readbuff << std::endl;
    }
    json bodyjs = json::parse(readbuff);
    auto& predjs = bodyjs["predictions"];
    // One batch at a time.
    float* predv = nullptr;
    for (json::iterator it = predjs.begin(); it != predjs.end(); ++it) {
      // std::cout << "Size: " << it->size() << std::endl;
      if (predv == nullptr) { predv = new float[it->size()]; }
      // Turn the string of floating points into raw bytes.
      int cntit = 0;
      for (auto& i : *it) {
        predv[cntit] = std::stof(i.dump());
        cntit++;
      }
      reply->add_raw_output(reinterpret_cast<const char*>(predv),
                            it->size() * sizeof(float));
    }
    if (predv) { delete[] predv; }
    curl_easy_cleanup(curl);
    curl_slist_free_all(curl_list);
    time2 = get_curr_timestamp();
    printf("[common_model_util.cc] TF-CPU inference time  %.4lf ms.\n",
           get_duration_ms(time1, time2));
  } else {
    std::cerr << "Don't support framework " << framework << std::endl;
    return -1;
  }
  // std::cout << "raw_output batch size: " << reply->raw_output().size()
  //          << "; each dimension: " << reply->raw_output()[0].size() <<
  //          std::endl;
  return 0;
}

int8_t CpuModelManager::QueryModelOffline(const std::string& model_name,
                                          const QueryOfflineRequest& request,
                                          std::unique_ptr<RedisMetadata>& rmd,
                                          std::unique_ptr<S3Client>& s3c) {
  std::string input_url = bucket_prefix + request.input_url();
  std::string output_url = bucket_prefix + request.output_url();
  std::string submitter = request.submitter();
  // NOTE: one level deeper for the folder.
  std::string local_instance_name = model_name + "_" + submitter;
  std::string local_input =
      local_input_dir + local_instance_name + "/" + local_input_leaf_dir;
  std::string local_output = local_output_dir + local_instance_name + "/";
  // Create input and output directory.
  auto res = create_dir(local_output);
  if (res < 0) {
    std::cerr << "Failed to create local output directory: " << local_output
              << "errno: " << errno << std::endl;
    return -1;
  }
  res = create_dir(local_input);
  if (res < 0) {
    std::cerr << "Failed to create local input directory: " << local_input
              << std::endl;
    return -1;
  }

  // Fix/pad the folder names.
  if (input_url.back() != '/') { input_url = input_url + "/"; }
  if (output_url.back() != '/') { output_url = output_url + "/"; }
  printf(
      "Process offline request... input: %s, output: %s, model: %s, submitter: "
      "%s.\n",
      input_url.c_str(), output_url.c_str(), model_name.c_str(),
      submitter.c_str());
  printf("Local input dir: %s; local output dir: %s \n", local_input.c_str(),
         local_output.c_str());
  fflush(stdout);

  // 0. Load model if not loaded.
  // Need to wait until model is loaded.
  int numtry = 0, maxtry = 60;
  int sleep_interval = 500;  // Sleep 500 ms.
  while (numtry < maxtry) {
    // It is possible that the model is being loaded but not ready to serve. So
    // model_to_names_offline_ will not have the instance available. Therefore,
    // we need to wait.
    if ((model_to_names_offline_.find(model_name) ==
         model_to_names_offline_.end()) ||
        (model_to_names_offline_[model_name].size() <= 0)) {
      // Load the model
      std::string src_url = bucket_prefix + infaas_bucket + "/" + model_name;
      auto res = LoadModel(src_url, model_name, rmd, s3c,
                           model_name + "_offline_0", false);
      if (res < 0) {
        std::cerr << "[CPU Manager] QueryModelOffline - Failed to load model: "
                  << model_name << std::endl;
        return -1;
      } else if (res > 0) {
        // If res > 0, means the model is being loaded by others, still need to
        // wait.
        std::cout << "Model " << model_name
                  << " is being loaded by others but not available \n";
      } else {
        // If res == 0, it means the model is loaded by this thread.
        break;
      }
    } else {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
    numtry++;
    std::cout << "Try #" << numtry << " failed, try again." << std::endl;
  }

  // Schedule in a round robin way - get one from the front and push to the
  // back.
  std::string instance_name;
  {
    std::lock_guard<std::mutex> lock(CpuModelManager::update_mutex_);
    instance_name = model_to_names_offline_[model_name].front();
    model_to_names_offline_[model_name].pop_front();
    model_to_names_offline_[model_name].push_back(instance_name);
  }

  int portnum = name_to_port_[instance_name];
  std::cout << "Offline serve with instance: " << instance_name
            << "; port number: " << std::to_string(portnum) << std::endl;

  struct Address dest_addr = {"localhost", std::to_string(portnum)};
  grpc::ChannelArguments arguments;
  arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
  arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);
  std::unique_ptr<Query::Stub> stub_(Query::NewStub(grpc::CreateCustomChannel(
      RedisMetadata::Address_to_str(dest_addr),
      grpc::InsecureChannelCredentials(), arguments)));

  QueryOfflineRequest container_request;
  QueryOfflineResponse reply;
  container_request.set_input_url(local_instance_name);
  container_request.add_model(model_name);
  container_request.set_output_url(local_instance_name);
  container_request.mutable_slo()->CopyFrom(request.slo());
  container_request.set_submitter(submitter);

  // 1. Get the names of all input files (just file names, not full path).
  std::string obj_name, src_bucket;
  std::vector<std::string> input_names;
  parse_s3_url(input_url, &src_bucket, &obj_name);
  std::cout << "Offline src_bucket: " << src_bucket
            << ", obj_name: " << obj_name << std::endl;
  if (list_s3_path(src_bucket, obj_name, s3c, &input_names) != 0) {
    std::cerr << "[Offline] Failed to list input bucket." << std::endl;
    return -1;
  }

  // 2. Process one batch at a time.
  int total_num = input_names.size();
  for (int iter = 0; iter < total_num; iter += offline_batch) {
    int batch = (iter + offline_batch >= total_num) ? (total_num - iter)
                                                    : offline_batch;
    // Download 'batch' files from the bucket.
    if (download_s3_local(
            src_bucket, obj_name,
            {input_names.begin() + iter, input_names.begin() + iter + batch},
            local_input, s3c) != 0) {
      std::cerr << "Failed to download #batch " << iter;
    }

    // Execute
    // Wait until the CPU util is below the threshold, avoiding interference
    // with online queries.
    if (OFFLINE_CONTROL) {
      double cpu_util = rmd->get_cpu_util(worker_name_);
      std::cout << "[common_model_util.cc] cpu util from offline: " << cpu_util
                << std::endl;
      bool has_blacklisted = CommonModelUtil::HasBlacklisted();
      while ((cpu_util > offline_cpu_thresh) || has_blacklisted) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
        cpu_util = rmd->get_cpu_util(worker_name_);
        has_blacklisted = CommonModelUtil::HasBlacklisted();
        std::cout << "[common_model_util.cc] try again, cpu util from offline: "
                  << cpu_util << "; blacklisted: " << has_blacklisted
                  << std::endl;
      }
    } else {
      std::cout << "No control over offline, run anyway." << std::endl;
    }

    ClientContext context;  // context should not be reused!
    Status status = stub_->QueryOffline(&context, container_request, &reply);
    if (!status.ok() ||
        !(reply.status().status() == InfaasRequestStatusEnum::SUCCESS)) {
      std::cerr << "Offline Execution failed: RPC: "
                << status.error_message() + "; INFaaS: " << reply.status().msg()
                << std::endl;
    }
    // Upload output directory
    std::vector<std::string> output_names;
    std::string command, retstr;
    std::string outobj_name, outsrc_bucket;
    parse_s3_url(output_url, &outsrc_bucket, &outobj_name);
    std::cout << "Offline output bucket: " << outsrc_bucket
              << ", outobj_name: " << outobj_name << std::endl;
    if (list_local_path(local_output, &output_names) != 0) {
      std::cerr << "Failed to list local output directory " << local_output
                << std::endl;
      // TODO: need to handle error?
    } else {
      // Upload to S3.
      if (upload_local_s3(local_output, output_names, outsrc_bucket,
                          outobj_name, s3c) != 0) {
        std::cerr << "Failed to upload to output_url: " << output_url;
        // TODO: need to handle error? For now we just don't remove the
        // directory.
      } else {
        command = "exec rm -r " + local_output + "*";
        retstr = execcmd(command.c_str());
        if (!retstr.empty()) {
          std::cerr << "Command" << command << " returned error: " << retstr
                    << std::endl;
        }
      }
    }
    // Cleanup input directory for the next batch.
    command = "exec rm -r " + local_input + "*";
    retstr = execcmd(command.c_str());
    if (!retstr.empty()) {
      std::cerr << "Command" << command << " returned error: " << retstr
                << std::endl;
    }
  }

  // Remove input/output directories
  if (rmdir(local_output.c_str()) != 0) {
    std::cerr << "Failed to remove " << local_output << std::endl;
    return -1;
  }
  if (rmdir(local_input.c_str())) {
    if (rmdir((local_input_dir + local_instance_name).c_str())) {
      std::cerr << "Failed to remove " << local_input << std::endl;
      return -1;
    }
  }
  std::cout << "Removed input and output directories" << std::endl;

  // Unload the offline model.
  // TODO: for now unload the model after one request. We might want to do
  // autoscaling for offline in the future.
  res = UnloadModel(model_name, rmd, model_name + "_offline_0", false);
  if (res != 0) {
    std::cerr << "Failed to unload offline model " << model_name + "_offline_0"
              << std::endl;
    return -1;
  }
  return 0;
}

int8_t CpuModelManager::QueryImageClassModel(
    const std::string& model_name,
    const google::protobuf::RepeatedPtrField<std::string>& raw_input,
    google::protobuf::RepeatedPtrField<std::string>* raw_output, size_t topk) {
  // TODO: This function may go deprecated.
  return 0;
}

// This function should be deprecated, but keep for future reference.
int8_t GpuModelManager::QueryImageClassModel(
    const std::string& model_name,
    const google::protobuf::RepeatedPtrField<std::string>& raw_input,
    google::protobuf::RepeatedPtrField<std::string>* raw_output, size_t topk) {
#ifdef INFAAS_GPU_WORKER
  // TODO: This function may go deprecated
#endif  // #ifdef INFAAS_GPU_WORKER.
  return 0;
}

}  // namespace internal
}  // namespace infaas
