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

// This file is used to test that both query executor and query client can work
// correctly.
#include <assert.h>
#include <unistd.h>
#include <cstdint>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>

#include "common_model_util.h"
#include "include/constants.h"
#include "metadata-store/redis_metadata.h"
#include "query_client.h"

using Aws::S3::S3Client;

#define FAIL(x) printf("[FAIL]: %s\n", x)
#define PASS(x) printf("[PASS]: %s\n", x)

struct test_model_info {
  std::string model_name;
  double comp_size;
  double acc;
  std::string dataset;
  std::string submitter;
  std::string framework;
  int64_t batch;
  std::string task;
  double load_lat;
  double inf_lat;
  double peak_mem;
  int16_t input_dim;
  double slope;
  double intercept;
};

static const std::string grandparent = "image_classification-imagenet-224";
static const std::string grandparent_translate = "translation";
static const struct test_model_info test_model = {
    "resnet_v1_50", 100,        71.2,     "imagenet",
    "me",           "tensorrt", 8,        "image_classification",
    1146.433452,    3.686,      92275040, 224,
    1.1902,         2.7069};
static const struct test_model_info test_cpu_model = {"resnet50_pytorch_4",
                                                      98,
                                                      71.2,
                                                      "imagenet",
                                                      "me",
                                                      "pytorch",
                                                      64,
                                                      "image_classification",
                                                      1524.285746,
                                                      499.30,
                                                      66460000,
                                                      224,
                                                      241.28,
                                                      247.86};
static const struct test_model_info test_tfgpu_model = {"resnet50_tf_gpu_nhwc",
                                                        98,
                                                        71.2,
                                                        "imagenet",
                                                        "me",
                                                        "tensorflow-gpu",
                                                        64,
                                                        "image_classification",
                                                        1307.610,
                                                        7.113,
                                                        2097160,
                                                        224,
                                                        1.6825,
                                                        5.2513};
static const struct test_model_info test_tfcpu_model = {
    "resnet50_tensorflow-cpu_4",
    98,
    71.2,
    "imagenet",
    "me",
    "tensorflow-cpu",
    64,
    "image_classification",
    1589.348217,
    250,
    3084000,
    224,
    186.35,
    57.811};

static const struct test_model_info test_infa_model = {
    "resnet50_inferentia_1_1",
    98,
    71.2,
    "imagenet",
    "me",
    "inferentia",
    1,
    "image_classification",
    3554.77,
    22,
    42079527,
    224,
    22,
    0};

static const struct test_model_info test_gnmt_cpu_model = {
    "gnmt_ende4_cpu_fp32_2",
    24.5,
    24.5,
    "wmt16_de_en",
    "me",
    "gnmt-nvpy-cpu",
    128,
    "translation",
    4000,
    700,
    1500000,
    0,
    186.35,
    57.811};

static const struct test_model_info test_gnmt_gpu_model = {
    "gnmt_ende4_gpu_fp16_2",
    24.5,
    24.5,
    "wmt16_de_en",
    "me",
    "gnmt-nvpy-gpu",
    128,
    "translation",
    4000,
    100,
    1500000,
    0,
    186.35,
    57.811};

// This address should not be running anything.
static const std::string invalid_addr = "localhost:50052";
static const std::string query_exe_addr = "localhost:50051";

static const int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;

namespace infaas {
namespace internal {
namespace {
// Timestamp to millisecond duration.
double ts_to_ms(const struct timeval& start, const struct timeval& end) {
  return (end.tv_sec - start.tv_sec) * 1000.0 +
         (end.tv_usec - start.tv_usec) / 1000.0;
}

// Test monitoring daemon can write to metadata store.
bool test_monitoring(const std::string& worker_name,
                     std::unique_ptr<RedisMetadata>& rmd) {
  std::string testcase = "CPU Utilization Monitoring Daemon";
  sleep(5);  // Sleep 5 sec to wait at least one measurement.
  double cpu_util = rmd->get_cpu_util(worker_name);
  std::cout << "Worker CPU util: " << cpu_util << std::endl;
  if (cpu_util < 0.0) {
    FAIL(std::string(testcase + " - cpu util less than 0").c_str());
    return false;
  }
  PASS(testcase.c_str());

  testcase = "GPU Utilization Monitoring Daemon";
#ifdef INFAAS_GPU_WORKER
  double gpu_util = rmd->get_gpu_util(worker_name);
  std::cout << "Worker GPU util: " << gpu_util << std::endl;
  if (gpu_util < 0.0) {
    FAIL(std::string(testcase + "- gpu util less than 0").c_str());
    return false;
  }
  PASS(testcase.c_str());
#endif
#ifdef INFAAS_NEURON_WORKER
  testcase = "Inferentia Utilization Monitoring Daemon";
  double infa_util = rmd->get_inferentia_util(worker_name);
  std::cout << "Worker Inferentia util: " << infa_util << std::endl;
  if ((infa_util < 0.0) || (infa_util > 100.0)) {
    FAIL(std::string(testcase + "- incorrect inferentia util").c_str());
    return false;
  }
  PASS(testcase.c_str());
#endif
  return true;
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

// Test heartbeat is correct
bool test_heartbeat(QueryClient& invalid_client, QueryClient& query_client) {
  auto inv_reply = invalid_client.Heartbeat();
  if (inv_reply.status() == InfaasRequestStatusEnum::INVALID) {
    PASS("Heartbeat-invalid");
  } else {
    FAIL("Heartbeat-invalid");
    return false;
  }

  auto reply = query_client.Heartbeat();
  if (reply.status() == InfaasRequestStatusEnum::SUCCESS) {
    PASS("Heartbeat");
  } else {
    std::string errmsg = "Heartbeat. Error msg: " + reply.msg();
    FAIL(errmsg.c_str());
    return false;
  }
  return true;
}

// Test the model can be loaded and unloaded on CPU.
bool test_model_util_cpu_basic(const std::string& worker_name,
                               const std::string& model_name,
                               const struct Address& redis_addr,
                               std::unique_ptr<RedisMetadata>& rmd,
                               std::unique_ptr<S3Client>& s3c) {
  CpuModelManager manager(worker_name);
  CpuModelManager manager2(worker_name);
  std::string testcase = "";
  // Test load model.
  std::string src_url = bucket_prefix + infaas_bucket + "/" + model_name;
  auto res = manager.LoadModel(src_url, model_name, rmd, s3c);
  int8_t is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "CPU Load Model - " + model_name;
  if ((res == 0) && is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  res = manager.LoadModel(src_url, model_name, rmd, s3c);
  is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "CPU Load Model again - " + model_name;
  if ((res > 0) && is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  // Load a replica
  res = manager2.LoadModel(src_url, model_name, rmd, s3c, "replica");
  is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "CPU Load Model 2 - " + model_name;
  if ((res == 0) && is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  // Test unload a model.
  res = manager.UnloadModel(model_name, rmd);
  is_running = rmd->is_model_running(model_name, worker_name);
  // We loaded twice, so the model is still running
  testcase = "CPU Unload Model - " + model_name;
  if ((res == 0) && is_running) {
    PASS(testcase.c_str());
  } else {
    std::cerr << "res, is_running: " << int(res) << ", " << int(is_running)
              << std::endl;
    FAIL(testcase.c_str());
    return false;
  }

  // We assume that it will unload replica first, then the model_name container.
  res = manager2.UnloadModel(model_name, rmd, model_name, true);
  is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "CPU Unload Model 2 - " + model_name;
  if ((res == 0) && !is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  // Load and unload for Offline path.
  res = manager2.LoadModel(src_url, model_name, rmd, s3c, "offline1", false);
  is_running = rmd->is_model_running(model_name, worker_name);
  // Offline model should not reflect in metadata
  testcase = "CPU Load Model Offline - " + model_name;
  if ((res == 0) && !is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  res = manager2.UnloadModel(model_name, rmd, "", false);
  is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "CPU Unload Model Offline - " + model_name;
  if ((res == 0) && !is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  res = manager.LoadModel(src_url, model_name, rmd, s3c, "offline1", false);
  is_running = rmd->is_model_running(model_name, worker_name);
  // Offline model should not reflect in metadata
  testcase = "CPU Load Model Offline 2- " + model_name;
  if ((res == 0) && !is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  res = manager.UnloadModel(model_name, rmd, "", false);
  is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "CPU Unload Model Offline 2- " + model_name;
  if ((res == 0) && !is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  return true;
}

// Test the model can be loaded and unloaded on GPU.
bool test_model_util_gpu_basic(const std::string& worker_name,
                               const struct Address& redis_addr,
                               std::unique_ptr<RedisMetadata>& rmd,
                               std::unique_ptr<S3Client>& s3c) {
  GpuModelManager manager(worker_name);

  // Test load model.
  std::string model_name = test_model.model_name;
  std::string src_url = bucket_prefix + infaas_bucket + "/" + model_name;
  auto res = manager.LoadModel(src_url, model_name, rmd, s3c);
  int8_t is_running = rmd->is_model_running(model_name, worker_name);
  if ((res == 0) && is_running) {
    PASS("GPU Load Model");
  } else {
    FAIL("GPU Load Model");
    return false;
  }

  std::string pbtxt_path = "/tmp/trtmodels/" + model_name + "/config.pbtxt";
  auto count = get_count_from_pbtxt(pbtxt_path);
  if (count > 0) {
    PASS("GPU get_count_from_pbtxt");
  } else {
    FAIL("GPU get_count_from_pbtxt");
    return false;
  }

  res = set_count_from_pbtxt(pbtxt_path, count + 2);
  auto count2 = get_count_from_pbtxt(pbtxt_path);
  // TODO: need to check with TRT server that the count really increased.
  // However, the current version of TRT server (19.01) has a bug that cannot
  // show the updated config. But this bug should be fixed for the next release.
  if ((res == 0) && (count2 - count == 2)) {
    PASS("GPU set_count_from_pbtxt");
  } else {
    std::cout << "count: " << count << ", count2: " << count2 << std::endl;
    FAIL("GPU set_count_from_pbtxt");
    return false;
  }

  // Test unload a model.
  res = manager.UnloadModel(model_name, rmd);
  is_running = rmd->is_model_running(model_name, worker_name);
  if ((res == 0) && !is_running) {
    PASS("GPU Unload Model");
  } else {
    FAIL("GPU Unload Model");
    return false;
  }

  // Test load a Tensorflow GPU model.
  model_name = test_tfgpu_model.model_name;
  src_url = bucket_prefix + infaas_bucket + "/" + model_name;
  res = manager.LoadModel(src_url, model_name, rmd, s3c);
  is_running = rmd->is_model_running(model_name, worker_name);
  if ((res == 0) && is_running) {
    PASS("GPU Load Model - Tensorflow GPU");
  } else {
    FAIL("GPU Load Model - Tensorflow GPU");
    return false;
  }

  // Test unload a Tensorflow GPU model.
  res = manager.UnloadModel(model_name, rmd);
  is_running = rmd->is_model_running(model_name, worker_name);
  if ((res == 0) && !is_running) {
    PASS("GPU Unload Model - Tensorflow GPU");
  } else {
    FAIL("GPU Unload Model - Tensorflow GPU");
    return false;
  }

  return true;
}

// Test the model can be loaded and unloaded on Inferentia.
bool test_model_util_infa_basic(const std::string& worker_name,
                               const std::string& model_name,
                               const struct Address& redis_addr,
                               std::unique_ptr<RedisMetadata>& rmd,
                               std::unique_ptr<S3Client>& s3c) {
  InfaModelManager manager(worker_name);
  InfaModelManager manager2(worker_name);
  std::string testcase = "";

  // Test load model.
  std::string src_url = bucket_prefix + infaas_bucket + "/" + model_name;
  auto res = manager.LoadModel(src_url, model_name, rmd, s3c);
  int8_t is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "Inferentia Load Model - " + model_name;
  if ((res == 0) && is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  // Load a replica
  res = manager2.LoadModel(src_url, model_name, rmd, s3c, "replica");
  is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "Inferentia Load Model 2 - " + model_name;
  if ((res == 0) && is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  // Unload one model, should unload the last one (replica)
  res = manager.UnloadModel(model_name, rmd);
  is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "Inferentia Unload Model - " + model_name;
  if ((res == 0) && is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }


  // Test unload model, the container name should match.
  res = manager.UnloadModel(model_name, rmd, model_name);
  is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "Inferentia Unload Model 2 - " + model_name;
  if ((res == 0) && !is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  return true;
}

bool test_translate_util_basic(const std::string& worker_name,
                               const std::string& model_name,
                               const struct Address& redis_addr,
                               std::unique_ptr<RedisMetadata>& rmd,
                               std::unique_ptr<S3Client>& s3c) {
  CpuModelManager manager(worker_name);
  std::string testcase = "";
  // Test load model.
  std::string src_url = bucket_prefix + infaas_bucket + "/" + model_name;
  auto res = manager.LoadModel(src_url, model_name, rmd, s3c);
  int8_t is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "Translate Load Model - " + model_name;
  if ((res == 0) && is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  res = manager.LoadModel(src_url, model_name, rmd, s3c);
  is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "CPU Load Model again - " + model_name;
  if ((res > 0) && is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  // Test unload a model.
  res = manager.UnloadModel(model_name, rmd, model_name, true);
  is_running = rmd->is_model_running(model_name, worker_name);
  testcase = "CPU Unload Model - " + model_name;
  if ((res == 0) && !is_running) {
    PASS(testcase.c_str());
  } else {
    FAIL(testcase.c_str());
    return false;
  }

  return true;
}
// Refer to:
// https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/c++/image_client.cc
// Format: FORMAT_NCHW, Scaletype: INCEPTION
void Preprocess(const cv::Mat& img, int img_type1, int img_type3,
                size_t img_channels, const cv::Size& img_size,
                std::vector<uint8_t>* input_data, bool preprocess=true) {
  // Image channels are in BGR order. Currently model configuration
  // data doesn't provide any information as to the expected channel
  // orderings (like RGB, BGR). We are going to assume that RGB is the
  // most likely ordering and so change the channels to that ordering.
  cv::Mat sample;
  if ((img.channels() == 3) && (img_channels == 1)) {
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  } else if ((img.channels() == 4) && (img_channels == 1)) {
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  } else if ((img.channels() == 3) && (img_channels == 3)) {
    cv::cvtColor(img, sample, CV_BGR2RGB);
  } else if ((img.channels() == 4) && (img_channels == 3)) {
    cv::cvtColor(img, sample, CV_BGRA2RGB);
  } else if ((img.channels() == 1) && (img_channels == 3)) {
    cv::cvtColor(img, sample, CV_GRAY2RGB);
  } else {
    std::cerr << "unexpected number of channels in input image or model"
              << std::endl;
    exit(1);
  }

  cv::Mat sample_resized;
  if (sample.size() != img_size) {
    cv::resize(sample, sample_resized, img_size);
  } else {
    sample_resized = sample;
  }

  cv::Mat sample_type;
  sample_resized.convertTo(sample_type,
                           (img_channels == 3) ? img_type3 : img_type1);

  cv::Mat sample_final;
  if (preprocess) {
    if (img_channels == 1) {
      sample_final = sample_type.mul(cv::Scalar(1 / 128.0));
      sample_final = sample_final - cv::Scalar(1.0);
    } else {
      sample_final = sample_type.mul(cv::Scalar(1 / 128.0, 1 / 128.0, 1 / 128.0));
      sample_final = sample_final - cv::Scalar(1.0, 1.0, 1.0);
    }
  } else {
    // Don't preprocess
    sample_final = sample_type;
  }

  // Allocate a buffer to hold all image elements.
  size_t img_byte_size = sample_final.total() * sample_final.elemSize();
  size_t pos = 0;
  input_data->resize(img_byte_size);

  // For CHW formats must split out each channel from the matrix and
  // order them as BBBB...GGGG...RRRR. To do this split the channels
  // of the image directly into 'input_data'. The BGR channels are
  // backed by the 'input_data' vector so that ends up with CHW
  // order of the data.
  std::vector<cv::Mat> input_bgr_channels;
  for (size_t i = 0; i < img_channels; ++i) {
    input_bgr_channels.emplace_back(img_size.height, img_size.width, img_type1,
                                    &((*input_data)[pos]));
    pos += input_bgr_channels.back().total() *
           input_bgr_channels.back().elemSize();
  }

  cv::split(sample_final, input_bgr_channels);

  if (pos != img_byte_size) {
    std::cerr << "unexpected total size of channels " << pos << ", expecting "
              << img_byte_size << std::endl;
    exit(1);
  }
}

void FileToInputData(const std::string& filename, size_t c, size_t h, size_t w,
                     int type1, int type3, std::vector<uint8_t>* input_data,
                     bool preprocess=true) {
  // Load image
  std::ifstream file(filename);
  std::vector<char> data;
  file >> std::noskipws;
  std::copy(std::istream_iterator<char>(file), std::istream_iterator<char>(),
            std::back_inserter(data));
  if (data.empty()) {
    std::cerr << "error: unable to read image file " << filename << std::endl;
    exit(1);
  }

  cv::Mat img = imdecode(cv::Mat(data), 1);
  if (img.empty()) {
    std::cerr << "error: unable to decode image " << filename << std::endl;
    exit(1);
  }
  // Pre-process the image to match input size expected by the model.
  Preprocess(img, type1, type3, c, cv::Size(w, h), input_data);
}

// Test query online is correct
bool test_query_online(QueryClient& invalid_client, QueryClient& query_client) {
  // Test invalid query client.
  QueryOnlineRequest request;
  request.add_raw_input("input");
  QueryOnlineResponse resp;
  auto output = resp.mutable_raw_output();

  auto inv_reply = invalid_client.QueryOnline(
      request.raw_input(), {"testmodel"}, "Qian", output, 1000, 71, 200, 1000);
  if (inv_reply.status() == InfaasRequestStatusEnum::SUCCESS) {
    std::string msg = "Query Online-invalid, should not success!";
    FAIL(msg.c_str());
    return false;
  }
  if (inv_reply.msg().find("RPC FAILURE") != std::string::npos) {
    PASS("Query Online-invalid");
  } else {
    std::string msg = "Query Online-invalid. Error msg: " + inv_reply.msg();
    FAIL(msg.c_str());
    return false;
  }

  // Test wrong input.
  // TODO: add more test cases once we finished.

  // Empty model.
  auto reply = query_client.QueryOnline(request.raw_input(), {}, "Qian", output,
                                        1000, 71, 200);
  if (reply.status() == InfaasRequestStatusEnum::SUCCESS) {
    std::string msg = "Query Online-no model, should not success!";
    FAIL(msg.c_str());
    return false;
  }
  if (reply.msg().find("INTERNAL FAILURE") != std::string::npos) {
    PASS("Query Online-no model");
  } else {
    std::string msg = "Query Online-no model. Error msg: " + reply.msg();
    FAIL(msg.c_str());
    return false;
  }

  // Test correct input.
  // TODO: this test may need to be modified once we finished the entire logic.
  // Now we only provide a fake input and check the fake output.
  reply = query_client.QueryOnline(request.raw_input(), {"testmodel"}, "Qian",
                                   output, 1000, 71, 200);
  if (reply.status() != InfaasRequestStatusEnum::SUCCESS) {
    std::string msg = "Query Online-testmodel, error msg: " + reply.msg();
    FAIL(msg.c_str());
    return false;
  }

  if (output->Get(0) == "SUCCESSFULQUERY") {
    PASS("Query Online-testmodel");
  } else {
    std::string msg = "Query Online-testmodel. Error msg: " + output->Get(0);
    FAIL(msg.c_str());
    return false;
  }

  return true;
}

// Test online image classification query
bool test_imgclass_online_query(const std::string& worker_name,
                                const struct Address& redis_addr,
                                const std::string& model_name,
                                std::unique_ptr<RedisMetadata>& rmd) {
  std::string testcase = "";
  bool preprocess = true;
  int batch_size = 2;
  if (model_name == test_model.model_name) {
    testcase = "GPU TensorRT - ImageClass Online Query";
  } else if (model_name == test_tfgpu_model.model_name) {
    testcase = "GPU TensorFlow - ImageClass Online Query";
  } else if (model_name == test_infa_model.model_name) {
    testcase = "Inferentia - ImageClass Online Query";
    preprocess = false;
    batch_size = 1;
  } else {
    testcase = "CPU - ImageClass Online Query";
  }

  grpc::ChannelArguments arguments;
  arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
  arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);
  infaas::internal::QueryClient query_client(grpc::CreateCustomChannel(
      query_exe_addr, grpc::InsecureChannelCredentials(), arguments));

  int slo = 30;
  std::string imgfile = "../data/mug_224.jpg";
  int8_t res = 0;

  // Prepare input
  std::vector<uint8_t> input_data;
  // Format: FORMAT_NCHWi, Scaletype: INCEPTION
  FileToInputData(imgfile, 3, 224, 224, /*type1=*/CV_32FC1, /*type3=*/CV_32FC3,
                  &input_data, preprocess);
  std::string input_str(input_data.begin(), input_data.end());
  QueryOnlineRequest request;
  for (int i = 0; i < batch_size; ++i) {
    request.add_raw_input(input_str);
  }

  QueryOnlineResponse resp;
  auto output = resp.mutable_raw_output();

  std::cout << "input_str size" << input_str.size() << std::endl;
  auto reply =
      query_client.QueryOnline(request.raw_input(), {model_name},
                               test_model.submitter, output, slo, 71, 200);

  if (reply.status() != InfaasRequestStatusEnum::SUCCESS) {
    std::string msg = testcase + ", error msg: " + reply.msg();
    FAIL(msg.c_str());
    return false;
  }

  std::string out_str = output->Get(0);
  std::cout << out_str.size() << std::endl;

  if (output->size() != batch_size) {
    FAIL(std::string(testcase + " - batch size not " + std::to_string(batch_size)).c_str());
    return false;
  }

  request.Clear();
  request.add_raw_input(input_str);
  for (int i = 0; i < 5; ++i) {
    reply =
        query_client.QueryOnline(request.raw_input(), {model_name},
                                 test_model.submitter, output, slo, 71, 200);
  }

  // Check the metadata is updated.
  double mod_qps = rmd->get_model_qps(worker_name, model_name);
  double mod_lat = rmd->get_model_avglat(worker_name, model_name);
  std::cout << "Model QPS: " << mod_qps << "; avg latency: " << mod_lat
            << std::endl;
  if (mod_qps < 0.0) {
    FAIL(std::string(testcase + " - qps less than 0").c_str());
    return false;
  }

  if (mod_lat < 0.0) {
    FAIL(std::string(testcase + " - avg latency less than 0").c_str());
    return false;
  }

  PASS(testcase.c_str());
  return true;
}

bool test_translate_online_query(const std::string& worker_name,
                                const struct Address& redis_addr,
                                const std::string& model_name,
                                std::unique_ptr<RedisMetadata>& rmd) {
  std::string testcase = "";
  int batch_size = 2;
  if (model_name == test_gnmt_cpu_model.model_name) {
    testcase = "CPU GNMT - Translate Online Query";
  } else if (model_name == test_gnmt_gpu_model.model_name) {
    testcase = "GPU GNMT - Granslate Online Query";
  }
  grpc::ChannelArguments arguments;
  arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
  arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);
  infaas::internal::QueryClient query_client(grpc::CreateCustomChannel(
      query_exe_addr, grpc::InsecureChannelCredentials(), arguments));

  int slo = 800;
  std::string input_str = "The quick brown fox jumps over the lazy dog";
  int8_t res = 0;

  // Prepare input
  QueryOnlineRequest request;
  for (int i = 0; i < batch_size; ++i) {
    request.add_raw_input(input_str);
  }

  QueryOnlineResponse resp;
  auto output = resp.mutable_raw_output();

  std::cout << "input_str size" << input_str.size() << std::endl;
  auto reply =
      query_client.QueryOnline(request.raw_input(), {model_name},
                               test_model.submitter, output, slo, 71, 200);

  if (reply.status() != InfaasRequestStatusEnum::SUCCESS) {
    std::cout << "RPC failed: " << reply.msg() << ". Try again." << std::endl;
    sleep(10);
    // Retry after 10 sec.
    reply =
        query_client.QueryOnline(request.raw_input(), {model_name},
                                 test_model.submitter, output, slo, 71, 200);
    if (reply.status() != InfaasRequestStatusEnum::SUCCESS) {
      std::string msg = testcase + ", error msg: " + reply.msg();
      FAIL(msg.c_str());
      return false;
    }
  }

  std::string out_str = output->Get(0);
  std::cout << "Translate result: " << out_str << std::endl;

  if (output->size() != batch_size) {
    FAIL(std::string(testcase + " - batch size not " + std::to_string(batch_size)).c_str());
    return false;
  }

  PASS(testcase.c_str());

  return true;
}


static bool send_and_wait_offline(QueryClient& query_client,
                                  std::unique_ptr<S3Client>& s3c,
                                  const std::string& output_url,
                                  const std::string& testcase) {
  auto reply = query_client.QueryOffline(infaas_bucket + "/offline_input",
                                         {test_cpu_model.model_name}, "Qian",
                                         infaas_bucket + "/" + output_url, 200);

  if ((reply.status() == InfaasRequestStatusEnum::SUCCESS) &&
      (reply.msg().find("Request accepted") != std::string::npos)) {
    // Check whether we have output in the bucket.
    int max_try = 10;
    int num_try = 1;
    std::vector<std::string> output_names;
    while (num_try < max_try) {
      std::cout << "try #" << num_try << std::endl;
      sleep(5);
      output_names.clear();
      if (list_s3_path(infaas_bucket, output_url, s3c, &output_names) != 0) {
        std::cerr << "list_s3_path failed!" << std::endl;
        break;
      }
      num_try++;
      std::cout << "Outnames size: " << output_names.size() << std::endl;
      if (output_names.size() == 4) { break; }
    }
    if (output_names.size() == 4) {
      PASS(testcase.c_str());
    } else {
      std::string errmsg =
          testcase + " Outnames size: " + std::to_string(output_names.size());
      FAIL(errmsg.c_str());
      return false;
    }
  } else {
    std::string errmsg = testcase + " Error code: " +
                         InfaasRequestStatusEnum_Name(reply.status()) +
                         ". Error msg: " + reply.msg();
    FAIL(errmsg.c_str());
    return false;
  }

  return true;
}

// Test query offline is correct
bool test_query_offline(QueryClient& invalid_client, QueryClient& query_client,
                        std::unique_ptr<S3Client>& s3c) {
  // Test invalid query client.
  auto inv_reply = invalid_client.QueryOffline("inputurl", {"testmodel"},
                                               "Qian", "outputurl", 200, 1000);
  if (inv_reply.status() == InfaasRequestStatusEnum::SUCCESS) {
    std::string msg = "Query Offline-invalid, should not success!";
    FAIL(msg.c_str());
    return false;
  }
  if (inv_reply.msg().find("RPC FAILURE") != std::string::npos) {
    PASS("Query Offline-invalid");
  } else {
    std::string msg = "Query Offline-invalid. Error msg: " + inv_reply.msg();
    FAIL(msg.c_str());
    return false;
  }

  // Test wrong input.
  // TODO: add more test cases once we finished.

  // Empty model
  auto reply =
      query_client.QueryOffline("inputurl", {}, "Qian", "outputurl", 200);

  if (reply.status() == InfaasRequestStatusEnum::SUCCESS) {
    std::string msg = "Query Offline-no model, should not success!";
    FAIL(msg.c_str());
    return false;
  }
  if (reply.msg().find("INTERNAL FAILURE") != std::string::npos) {
    PASS("Query Offline-no model");
  } else {
    std::string msg = "Query Offline-no model. Error msg: " + reply.msg();
    FAIL(msg.c_str());
    return false;
  }

  std::string output_url = "offline_output/testptmodel_qian/";
  bool rs = send_and_wait_offline(query_client, s3c, output_url,
                                  "Query Offline-testptmodel");
  if (!rs) { return rs; }

  // Try again.
  output_url = "offline_output/testptmodel_qian2/";
  rs = send_and_wait_offline(query_client, s3c, output_url,
                             "Query Offline-testptmodel 2");
  if (!rs) { return rs; }

  // Test the offline container directly
  // TODO: this is just a temporary test, remove later.
  /*
  infaas::internal::QueryClient container_client(grpc::CreateChannel(
      "localhost:9001", grpc::InsecureChannelCredentials()));

  reply = container_client.QueryOffline("testptmodel", {"testpt"}, "Qian",
      "testptmodel", 200);
  if (reply.status() != InfaasRequestStatusEnum::SUCCESS) {
    std::string errmsg = "Container returned error: " +
  InfaasRequestStatusEnum_Name(reply.status()) +                         ".
  Error msg: " + reply.msg(); FAIL(errmsg.c_str()); return false;
  }
  PASS("Query Offline - PyTorch Container");
  */
  return true;
}

// This one is deprecated, but keep here for future reference.
// Test we can query image classification model.
// bool test_imgclass_model_gpu_query(const std::string& worker_name,
//                                    const struct Address& redis_addr,
//                                    std::unique_ptr<RedisMetadata>& rmd,
//                                    std::unique_ptr<S3Client>& s3c) {
//   GpuModelManager manager(worker_name);
//
//   // Load the model.
//   std::string model_name = test_model.model_name;
//   std::string model_url =  bucket_prefix + infaas_bucket + "/" + model_name;
//   std::string imgfile = "../data/mug.jpg";
//   auto res = manager.LoadModel(model_url, model_name, rmd, s3c);
//   if (res != 0) {
//     FAIL("GPU ImageClass Query Util");
//     return false;
//   }
//   std::cout << "Success loading the model!" << std::endl;
//
//   // Prepare input
//   QueryOnlineRequest request;
//   std::vector<uint8_t> input_data;
//   // Format: FORMAT_NCHWi, Scaletype: INCEPTION
//   FileToInputData(imgfile, 3, 224, 224, /*type1=*/CV_32FC1,
//   /*type3=*/CV_32FC3,
//                   &input_data);
//   std::string input_str(input_data.begin(), input_data.end());
//   request.add_raw_input(input_str);
//
//   QueryOnlineResponse reply;
//   auto output = reply.mutable_raw_output();
//
//   struct timeval time1, time2;
//   gettimeofday(&time1, NULL);
//   res = manager.QueryImageClassModel(model_name, request.raw_input(),
//                                      output, 1);
//   gettimeofday(&time2, NULL);
//   printf("[query_test.cc] manager.QueryImageClassModel time: %.4lf ms.\n",
//          ts_to_ms(time1, time2));
//   if (res != 0) {
//     FAIL("GPU ImageClass Query Util");
//     return false;
//   }
//
//   std::string out_str = output->Get(0);
//   std::cout << out_str << std::endl;
//   if (out_str.find("COFFEE MUG") == std::string::npos) {
//     FAIL("GPU ImageClass Query Util - couldn't find COFFEE MUG.");
//     return false;
//   }
//   PASS("GPU ImageClass Query Util");
//   res = manager.UnloadModel(model_name, rmd);
//   return true;
// }

}  // namespace
}  // namespace internal
}  // namespace infaas

// IMPORTANT: need to launch query executor on the same machine port 50051 to
// run this test
int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: ./query_test <worker_name> <redis_ip> <redis_port> "
                 "[offline_only] [translate_test_only]"
              << std::endl;
    exit(1);
  }
  std::string worker_name = argv[1];
  const struct Address redis_addr = {argv[2], argv[3]};
  if (RedisMetadata::is_empty_address(redis_addr)) {
    std::cerr << "Invalid redis server address: " << argv[1] << ":" << argv[2]
              << std::endl;
    exit(1);
  }

  bool offline_only = false;
  bool translate_test_only = false;
  if (argc >= 5) {
    offline_only = (bool)atoi(argv[4]);
    if (argc >=6) {
      // Only use GPU for translate. No trtserver.
      translate_test_only = (bool)atoi(argv[5]);
    }
  }

  // First, register a model for test.
  // We now use the same model name for both variant and parent model.
  auto rmd = std::unique_ptr<RedisMetadata>(new RedisMetadata(redis_addr));
  Aws::SDKOptions s3_options;
  Aws::InitAPI(s3_options);
  Aws::Client::ClientConfiguration s3cfg;
  s3cfg.scheme = Aws::Http::Scheme::HTTPS;
  s3cfg.region = Aws::String("us-west-2");
  s3cfg.endpointOverride = Aws::String("s3.us-west-2.amazonaws.com");
  s3cfg.connectTimeoutMs = 1000 * 60 * 3;  // Connection timeout = 3min
  s3cfg.requestTimeoutMs = 1000 * 60 * 3;  // Request timeout = 3min.
  auto s3c = std::unique_ptr<S3Client>(new S3Client(s3cfg));

  int8_t rc = -1;
  rc = rmd->add_gparent_model(grandparent);
  if (rc) { FAIL("Add grandparent model."); }
  rc = rmd->add_gparent_model(grandparent_translate);
  if (rc) { FAIL("Add grandparent model for translation."); }

  rc = rmd->add_parent_model("resnet50");
  if (rc < 0) { FAIL("Add parent model resnet50."); }
  rc = rmd->add_parent_model("gnmt_ende4");
  if (rc < 0) { FAIL("Add parent model gnmt_ende4."); }


  rc = rmd->add_model(
      test_model.model_name, "resnet50", grandparent, test_model.comp_size,
      test_model.acc, test_model.dataset, test_model.submitter,
      test_model.framework, test_model.task, test_model.input_dim,
      test_model.batch, test_model.load_lat, test_model.inf_lat,
      test_model.peak_mem, test_model.slope, test_model.intercept);
  if (rc < 0) { FAIL("Add test model."); }

  rc = rmd->add_model(
      test_cpu_model.model_name, "resnet50", grandparent,
      test_cpu_model.comp_size, test_cpu_model.acc, test_cpu_model.dataset,
      test_cpu_model.submitter, test_cpu_model.framework, test_cpu_model.task,
      test_cpu_model.input_dim, test_cpu_model.batch, test_cpu_model.load_lat,
      test_cpu_model.inf_lat, test_cpu_model.peak_mem, test_cpu_model.slope,
      test_cpu_model.intercept);
  if (rc < 0) { FAIL("Add test cpu model."); }

  rc = rmd->add_model(test_tfgpu_model.model_name, "resnet50", grandparent,
                      test_tfgpu_model.comp_size, test_tfgpu_model.acc,
                      test_tfgpu_model.dataset, test_tfgpu_model.submitter,
                      test_tfgpu_model.framework, test_tfgpu_model.task,
                      test_tfgpu_model.input_dim, test_tfgpu_model.batch,
                      test_tfgpu_model.load_lat, test_tfgpu_model.inf_lat,
                      test_tfgpu_model.peak_mem, test_tfgpu_model.slope,
                      test_tfgpu_model.intercept);
  if (rc < 0) { FAIL("Add test tensorflow-GPU model."); }

  rc = rmd->add_model(test_tfcpu_model.model_name, "resnet50", grandparent,
                      test_tfcpu_model.comp_size, test_tfcpu_model.acc,
                      test_tfcpu_model.dataset, test_tfcpu_model.submitter,
                      test_tfcpu_model.framework, test_tfcpu_model.task,
                      test_tfcpu_model.input_dim, test_tfcpu_model.batch,
                      test_tfcpu_model.load_lat, test_tfcpu_model.inf_lat,
                      test_tfcpu_model.peak_mem, test_tfcpu_model.slope,
                      test_tfcpu_model.intercept);
  if (rc < 0) { FAIL("Add test tensorflow-CPU model."); }

  rc = rmd->add_model(test_infa_model.model_name, "resnet50", grandparent,
                      test_infa_model.comp_size, test_infa_model.acc,
                      test_infa_model.dataset, test_infa_model.submitter,
                      test_infa_model.framework, test_infa_model.task,
                      test_infa_model.input_dim, test_infa_model.batch,
                      test_infa_model.load_lat, test_infa_model.inf_lat,
                      test_infa_model.peak_mem, test_infa_model.slope,
                      test_infa_model.intercept);
  if (rc < 0) { FAIL("Add test Inferentia model."); }

  rc = rmd->add_model(test_gnmt_cpu_model.model_name, "gnmt_ende4", grandparent_translate,
                      test_gnmt_cpu_model.comp_size, test_gnmt_cpu_model.acc,
                      test_gnmt_cpu_model.dataset, test_gnmt_cpu_model.submitter,
                      test_gnmt_cpu_model.framework, test_gnmt_cpu_model.task,
                      test_gnmt_cpu_model.input_dim, test_gnmt_cpu_model.batch,
                      test_gnmt_cpu_model.load_lat, test_gnmt_cpu_model.inf_lat,
                      test_gnmt_cpu_model.peak_mem, test_gnmt_cpu_model.slope,
                      test_gnmt_cpu_model.intercept);
  if (rc < 0) { FAIL("Add test GNMT-cpu model."); }

  rc = rmd->add_model(test_gnmt_gpu_model.model_name, "gnmt_ende4", grandparent_translate,
                      test_gnmt_gpu_model.comp_size, test_gnmt_gpu_model.acc,
                      test_gnmt_gpu_model.dataset, test_gnmt_gpu_model.submitter,
                      test_gnmt_gpu_model.framework, test_gnmt_gpu_model.task,
                      test_gnmt_gpu_model.input_dim, test_gnmt_gpu_model.batch,
                      test_gnmt_gpu_model.load_lat, test_gnmt_gpu_model.inf_lat,
                      test_gnmt_gpu_model.peak_mem, test_gnmt_gpu_model.slope,
                      test_gnmt_gpu_model.intercept);
  if (rc < 0) { FAIL("Add test GNMT-gpu model."); }

  // Mock the invalid case.
  infaas::internal::QueryClient invalid_client(
      grpc::CreateChannel(invalid_addr, grpc::InsecureChannelCredentials()));

  // The real query executor.
  infaas::internal::QueryClient query_client(
      grpc::CreateChannel(query_exe_addr, grpc::InsecureChannelCredentials()));

  // test heartbeat
  std::cout << "=== HEARTBEAT TESTS ===" << std::endl;
  assert(infaas::internal::test_heartbeat(invalid_client, query_client));

  // Only test translate related.
  if (translate_test_only) {
    std::cout << "<<<<<<< ONLY TEST TRANSLATIONS <<<<<<<<" << std::endl;
    std::cout << "\n=== COMMON MODEL UTIL TESTS ===" << std::endl;
    assert(infaas::internal::test_translate_util_basic(
        worker_name, test_gnmt_cpu_model.model_name, redis_addr, rmd, s3c));

#ifdef INFAAS_GPU_WORKER
    assert(infaas::internal::test_translate_util_basic(
        worker_name, test_gnmt_gpu_model.model_name, redis_addr, rmd, s3c));
#else
    std::cout << "[SKIP] test_translate_util_basic for GPU model"
              << std::endl;
#endif

    std::cout << "\n=== QUERY ONLINE TESTS ===" << std::endl;
    // Test translation model.
    assert(infaas::internal::test_translate_online_query(
        worker_name, redis_addr, test_gnmt_cpu_model.model_name, rmd));
#ifdef INFAAS_GPU_WORKER
    assert(infaas::internal::test_translate_online_query(
        worker_name, redis_addr, test_gnmt_gpu_model.model_name, rmd));
#else
    std::cout << "[SKIP] test_translate_online_query for GPU model"
              << std::endl;
#endif
    // Test monitoring daemon.
    std::cout << "=== MONITORING TESTS ===" << std::endl;
    assert(infaas::internal::test_monitoring(worker_name, rmd));

    std::cout << "\nAll tests passed!" << std::endl;
    Aws::ShutdownAPI(s3_options);
    // Early return.
    return 0;
  }

  // test common model utils
  std::cout << "\n=== COMMON MODEL UTIL TESTS ===" << std::endl;
  assert(infaas::internal::test_model_util_cpu_basic(
      worker_name, test_cpu_model.model_name, redis_addr, rmd, s3c));
  assert(infaas::internal::test_model_util_cpu_basic(
      worker_name, test_tfcpu_model.model_name, redis_addr, rmd, s3c));
  assert(infaas::internal::test_translate_util_basic(
        worker_name, test_gnmt_cpu_model.model_name, redis_addr, rmd, s3c));

#ifdef INFAAS_NEURON_WORKER
  assert(infaas::internal::test_model_util_infa_basic(
      worker_name, test_infa_model.model_name, redis_addr, rmd, s3c));
#else
  std::cout << "[SKIP] test_model_util_infa_basic" << std::endl;
#endif

#ifdef INFAAS_GPU_WORKER
  assert(infaas::internal::test_model_util_gpu_basic(worker_name, redis_addr,
                                                     rmd, s3c));
  // assert(infaas::internal::test_imgclass_model_gpu_query(worker_name,
  // redis_addr, rmd, s3c));
  assert(infaas::internal::test_translate_util_basic(
        worker_name, test_gnmt_gpu_model.model_name, redis_addr, rmd, s3c));
#else
  std::cout << "[SKIP] test_model_util_gpu_basic" << std::endl;
  // std::cout << "[SKIP] test_imgclass_model_gpu_query" << std::endl;
#endif  // #ifdef INFAAS_GPU_WORKER

  if (!offline_only) {
    // test query online
    std::cout << "\n=== QUERY ONLINE TESTS ===" << std::endl;
    assert(infaas::internal::test_query_online(invalid_client, query_client));

#ifdef INFAAS_GPU_WORKER
    // Test GPU model.

    assert(infaas::internal::test_imgclass_online_query(
      worker_name, redis_addr, test_model.model_name, rmd));
    assert(infaas::internal::test_translate_online_query(
        worker_name, redis_addr, test_gnmt_gpu_model.model_name, rmd));

    assert(infaas::internal::test_imgclass_online_query(
      worker_name, redis_addr, test_tfgpu_model.model_name, rmd));
#else
    std::cout << "[SKIP] test_imageclass_online_query for GPU model"
              << std::endl;
#endif  // #ifdef INFAAS_GPU_WORKER

    // Test CPU model.
    assert(infaas::internal::test_translate_online_query(
        worker_name, redis_addr, test_gnmt_cpu_model.model_name, rmd));

    assert(infaas::internal::test_imgclass_online_query(
        worker_name, redis_addr, test_cpu_model.model_name, rmd));
    assert(infaas::internal::test_imgclass_online_query(
        worker_name, redis_addr, test_tfcpu_model.model_name, rmd));

#ifdef INFAAS_NEURON_WORKER
    // Test Inferentia model
    assert(infaas::internal::test_imgclass_online_query(
        worker_name, redis_addr, test_infa_model.model_name, rmd));
#else
    std::cout << "[SKIP] test_imageclass_online_query for Inferentia model"
              << std::endl;
#endif
  }

  // test query offline
  std::cout << "=== QUERY OFFLINE TESTS ===" << std::endl;
  assert(
      infaas::internal::test_query_offline(invalid_client, query_client, s3c));

  // Test monitoring daemon.
  std::cout << "=== MONITORING TESTS ===" << std::endl;
  assert(infaas::internal::test_monitoring(worker_name, rmd));

  std::cout << "\nAll tests passed!" << std::endl;
  Aws::ShutdownAPI(s3_options);
  return 0;
}
