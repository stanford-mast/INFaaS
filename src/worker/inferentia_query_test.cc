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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

// Refer to:
// https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/c++/image_client.cc
// Format: FORMAT_NCHW, Scaletype: INCEPTION
void Preprocess(const cv::Mat& img, int img_type1, int img_type3,
                size_t img_channels, const cv::Size& img_size,
                std::vector<uint8_t>* input_data) {
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
  if (img_channels == 1) {
    sample_final = sample_type.mul(cv::Scalar(1 / 128.0));
    sample_final = sample_final - cv::Scalar(1.0);
  } else {
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
                     int type1, int type3, std::vector<uint8_t>* input_data) {
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

bool test_online_simple(QueryClient& query_client, const int& img_scale) {
  // Prepare input
  std::string imgfile = "../../data/mug_224.jpg";
  std::vector<uint8_t> input_data;
  // Format: FORMAT_NCHWi, Scaletype: INCEPTION
  FileToInputData(imgfile, 3, img_scale, img_scale, /*type1=*/CV_32FC1, /*type3=*/CV_32FC3,
                  &input_data);
  std::string input_str(input_data.begin(), input_data.end());
  QueryOnlineRequest request;
  request.add_raw_input(input_str);
  std::cout << "input_str size" << input_str.size() << std::endl;

  QueryOnlineResponse resp;
  auto output = resp.mutable_raw_output();
  auto reply = query_client.QueryOnline(request.raw_input(), {"testmodel"},
    "Qian", output, 1000, 71, 200);
  if (reply.status() != InfaasRequestStatusEnum::SUCCESS) {
    std::string msg = "Query Online-testmodel, error msg: " + reply.msg();
    FAIL(msg.c_str());
    return false;
  }

  std::string out_str = output->Get(0);
  std::cout << "INFER OUTPUT: " << out_str.size() << std::endl;
  return true;
}

}  // namespace
}  // namespace internal
}  // namespace infaas

int main(int argc, char** argv) {
  // NOTE: need to launch Inferentia container/script before running this script.
  // The Inferentia container/script by default should run at port 9001
  std::string inf_addr = "localhost:9001";

  int img_scale = 224;
  if (argc > 1) {
    img_scale = atoi(argv[1]);
    if (argc > 2) {
      inf_addr = "localhost:" + std::string(argv[2]);
    }
  } else {
    std::cerr << "Usage: ./inferentia_query_test <img_scale (default 224)> <port (default 9001)>"
              << std::endl;
  }

  // Connect to the container.
  infaas::internal::QueryClient query_client(
    grpc::CreateChannel(inf_addr, grpc::InsecureChannelCredentials()));

  // Test heatbeat.
  std::cout << "=== HEARTBEAT TESTS ===" << std::endl;
  assert(infaas::internal::test_heartbeat(query_client));

  // Test online query.
  std::cout << "=== QUERY ONLINE TESTS ===" << std::endl;
  assert(infaas::internal::test_online_simple(query_client, img_scale));

  return 0;
}
