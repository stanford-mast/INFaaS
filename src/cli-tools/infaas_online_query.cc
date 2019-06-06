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

#include <getopt.h>
#include <sys/stat.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

#include "include/constants.h"
#include "master/modelreg_client.h"
#include "master/queryfe_client.h"

static const bool debug = false;
static const int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;

void usage() {
  std::cout << "Usage: infaas_online_query -i input -d input_dim [-a "
               "model_architecture]";
  std::cout << " [-v model_variant] [-t task] [-D dataset] [-A accuracy] [-l "
               "latency(ms)]";
  std::cout << std::endl;
  std::cout
      << "input: Path to raw image. Pass each input with an individual flag"
      << std::endl;
  std::cout
      << "In addition to the required inputs, one of the following input ";
  std::cout << "combinations is required:" << std::endl;
  std::cout << "\tmodel_variant" << std::endl;
  std::cout << "\tmodel_architecture (input_dim must be valid), latency"
            << std::endl;
  std::cout << "\ttask, dataset, accuracy, latency" << std::endl;
}

// Refer to:
// https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/c++/image_client.cc
// Format: FORMAT_NCHW, Scaletype: INCEPTION
static void Preprocess(const cv::Mat& img, int img_type1, int img_type3,
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
    sample_final = sample_type.mul(cv::Scalar(1 / 128.0, 1 / 128.0, 1 / 128.0));
    sample_final = sample_final - cv::Scalar(1.0, 1.0, 1.0);
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
    std::cerr << "Unexpected total size of channels " << pos << ", expecting "
              << img_byte_size << std::endl;
    exit(1);
  }
}

static void FileToInputData(const std::string& filename, size_t c, size_t h,
                            size_t w, int type1, int type3,
                            std::vector<uint8_t>* input_data) {
  // Load image
  std::ifstream file(filename);
  std::vector<char> data;
  file >> std::noskipws;
  std::copy(std::istream_iterator<char>(file), std::istream_iterator<char>(),
            std::back_inserter(data));
  if (data.empty()) {
    std::cerr << "error: Unable to read image file " << filename << std::endl;
    exit(1);
  }

  cv::Mat img = imdecode(cv::Mat(data), 1);
  if (img.empty()) {
    std::cerr << "error: Unable to decode image " << filename << std::endl;
    exit(1);
  }
  // Pre-process the image to match input size expected by the model.
  Preprocess(img, type1, type3, c, cv::Size(w, h), input_data);
}

int main(int argc, char** argv) {
  std::vector<std::string> input_paths;
  size_t input_dim = 0;

  std::string model_architecture;
  std::string model_variant;
  std::string task;
  std::string dataset;
  double accuracy = 0.0;
  double latency = 0.0;

  struct option long_options[] = {
      {"inputs", required_argument, nullptr, 'i'},
      {"input_dim", required_argument, nullptr, 'd'},
      {"model_architecture", required_argument, nullptr, 'a'},
      {"model_variant", required_argument, nullptr, 'v'},
      {"task", required_argument, nullptr, 't'},
      {"dataset", required_argument, nullptr, 'D'},
      {"accuracy", required_argument, nullptr, 'A'},
      {"latency", required_argument, nullptr, 'l'},
      {nullptr, 0, nullptr, 0},
  };

  while (true) {
    const int opt =
        getopt_long(argc, argv, "i:d:a:v:t:D:A:l:", long_options, NULL);

    if (opt == -1) { break; }

    switch (opt) {
      case 'i':
        input_paths.push_back(std::string(optarg));
        break;

      case 'd':
        input_dim = std::stoi(optarg);
        break;

      case 'a':
        model_architecture = std::string(optarg);
        break;

      case 'v':
        model_variant = std::string(optarg);
        break;

      case 't':
        task = std::string(optarg);
        break;

      case 'D':
        dataset = std::string(optarg);
        break;

      case 'A':
        accuracy = std::stod(optarg);
        break;

      case 'l':
        latency = std::stod(optarg);
        break;

      default:
        throw std::runtime_error("Invalid option");
    }
  }

  // Check for valid inputs
  if (!input_dim || input_paths.empty()) {
    std::cout << "Must specify both input_dim (-d) and input (-i)" << std::endl
              << std::endl;
    usage();
    return 1;
  } else if (model_variant.empty() &&
             (model_architecture.empty() || !latency) &&
             (task.empty() || dataset.empty() || !accuracy || !latency)) {
    std::cout << "Invalid input combination" << std::endl << std::endl;
    usage();
    return 1;
  }

  // Set up online query
  grpc::ChannelArguments arguments;
  arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
  arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);
  infaaspublic::infaasqueryfe::QueryFEClient queryfe(grpc::CreateCustomChannel(
      "localhost:50052", grpc::InsecureChannelCredentials(), arguments));

  // Prepare inputs
  std::cout << "Preparing inputs..." << std::endl;
  struct stat buffer;
  std::vector<uint8_t> img_data;
  std::vector<std::string> input_vector;
  for (std::string inp : input_paths) {
    // Check if file exists
    if (stat(inp.c_str(), &buffer)) {
      std::cerr << inp << " is not a valid input path" << std::endl;
      return 1;
    }

    FileToInputData(inp, 3, input_dim, input_dim, CV_32FC1, CV_32FC3,
                    &img_data);
    std::string next_input(img_data.begin(), img_data.end());
    input_vector.push_back(next_input);
  }
  std::cout << "Inputs ready for INFaaS" << std::endl;

  std::vector<std::string> query_reply;

  // By this point, we know the proper input configurations have been passed
  // Model-variant provided
  if (!model_variant.empty()) {
    if (debug) { std::cout << "Model Variant" << std::endl; }
    query_reply = queryfe.QueryOnline(input_vector, "", "", model_variant, "",
                                      0.0, 0.0, 0.0);
  } else if (!model_architecture.empty()) {  // Model architecture provided
    if (debug) { std::cout << "Model Architecture" << std::endl; }
    query_reply = queryfe.QueryOnline(input_vector, "", model_architecture, "",
                                      "", latency, 0.0, 0.0);
  } else {  // Use-case provided
    if (debug) { std::cout << "Use Case" << std::endl; }
    std::string use_case(task + "-" + dataset + "-" +
                         std::to_string(input_dim));
    query_reply = queryfe.QueryOnline(input_vector, use_case, "", "", "",
                                      latency, accuracy, 0.0);
  }

  if (query_reply.empty()) { std::cerr << "Failed query" << std::endl; }

  std::cout << query_reply.size()
            << " reply/replies, reply[0] size: " << query_reply[0].size();
  std::cout << std::endl;

  if (query_reply[0].size() % sizeof(float)) { return 0; }

  int cnt = 1;
  float f;
  for (std::string qr : query_reply) {
    std::cout << "Prediction " << cnt << ": " << std::endl;
    for (int i = 0; i < qr.size() / sizeof(f); i += sizeof(f)) {
      uchar b[] = {qr[i], qr[i + 1], qr[i + 2], qr[i + 3]};
      memcpy(&f, &b, sizeof(f));
      std::cout << f << " ";
    }
    std::cout << std::endl << std::endl;

    cnt++;
  }

  return 0;
}
