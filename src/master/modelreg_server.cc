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

#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <math.h>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/CopyObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/Object.h>

#include <grpcpp/grpcpp.h>
#include "include/constants.h"
#include "metadata-store/redis_metadata.h"
#include "modelreg.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

// Function to compute the linear regression parameters for batch prediction
void compute_linreg(const double* batch_sizes, const double* measured_inflat,
                    const int16_t num_items, double* slope, double* intercept) {
  if (num_items == 1) {
    *slope = measured_inflat[0];
    *intercept = 0.0;
    return;
  }

  double xsum = 0, x2sum = 0, ysum = 0, xysum = 0;
  for (int i = 0; i < num_items; ++i) {
    xsum = xsum + batch_sizes[i];
    ysum = ysum + measured_inflat[i];
    x2sum = x2sum + pow(batch_sizes[i], 2);
    xysum = xysum + batch_sizes[i] * measured_inflat[i];
  }

  // Now compute the slope and the y-intercept
  *slope =
      (num_items * xysum - xsum * ysum) / (num_items * x2sum - xsum * xsum);
  *intercept =
      (x2sum * ysum - xsum * xysum) / (num_items * x2sum - xsum * xsum);

  return;
}

namespace infaaspublic {
namespace infaasmodelreg {

// Logic and data behind the server's behavior.
class ModelRegServiceImpl final : public ModelReg::Service {
public:
  ModelRegServiceImpl(const struct Address& redis_addr)
      : redis_addr_(redis_addr) {
    rm_ = std::unique_ptr<RedisMetadata>(new RedisMetadata(redis_addr_));
  }

private:
  Status RegisterModel(ServerContext* context, const ModelRegRequest* request,
                       ModelRegResponse* reply) override {
    infaaspublic::RequestReply* rs = reply->mutable_status();
    int8_t rc = -1;

    // Define all model metadata inputs
    std::string grandparent_model, parent_model, variant_name, dataset,
        submitter, framework, task, url;
    double accuracy, load_latency, peak_memory;
    int16_t input_dim, batch_size;
    uint32_t comp_size;

    // Inference latency will be used for linear regression
    int16_t num_batch_items = 3;              // Changed based on max batch size
    double batch_values[] = {1.0, 4.0, 8.0};  // Hard-coded for now
    double inf_latency[num_batch_items];

    submitter = request->submitter();
    url = request->url();

    // Initialize AWS SDK, it is important to increase the timeout.
    Aws::SDKOptions options;
    Aws::InitAPI(options);
    Aws::Client::ClientConfiguration clientConfig;
    clientConfig.region = Aws::String(region.c_str());
    clientConfig.scheme = Aws::Http::Scheme::HTTPS;
    clientConfig.connectTimeoutMs = 30000;
    clientConfig.requestTimeoutMs = 600000;
    Aws::S3::S3Client s3_client(clientConfig);

    // If submitter == "tester", read config_url and set up from there
    // Otherwise, profile model from here, which is a TODO
    if (submitter == "tester") {
      Aws::S3::Model::GetObjectRequest config_download_request;
      config_download_request
          .WithBucket(Aws::String(infaas_config_bucket.c_str()))
          .WithKey(Aws::String(request->config_name().c_str()));
      auto get_object_outcome = s3_client.GetObject(config_download_request);
      if (get_object_outcome.IsSuccess()) {
        auto& retrieved_file = get_object_outcome.GetResult().GetBody();
        char file_data[255] = {0};
        while (retrieved_file.getline(file_data, 254)) {
          std::string next_input(file_data);
          std::stringstream ss(next_input);
          std::string metadata_item, value;
          ss >> metadata_item;
          ss >> value;
          if (metadata_item == "parname") {
            parent_model = value;
          } else if (metadata_item == "varname") {
            variant_name = value;
          } else if (metadata_item == "dataset") {
            dataset = value;
          } else if (metadata_item == "task") {
            task = value;
          } else if (metadata_item == "framework") {
            framework = value;
          } else if (metadata_item == "inputdim") {
            input_dim = std::stoi(value);
          } else if (metadata_item == "maxbatch") {
            batch_size = std::stoi(value);
          } else if (metadata_item == "loadlat") {
            load_latency = std::stod(value);
          } else if (metadata_item == "inflatb1") {
            inf_latency[0] = std::stod(value);
          } else if (metadata_item == "inflatb4") {
            inf_latency[1] = std::stod(value);
          } else if (metadata_item == "inflatb8") {
            inf_latency[2] = std::stod(value);
          } else if (metadata_item == "accuracy") {
            accuracy = std::stod(value);
          } else if (metadata_item == "compsize") {
            comp_size = std::stoi(value);
          } else if (metadata_item == "peakmemory") {
            peak_memory = std::stod(value);
          }
          memset(file_data, 0, sizeof file_data);
        }
        // Set grandparent model
        grandparent_model =
            task + "-" + dataset + "-" + std::to_string(input_dim);
      } else {
        rs->set_status(RequestReplyEnum::INVALID);
        rs->set_msg("Failed to GET configuration file");
        return Status::OK;
      }
    } else {
      grandparent_model = request->grandparent_model();
      parent_model = request->parent_model();
      variant_name = request->first_variant();
      dataset = request->dataset();
      framework = request->framework();
      task = request->task();
      accuracy = request->accuracy();
      batch_size = 128;     // Dummy value for now
      input_dim = 224;      // Dummy value for now
      load_latency = 1000;  // Dummy value for now
      inf_latency[0] = 10;  // Dummy value for now
      inf_latency[1] = 20;  // Dummy value for now
      inf_latency[2] = 30;  // Dummy value for now
      peak_memory = 1000;   // Dummy value for now
      comp_size = 1000;     // Dummy value for now
    }

    // Check if model exists, and return immediately if it already does.
    if (rm_->model_registered(variant_name)) {
      rs->set_status(RequestReplyEnum::INVALID);
      rs->set_msg("Model already exists");
      return Status::OK;
    }

    size_t ind = url.find("/");
    std::string sb = url.substr(0, ind);
    std::string so = url.substr(ind + 1, url.length() - ind);

    Aws::String source_bucket(sb.c_str(), sb.size());
    Aws::String source_object(so.c_str(), so.size());
    Aws::String destination_bucket(infaas_bucket.c_str(), infaas_bucket.size());
    Aws::String destination_object(variant_name.c_str(), variant_name.size());
    Aws::String complete_source = {url.c_str(), url.size()};
    Aws::String complete_destination =
        destination_bucket + "/" + destination_object + "/";

    // Copy model
    if (complete_source == complete_destination) {
      // Skip if same
      std::cout << "[LOG]: Skip copying because the source == destination"
                << std::endl;
    } else {
      Aws::S3::Model::ListObjectsV2Request list_request;
      list_request.SetBucket(source_bucket);
      list_request.SetPrefix(source_object);

      auto list_objects_outcome = s3_client.ListObjectsV2(list_request);
      if (list_objects_outcome.IsSuccess()) {
        Aws::Vector<Aws::S3::Model::Object> object_list =
            list_objects_outcome.GetResult().GetContents();
        for (auto const& i : object_list) {
          Aws::String source_bucket_object = source_bucket + "/" + i.GetKey();

          // Replace original bucket name with new one
          std::string next_object(i.GetKey().c_str(), i.GetKey().size());
          size_t index = next_object.find(so);
          next_object.replace(index, so.length() - 1, variant_name);

          Aws::S3::Model::CopyObjectRequest object_request;
          object_request.WithBucket(destination_bucket)
              .WithKey({next_object.c_str(), next_object.size()})
              .WithCopySource(source_bucket_object);

          auto copy_object_outcome = s3_client.CopyObject(object_request);
          if (!copy_object_outcome.IsSuccess()) {
            auto error = copy_object_outcome.GetError();
            std::cout << "ERROR: " << error.GetExceptionName() << ": "
                      << error.GetMessage() << std::endl;

            rs->set_status(RequestReplyEnum::INVALID);
            rs->set_msg("Error when trying to copy model");
            return Status::OK;
          }
        }
      } else {
        auto error = list_objects_outcome.GetError();
        std::cout << "ERROR: " << error.GetExceptionName() << ": "
                  << error.GetMessage() << std::endl;

        rs->set_status(RequestReplyEnum::INVALID);
        rs->set_msg("Bucket not found!");
        return Status::OK;
      }

      // This is a better way to do it, but for some reason it is not finding the bucket object
      //// even though it is successfully copied.
      /*
      Aws::S3::Model::HeadObjectRequest check_request;
      check_request.WithBucket(destination_bucket).WithKey(destination_object);
      auto check_outcome = s3_client.HeadObject(check_request);
      if (check_outcome.IsSuccess()) {
        std::cout << "[LOG]: Successfully copied" << std::endl;
      } else {
        std::cout << "[LOG]: Model path probably wrong..." << std::endl;
        auto error = check_outcome.GetError();
        std::cout << "ERROR: " << error.GetExceptionName() << ": "
                  << error.GetMessage() << std::endl;

        rs->set_status(RequestReplyEnum::INVALID);
        rs->set_msg(
            "Model not copied successfully. Model path likely invalid!");
        return Status::OK;
      }
      */

      Aws::S3::Model::ListObjectsV2Request check_request;
      check_request.WithBucket(destination_bucket);
      auto check_outcome = s3_client.ListObjectsV2(check_request);

      bool check_passed = false;
      if (check_outcome.IsSuccess()) {
        Aws::Vector<Aws::S3::Model::Object> check_object_list =
          check_outcome.GetResult().GetContents();

        for (auto const &s3_object : check_object_list) {
          std::string next_object(s3_object.GetKey().c_str(),
                                  s3_object.GetKey().size());
          std::cout << "* " << next_object << std::endl;
          size_t ind = next_object.find("/");
          std::string dest_check = next_object.substr(0, ind);
          // variant_name is the string version of destination_object above
          if (dest_check == variant_name) {
            check_passed = true;
            break;
          }
        }
      }

      if (check_passed) { 
        std::cout << "[LOG]: Successfully copied" << std::endl;
      } else {
        std::cout << "[LOG]: Model path probably wrong..." << std::endl;
        auto error = check_outcome.GetError();
        std::cout << "ERROR: " << error.GetExceptionName() << ": "
                  << error.GetMessage() << std::endl;

        rs->set_status(RequestReplyEnum::INVALID);
        rs->set_msg(
            "Model not copied successfully. Model path likely invalid!");
        return Status::OK;
      }
    }

    // Create grandparent model if needed
    if (!rm_->gparent_model_registered(grandparent_model)) {
      rc = rm_->add_gparent_model(grandparent_model);
      if (rc) {
        rs->set_status(RequestReplyEnum::INVALID);
        rs->set_msg("Failed to add grandparent model");
        return Status::CANCELLED;
      }
    }

    // Create parent model if needed
    if (!rm_->parent_model_registered(parent_model)) {
      rc = rm_->add_parent_model(parent_model);
      if (rc) {
        rs->set_status(RequestReplyEnum::INVALID);
        rs->set_msg("Failed to add parent model");
        return Status::CANCELLED;
      }
    }

    // Compute slope and intercept for inference latencies
    double slope, intercept;

    // Set num_batch_items based on max batch size
    if (batch_size == 1) {
      num_batch_items = 1;
    } else if (batch_size < 8) {
      num_batch_items = 2;
    }

    compute_linreg(batch_values, inf_latency, num_batch_items, &slope,
                   &intercept);

    // Store model metadata
    rc = rm_->add_model(variant_name, parent_model, grandparent_model,
                        comp_size, accuracy, dataset, submitter, framework,
                        task, input_dim, batch_size, load_latency,
                        inf_latency[0], peak_memory, slope, intercept);

    if (rc) {
      rs->set_status(RequestReplyEnum::INVALID);
      rs->set_msg("Failed to register model");
      return Status::OK;
    }

    std::cout << "[LOG]: Successfully registered " << variant_name;
    std::cout << " (Parent: " << parent_model << ")" << std::endl;
    std::cout << "[LOG]: Calculated slope: " << slope
              << "; Calculated intercept: ";
    std::cout << intercept << std::endl;
    std::cout << "================================================="
              << std::endl;

    rs->set_status(RequestReplyEnum::SUCCESS);
    rs->set_msg("Successfully registered model");
    return Status::OK;
  }

  Status Heartbeat(ServerContext* context, const HeartbeatRequest* request,
                   HeartbeatResponse* reply) override {
    infaaspublic::RequestReply* rs = reply->mutable_status();
    if (request->status().status() != RequestReplyEnum::SUCCESS) {
      rs->set_status(RequestReplyEnum::INVALID);
      rs->set_msg("Received invalid heartbeat");
      return Status::OK;
    }

    rs->set_status(RequestReplyEnum::SUCCESS);
    rs->set_msg("Successfully received heartbeat");
    return Status::OK;
  }

  // Internal variables
  const struct Address redis_addr_;
  std::unique_ptr<RedisMetadata> rm_;
};

}  // namespace infaasmodelreg
}  // namespace infaaspublic

void RunModelRegServer(const struct Address& redis_addr) {
  std::string server_address("0.0.0.0:50053");
  infaaspublic::infaasmodelreg::ModelRegServiceImpl service(redis_addr);

  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: ./modelreg_server <redis_ip> <redis_port>"
              << std::endl;
    return 1;
  }

  const struct Address redis_addr = {argv[1], argv[2]};
  if (RedisMetadata::is_empty_address(redis_addr)) {
    std::cerr << "Invalid redis server address: "
              << RedisMetadata::Address_to_str(redis_addr) << std::endl;
    return 1;
  }

  RunModelRegServer(redis_addr);

  return 0;
}
