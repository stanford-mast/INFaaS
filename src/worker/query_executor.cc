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

// This process is running on a worker machine to receive query requests from
// INFaaS master.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <grpcpp/grpcpp.h>

#include "autoscaler.h"
#include "common_model_util.h"
#include "include/constants.h"
#include "internal/query.grpc.pb.h"
#include "metadata-store/redis_metadata.h"

using Aws::S3::S3Client;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

const std::string query_exe_addr = "0.0.0.0:50051";
const std::string test_model = "testmodel";
const std::string infaas_aws_region = "us-west-2";
const std::string infaas_s3_endpoint = "s3.us-west-2.amazonaws.com";

static const int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;
// TODO: we only allow 1 thread processing offline requests. May need multiple
// threads in the future.
static const int OFFLINE_THREAD_POOL_SIZE = 1;
static const int AUTOSCALER_THREAD_POOL_SIZE = 1;

namespace infaas {
namespace internal {
namespace {
// For CPU utilization
static unsigned long long lastTotalUser, lastTotalUserLow, lastTotalSys,
    lastTotalIdle;

// From answer: https://stackoverflow.com/a/1911863
void initCPUutil() {
  FILE* file = fopen("/proc/stat", "r");
  fscanf(file, "cpu %llu %llu %llu %llu", &lastTotalUser, &lastTotalUserLow,
         &lastTotalSys, &lastTotalIdle);
  fclose(file);
}

// Get average CPU Utilization over the past duration.
// https://stackoverflow.com/a/1911863
double getCPUutil() {
  double percent;
  FILE* file;
  unsigned long long totalUser, totalUserLow, totalSys, totalIdle, total;

  file = fopen("/proc/stat", "r");
  fscanf(file, "cpu %llu %llu %llu %llu", &totalUser, &totalUserLow, &totalSys,
         &totalIdle);
  fclose(file);

  if (totalUser < lastTotalUser || totalUserLow < lastTotalUserLow ||
      totalSys < lastTotalSys || totalIdle < lastTotalIdle) {
    std::cerr << "CPU util overflow detection. Just skip this value"
              << std::endl;
    percent = -1.0;
  } else {
    total = (totalUser - lastTotalUser) + (totalUserLow - lastTotalUserLow) +
            (totalSys - lastTotalSys);
    percent = total;
    total += (totalIdle - lastTotalIdle);
    percent /= total;
    percent *= 100;
  }

  lastTotalUser = totalUser;
  lastTotalUserLow = totalUserLow;
  lastTotalSys = totalSys;
  lastTotalIdle = totalIdle;

  return percent;
}

}  // namespace

// Implementation of the query service.
class QueryServiceImpl final : public Query::Service {
public:
  QueryServiceImpl(std::string worker_name, struct Address redis_addr,
                   infaas::internal::AutoscalerType autoscaler_type)
      : worker_name_(worker_name), redis_addr_(redis_addr) {
    monitoring_run_ = true;
    redis_metadata_ =
        std::unique_ptr<RedisMetadata>(new RedisMetadata(redis_addr_));
    // For communicating with S3.
    Aws::InitAPI(s3_options_);
    Aws::Client::ClientConfiguration s3cfg;
    s3cfg.scheme = Aws::Http::Scheme::HTTPS;
    s3cfg.region = Aws::String(infaas_aws_region.c_str());
    s3cfg.endpointOverride = Aws::String(infaas_s3_endpoint.c_str());
    s3cfg.connectTimeoutMs = 1000 * 60 * 3;  // Connection timeout = 3min
    s3cfg.requestTimeoutMs = 1000 * 60 * 3;  // Request timeout = 3min.
    s3_client_ = std::unique_ptr<S3Client>(new S3Client(s3cfg));
    qpsMonitorThread_ = new std::thread(&QueryServiceImpl::qpsMonitor, this);
    resourceMonitorThread_ =
        new std::thread(&QueryServiceImpl::resourceMonitor, this);
    for (int i = 0; i < OFFLINE_THREAD_POOL_SIZE; ++i) {
      offlineProcessPool_.push_back(
          new std::thread(&QueryServiceImpl::offlineProccess, this));
    }

    for (int i = 0; i < AUTOSCALER_THREAD_POOL_SIZE; ++i) {
#ifdef INFAAS_GPU_WORKER
      autoscalerPool_.push_back(
          new std::thread(&Autoscaler::GpuAutoscalerDaemon, worker_name_,
                          std::ref(redis_metadata_), std::ref(s3_client_)));
#endif  // #ifdef INFAAS_GPU_WORKER
      autoscalerPool_.push_back(
          new std::thread(&Autoscaler::CpuAutoscalerDaemon, worker_name_,
                          std::ref(redis_metadata_), std::ref(s3_client_)));
      autoscalerPool_.push_back(new std::thread(&Autoscaler::AutoscalerArbiter,
                                                worker_name_, autoscaler_type,
                                                std::ref(redis_metadata_)));
    }
  }

  ~QueryServiceImpl() {
    monitoring_run_ = false;
    qpsMonitorThread_->join();
    resourceMonitorThread_->join();
    for (int i = 0; i < OFFLINE_THREAD_POOL_SIZE; ++i) {
      offlineProcessPool_[i]->join();
    }
    for (int i = 0; i < autoscalerPool_.size(); ++i) {
      autoscalerPool_[i]->join();
    }
    Aws::ShutdownAPI(s3_options_);
  }

private:
  // Monitor QPS (request arrival rate) for each model instance.
  void qpsMonitor();

  // Monitor Resource utilization (CPU, GPU, DRAM) on the server.
  void resourceMonitor();

  // Process Offline requests in the queue.
  void offlineProccess();

  Status QueryOnline(ServerContext* context, const QueryOnlineRequest* request,
                     QueryOnlineResponse* reply) override;

  Status QueryOffline(ServerContext* context,
                      const QueryOfflineRequest* request,
                      QueryOfflineResponse* reply) override;

  Status Heartbeat(ServerContext* context, const HeartbeatRequest* request,
                   HeartbeatResponse* reply) override;

  // Internal variables
  std::string worker_name_;
  struct Address redis_addr_;
  std::unique_ptr<RedisMetadata> redis_metadata_;
  Aws::SDKOptions s3_options_;
  std::unique_ptr<S3Client> s3_client_;
  // For Offline queries
  // The mutex protects concurrent access to offline_reqs_.
  std::mutex offline_mutex_;
  std::deque<QueryOfflineRequest> offline_reqs_;
  std::thread* qpsMonitorThread_;
  std::thread* resourceMonitorThread_;
  // Thread pool for offline requests processing
  std::vector<std::thread*> offlineProcessPool_;
  // Thread pool for autoscaling daemon
  std::vector<std::thread*> autoscalerPool_;
  bool monitoring_run_;
  // Total number of requests received by each model.
  static std::map<std::string, std::atomic<uint64_t>> model_total_reqs_;
  // Sum of batch sizes per model.
  static std::map<std::string, std::atomic<uint64_t>> model_total_batch_;
  // Sum of latencies per model, in usec.
  static std::map<std::string, std::atomic<uint64_t>> model_total_lat_;
  // Total number of requests completed.
  static std::map<std::string, std::atomic<uint64_t>> model_total_comp_;
  // Sum of slo-latency per model.
  static std::map<std::string, std::atomic<uint64_t>> model_total_slo_;
};

std::map<std::string, std::atomic<uint64_t>>
    QueryServiceImpl::model_total_reqs_;
std::map<std::string, std::atomic<uint64_t>>
    QueryServiceImpl::model_total_batch_;
std::map<std::string, std::atomic<uint64_t>> QueryServiceImpl::model_total_lat_;
std::map<std::string, std::atomic<uint64_t>>
    QueryServiceImpl::model_total_comp_;
std::map<std::string, std::atomic<uint64_t>> QueryServiceImpl::model_total_slo_;

Status QueryServiceImpl::Heartbeat(ServerContext* context,
                                   const HeartbeatRequest* request,
                                   HeartbeatResponse* reply) {
  if (request->status().status() != InfaasRequestStatusEnum::SUCCESS) {
    std::cout << "Heartbeat request invalid status: "
              << request->status().status() << std::endl;
    reply->mutable_status()->set_status(InfaasRequestStatusEnum::INVALID);
    return Status::CANCELLED;
  }
  std::cout << "Received heartbeat" << std::endl;
  reply->mutable_status()->set_status(InfaasRequestStatusEnum::SUCCESS);
  return Status::OK;
}

Status QueryServiceImpl::QueryOnline(ServerContext* context,
                                     const QueryOnlineRequest* request,
                                     QueryOnlineResponse* reply) {
  uint64_t time1, time2;
  time1 = get_curr_timestamp();

  auto models = request->model();
  auto slo = request->slo();

  InfaasRequestStatus* request_status = reply->mutable_status();

  // Check model pool not empty.
  if (models.size() == 0) {
    request_status->set_status(InfaasRequestStatusEnum::INVALID);
    request_status->set_msg("No model provided!");
    return Status(grpc::StatusCode::INVALID_ARGUMENT, "No model provided!");
  }

  // The main part of scheduling
  // 1. Check how many models provided. If only one:
  // 1.1. Check whether a model instance exists.
  // 1.2. If doesn't exist, create a container (for CPU) or add the model
  //      to TRTIS models/ (for GPU)
  // 1.3. If model instance exists, direct the request to that container.
  // 2. If multiple models provided, pick up one model based on whether the
  // model is running and latency/accuracy trade-off.

  if (models.size() == 1) {
    // For tests:
    if (models[0] == test_model) {
      reply->add_raw_output("SUCCESSFULQUERY");
      request_status->set_status(InfaasRequestStatusEnum::SUCCESS);

      time2 = get_curr_timestamp();
      printf("worker [query_executor.cc] QueryOnline total time: %.4lf ms.\n",
             get_duration_ms(time1, time2));
      fflush(stdout);
      return Status::OK;
    }

    // Choose targeting hardware (GPU, CPU, etc).
    auto hw = ChooseHardware(models[0], redis_metadata_);
    model_total_slo_[models[0]].fetch_add(slo.latencyinusec());
    if (hw == "GPU") {
      model_total_reqs_[models[0]].fetch_add(1);
      model_total_batch_[models[0]].fetch_add(request->raw_input().size());

      GpuModelManager manager(worker_name_);
      uint64_t time3, time4;
      time3 = get_curr_timestamp();
      // printf("[query_executor.cc] manager QueryModelOnline start time:
      // %lu.\n",
      //       time3);

      int8_t res = manager.QueryModelOnline(models[0], request, reply,
                                            redis_metadata_, s3_client_);
      time4 = get_curr_timestamp();
      // printf("[query_executor.cc] manager QueryModelOnline end time: %lu.\n",
      //       time4);
      model_total_comp_[models[0]].fetch_add(1);
      model_total_lat_[models[0]].fetch_add(time4 - time3);

      printf(
          "[query_executor.cc] manager QueryModelOnline total time: %.4lf "
          "ms.\n",
          get_duration_ms(time3, time4));

      if (res < 0) {
        // TODO: probably try to run on other hardware or retry instead of
        // failing.
        std::cerr << "Failed to query, res = " << int(res) << std::endl;
        request_status->set_status(InfaasRequestStatusEnum::UNAVAILABLE);
        request_status->set_msg("Failed to serve online query on GPU!");
        return Status(grpc::StatusCode::UNAVAILABLE, "Failed to serve!");
      }
      time2 = get_curr_timestamp();
      printf("worker [query_executor.cc] QueryOnline total time: %.4lf ms.\n",
             get_duration_ms(time1, time2));
      fflush(stdout);
      request_status->set_status(InfaasRequestStatusEnum::SUCCESS);
      return Status::OK;
    } else if (hw == "CPU") {
      model_total_reqs_[models[0]].fetch_add(1);
      model_total_batch_[models[0]].fetch_add(request->raw_input().size());

      CpuModelManager manager(worker_name_);
      uint64_t time3, time4;
      time3 = get_curr_timestamp();
      // printf("[query_executor.cc] manager QueryModelOnline start time:
      // %lu.\n",
      //       time3);

      int8_t res = manager.QueryModelOnline(models[0], request, reply,
                                            redis_metadata_, s3_client_);
      time4 = get_curr_timestamp();
      // printf("[query_executor.cc] manager QueryModelOnline end time: %lu.\n",
      //       time4);
      model_total_comp_[models[0]].fetch_add(1);
      model_total_lat_[models[0]].fetch_add(time4 - time3);

      printf(
          "[query_executor.cc] manager QueryModelOnline total time: %.4lf "
          "ms.\n",
          get_duration_ms(time3, time4));

      if (res < 0) {
        // TODO: probably try to run on other hardware or retry instead of
        // failing.
        std::cerr << "Failed to query, res = " << int(res) << std::endl;
        request_status->set_status(InfaasRequestStatusEnum::UNAVAILABLE);
        request_status->set_msg("Failed to serve online query on GPU!");
        return Status(grpc::StatusCode::UNAVAILABLE, "Failed to serve!");
      }
      time2 = get_curr_timestamp();
      // printf("worker [query_executor.cc] QueryOnline total time: %.4lf
      // ms.\n",
      //       get_duration_ms(time1, time2));
      // fflush(stdout);
      request_status = reply->mutable_status();
      request_status->set_status(InfaasRequestStatusEnum::SUCCESS);
      return Status::OK;
    } else {
      std::cerr << "No support hardware: " << hw << std::endl;
      request_status->set_status(InfaasRequestStatusEnum::INVALID);
      request_status->set_msg("No support hardware");
      return Status::OK;
    }
  } else {
    std::cerr << "[NOT IMPLEMENTED] multiple model variants: " << models.size()
              << std::endl;
    request_status->set_status(InfaasRequestStatusEnum::INVALID);
    request_status->set_msg(
        "[NOT IMPLEMENTED] Don't support more than one model variants!");
    return Status::OK;
  }
  request_status->set_status(InfaasRequestStatusEnum::SUCCESS);
  return Status::OK;
}

Status QueryServiceImpl::QueryOffline(ServerContext* context,
                                      const QueryOfflineRequest* request,
                                      QueryOfflineResponse* reply) {
  std::string input_url = request->input_url();
  auto models = request->model();
  std::string output_url = request->output_url();

  InfaasRequestStatus* request_status = reply->mutable_status();

  // Check model pool not empty.
  if (models.size() == 0) {
    request_status->set_status(InfaasRequestStatusEnum::INVALID);
    request_status->set_msg("No model provided!");
    return Status(grpc::StatusCode::INVALID_ARGUMENT, "No model provided!");
  }

  if (models.size() == 1) {
    // For tests: directly return
    if (models[0] == test_model) {
      request_status->set_status(InfaasRequestStatusEnum::SUCCESS);
      request_status->set_msg(input_url + " : " + output_url);
      return Status::OK;
    }

    // Push the request into the queue of offline jobs.
    {
      std::lock_guard<std::mutex> lock(offline_mutex_);
      offline_reqs_.push_back(*request);
    }
    request_status->set_status(InfaasRequestStatusEnum::SUCCESS);
    request_status->set_msg("Request accepted");
    return Status::OK;
  } else {
    request_status->set_status(InfaasRequestStatusEnum::INVALID);
    request_status->set_msg("Should only provide one model!");
    return Status(grpc::StatusCode::INVALID_ARGUMENT,
                  "[NOT IMPLEMENTED] Only one model is allowed.");
  }

  request_status->set_status(InfaasRequestStatusEnum::INVALID);
  request_status->set_msg("Something wrong: " + input_url + " : " + output_url);
  return Status::OK;
}

void QueryServiceImpl::offlineProccess() {
  // Set nice value = 10 to be a lower priority.
  int curr_nice = nice(10);
  std::cout << "Set offlineProcess thread nice = " << curr_nice << std::endl;

  uint64_t time1, time2;
  QueryOfflineRequest request;
  std::cout << "Offline Process thread is ready " << std::endl;
  int sleep_interval = 1000;  // Sleep 1 sec.
  bool has_job = false;
  while (monitoring_run_) {
    time1 = get_curr_timestamp();
    // Get one offline job from the front.
    {
      std::lock_guard<std::mutex> lock(offline_mutex_);
      if (offline_reqs_.size() > 0) {
        request = offline_reqs_.front();
        offline_reqs_.pop_front();
        has_job = true;
      } else {
        has_job = false;
      }
    }
    if (!has_job) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
      continue;
    }

    std::string model_name = request.model()[0];
    auto hw = ChooseHardware(model_name, redis_metadata_);
    if (hw == "CPU") {
      CpuModelManager manager(worker_name_);
      int8_t res = manager.QueryModelOffline(model_name, request,
                                             redis_metadata_, s3_client_);
      if (res < 0) {
        // TODO: need a better way to handle error.
        std::cerr << "Failed to serve offline query for model " << model_name
                  << std::endl;
      }
    } else {
      std::cerr << "Offline doesn't support hardware: " << hw << std::endl;
    }

    time2 = get_curr_timestamp();
    double interval = get_duration_ms(time1, time2);
    std::cout << "Process current offline query: " << interval << " ms."
              << std::endl;
  }

  std::cout << "Offline Processing thread shut off!" << std::endl;
}

void QueryServiceImpl::qpsMonitor() {
  // Log to the file "INFaaS/worker/qps_daemon.log"
  std::ofstream logfile;
  logfile.open(infaas_log_dir + "/worker/qps_daemon.log");

  // For experiment.
  std::ofstream explog;
  explog.open(infaas_log_dir + "/worker/infaas_exp_qps.log");
  explog << "TimeStampInUsec, ModvarName, CurrQPS, NumReplicas" << std::endl;

  // Set nice value = 10 to be a lower priority.
  int curr_nice = nice(10);
  logfile << "Set qpsMonitor thread nice = " << curr_nice << std::endl;

  uint64_t curr_time, prev_time;
  prev_time = get_curr_timestamp();
  int sleep_interval = 1000;  // Sleep 1 sec.
  std::map<std::string, uint64_t>
      model_last_reqs_;  // the qps we've seen last time.
  std::map<std::string, uint64_t>
      model_last_batch_;  // the batch size we've seen last time.
  std::map<std::string, uint64_t>
      model_last_comp_;  // the completed requests we've seen last time.
  std::map<std::string, uint64_t>
      model_last_lat_;  // the sum of latencies we've seen last time.
  std::map<std::string, uint64_t>
      model_last_slo_;  // the sum of slo-latencies we've seen last time.
  while (monitoring_run_) {
    curr_time = get_curr_timestamp();
    logfile << "Logging QPS at timestamp: " << std::fixed << curr_time
            << std::endl;
    double interval = get_duration_ms(prev_time, curr_time);
    // Skip the very first interval
    if (interval >= sleep_interval) {
      // Consider per parent model.
      std::vector<std::string> running_parents =
          redis_metadata_->get_parent_models_on_executor(worker_name_);
      bool has_blacklisted = false;
      for (auto& parent_name : running_parents) {
        logfile << "Parent: " << parent_name << std::endl;
        std::vector<std::string> running_modvars =
            redis_metadata_->get_parents_variants_on_executor(parent_name,
                                                              worker_name_);
        bool parent_scaledown =
            false;  // Whether we should force to scale down to CPU.
        bool has_cpu = false;
        for (auto& model_name : running_modvars) {
          // Calculate qps
          uint64_t curr_cnt = model_total_reqs_[model_name].load();
          uint64_t curr_batch_cnt = model_total_batch_[model_name].load();
          uint64_t curr_comp_cnt = model_total_comp_[model_name].load();
          uint64_t curr_lat_cnt = model_total_lat_[model_name].load();
          uint64_t curr_slo_cnt = model_total_slo_[model_name].load();
          auto hw = ChooseHardware(model_name, redis_metadata_);
          size_t num_replicas = 1;
          if (hw == "CPU") {
            num_replicas = CpuModelManager::numReplicas(model_name);
            // logfile << "CPU model " << model_name << " num_replicas: "
            //        << num_replicas << std::endl;
            has_cpu = true;
          } else if (hw == "GPU") {
            num_replicas = GpuModelManager::numReplicas(model_name);
            // logfile << "GPU model " << model_name << " num_replicas: "
            //        << num_replicas << std::endl;
          } else {
            // logfile << "Unsupported hardware model " << model_name <<
            // std::endl;
            continue;
          }
          if (num_replicas < 1) {
            logfile << "Model " << model_name << " has 0 replicas."
                    << std::endl;
            continue;
          }
          uint64_t last_cnt = model_last_reqs_[model_name];
          uint64_t last_batch_cnt = model_last_batch_[model_name];
          uint64_t last_comp_cnt = model_last_comp_[model_name];
          uint64_t last_lat_cnt = model_last_lat_[model_name];
          uint64_t last_slo_cnt = model_last_slo_[model_name];
          double curr_qps = (curr_cnt - last_cnt) /
                            ((interval / 1000.0) * (double)num_replicas);
          int curr_avg_batch = Autoscaler::getAvgBatch(model_name);
          double curr_avg_slo = 0;
          // No need to update batch size if there is no requests.
          if (curr_cnt > last_cnt) {
            curr_avg_batch = (int)std::ceil((curr_batch_cnt - last_batch_cnt) /
                                            (curr_cnt - last_cnt));
            curr_avg_slo =
                (curr_slo_cnt - last_slo_cnt) / (double)(curr_cnt - last_cnt);
          }
          double curr_avg_lat = std::max(
              0.0, redis_metadata_->get_model_avglat(worker_name_, model_name));
          if (curr_comp_cnt > last_comp_cnt) {
            curr_avg_lat = get_duration_ms(last_lat_cnt, curr_lat_cnt) /
                           (double)(curr_comp_cnt - last_comp_cnt);
          } else {
            // If there is no request completed within the last second,
            // gradually reduce the latency.
            curr_avg_lat = curr_avg_lat / 1.5;
          }
          model_last_reqs_[model_name] = curr_cnt;
          model_last_batch_[model_name] = curr_batch_cnt;
          model_last_comp_[model_name] = curr_comp_cnt;
          model_last_lat_[model_name] = curr_lat_cnt;
          model_last_slo_[model_name] = curr_slo_cnt;
          Autoscaler::setAvgBatch(model_name, curr_avg_batch);
          logfile << "[Interval = " << interval << " ] ";
          logfile << "Model: " << model_name << " ; total count: " << curr_cnt
                  << " ; current QPS: " << curr_qps
                  << "; current batch size: " << curr_avg_batch
                  << "; completed reqs: " << curr_comp_cnt
                  << "; avg latency: " << curr_avg_lat << "msec; curr_avg_slo "
                  << curr_avg_slo << std::endl;
          // Log for experiment
          explog << curr_time << ", " << model_name << ", "
                 << curr_qps * (double)num_replicas << ", " << num_replicas
                 << std::endl;
          auto rs = redis_metadata_->update_model_qps(worker_name_, model_name,
                                                      curr_qps);
          if (rs < 0) {
            logfile << "[qpsMonitor]Failed to update qps for model: "
                    << model_name << ". Status: " << int(rs) << std::endl;
          }
          // TODO: we may not need to update model avglat since the master
          // doesn't need it anymore. But let's leave it here right now.
          rs = redis_metadata_->update_model_avglat(worker_name_, model_name,
                                                    curr_avg_lat);
          if (rs < 0) {
            logfile << "[qpsMonitor]Failed to update avglat for model: "
                    << model_name << ". Status: " << int(rs) << std::endl;
          }

          // If the current model latency is 3x longer than the recorded one,
          // and the qps is 1.5x higher than the theoretical qps (avoid
          // oscillation when the model first loaded), then add to the
          // blacklist. If it became lower than 1.5x of the recorded latency,
          // unset the blacklist. The inf_lat is for the current average batch
          // size. int max_batch =

          double slope =
              std::stod(redis_metadata_->get_model_info(model_name, "slope"));
          double intercept = std::stod(
              redis_metadata_->get_model_info(model_name, "intercept"));
          double inf_lat = slope * curr_avg_batch + intercept;
          // if ((max_batch >= MAX_ONLINE_BATCH) || (hw == "CPU")) {
          // Blacklist logic is used for both CPU and GPU models.
          // Heuristics for blacklist.
          double blist_lat_heuristic = 1.5;
          double blist_qps_heuristic = 0.5;
          // Heuristic should be different for GPU.
          if (hw == "GPU") {
            blist_lat_heuristic = 5;
            blist_qps_heuristic = 0.03;  // When colocated, it can easily get
                                         // interfered at low load.
          }
          if (((curr_avg_lat > inf_lat * blist_lat_heuristic) &&
               (curr_qps > blist_qps_heuristic * 1000.0 / inf_lat)) ||
              (curr_cnt - curr_comp_cnt >
               blist_qps_heuristic * 1000.0 / inf_lat)) {
            rs = redis_metadata_->set_model_avglat_blacklist(worker_name_,
                                                             model_name);
            if (rs < 0) {
              logfile << "[qpsMonitor] Failed to set blacklist for model: "
                      << model_name << std::endl;
            }
            has_blacklisted = true;
            logfile << "Blacklisted model: " << model_name << std::endl;
          } else if (((curr_cnt - curr_comp_cnt) < 1000.0 / inf_lat) &&
                     curr_avg_lat < inf_lat) {
            // The first term makes sure requests will not queue up.
            rs = redis_metadata_->unset_model_avglat_blacklist(worker_name_,
                                                               model_name);
            if (rs < 0) {
              logfile << "[qpsMonitor] Failed to unset blacklist for model: "
                      << model_name << std::endl;
            }
            logfile << "Unset blacklist model: " << model_name << std::endl;
          }
          // }
          if (hw == "GPU") {
            // Check whether we need to set parent scaledown: slo is much lower
            // than the inf_latency and the qps is low, and there is no CPU
            // model available.
            // TODO: 0.03 is a heuristic. This logic is wrong because it cannot
            // make sure there is no CPU model.
            if ((curr_avg_slo > 10 * inf_lat) &&
                (curr_qps < 0.03 * 1000.0 / inf_lat)) {
              parent_scaledown = true;
            }
          }
        }
        // Make sure there is no CPU models. Otherwise, it is possible that we
        // ping-pong between CPU and GPU models. (CPU blacklisted -> GPU force
        // to choose CPU).
        if (parent_scaledown && !has_cpu) {
          auto rs =
              redis_metadata_->set_parent_scaledown(worker_name_, parent_name);
          if (rs < 0) {
            logfile << "[qpsMonitor] Failed to set scaledown for parent model: "
                    << parent_name << std::endl;
          } else {
            logfile << "Set scaledown for parent model: " << parent_name
                    << std::endl;
          }
        }
        logfile << "\n" << std::endl;
      }
      // Set blacklisted to true/false after testing all parent models.
      // Cannot run offline if even there is one model got blacklisted.
      CommonModelUtil::SetBlacklisted(has_blacklisted);
    }
    prev_time = curr_time;
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
  }
  logfile << "qpsMonitor shut down!" << std::endl;
  logfile.close();
}

void QueryServiceImpl::resourceMonitor() {
  // Log to the file "INFaaS/worker/resource_daemon.log"
  std::ofstream logfile;
  logfile.open(infaas_log_dir + "/worker/resource_daemon.log");

  // Set nice value = 10 to be a lower priority.
  int curr_nice = nice(10);
  logfile << "Set resourceMonitor thread nice = " << curr_nice << std::endl;

  uint64_t curr_time, prev_time;
  prev_time = get_curr_timestamp();
  // TODO: the sleep interval should not be too short or too long.
  int sleep_interval = 2500;  // Sleep 2.5 sec.
  initCPUutil();
  while (monitoring_run_) {
    curr_time = get_curr_timestamp();
    logfile << "Logging Resource Util at timestamp: " << std::fixed << curr_time
            << std::endl;
    double interval = get_duration_ms(prev_time, curr_time);
    // Skip the very first interval
    if (interval >= sleep_interval) {
      // Get CPU utilization
      double cpu_util = getCPUutil();
      logfile << "[Interval = " << interval << " ] ";
      logfile << "Worker: " << worker_name_
              << " ; current CPU util: " << cpu_util << std::endl;
      int8_t rs = -1;
      // Only update to metadata if it's a reasonable point
      if (cpu_util > 0.0) {
        rs = redis_metadata_->update_cpu_util(worker_name_, cpu_util);
        if (rs < 0) {
          logfile << "[resourceMonitor] Failed to update cpu util for worker: "
                  << worker_name_ << ". Status: " << int(rs) << std::endl;
        }
      }
      logfile << "////////////////////////////////////////////////"
              << std::endl;
    }
    prev_time = curr_time;
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
  }
  logfile << "resourceMonitor shut down!" << std::endl;
  logfile.close();
}

}  // namespace internal
}  // namespace infaas

void RunExecutor(const std::string& worker_name, struct Address redis_addr,
                 infaas::internal::AutoscalerType autoscaler_type) {
  std::string server_address(query_exe_addr);
  infaas::internal::QueryServiceImpl service(worker_name, redis_addr,
                                             autoscaler_type);

  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);

  // Set max message size.
  builder.SetMaxMessageSize(MAX_GRPC_MESSAGE_SIZE);

  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: ./query_executor <worker_name> <redis_ip> "
                 "<redis_port> [<autoscaler_type>]"
              << "autoscaler type: 0=NONE, 1=STATIC, 2=INDIVIDUAL, 3=INFaaS"
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
  infaas::internal::AutoscalerType autoscaler_type =
      infaas::internal::AUTOSCALE_NONE;
  if (argc > 4) {
    int atype = std::stoi(argv[4]);
    switch (atype) {
      case 0:
        autoscaler_type = infaas::internal::AUTOSCALE_NONE;
        break;
      case 1:
        autoscaler_type = infaas::internal::AUTOSCALE_STATIC;
        break;
      case 2:
        autoscaler_type = infaas::internal::AUTOSCALE_INDIVIDUAL;
        break;
      case 3:
        autoscaler_type = infaas::internal::AUTOSCALE_INFAAS;
        break;
      default:
        std::cerr << "Invalid autoscaler type: " << atype << std::endl;
        exit(1);
    }
  }
  std::cout << "Autoscaler type: " << autoscaler_type << std::endl;
  // Start the main executor deamon.
  RunExecutor(worker_name, redis_addr, autoscaler_type);

  return 0;
}
