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

#include <grpcpp/grpcpp.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <string>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>

#include "autoscaler.h"
#include "common_model_util.h"
#include "include/constants.h"

using Aws::S3::S3Client;

static const std::string local_trt_model_dir = "/tmp/trtmodels/";
static const std::string trt_pbtxt_file = "config.pbtxt";
// At least SLACK_SIZE replicas should be kept loaded.
static const int CPU_SLACK_SIZE = 2;
static const int GPU_SLACK_SIZE = 1;
static int GPU_MAX_REPLICAS = 1;
static const int CPU_MAX_REPLICAS = 2;
// We can scale down if we get 10 continuous scale down requests.
static int GPU_SCALE_DOWN_DELAY = 20;
static const int CPU_SCALE_DOWN_DELAY = 10;
// Need to change this to a dynamic value.
static const double total_gpu_memory = 17179869184;

static const double loadHeuristic = 0.0002;
static const double cpuHueristic = 0.05;
static const double memorySlack = 1024.0;  // At least 1 GB of free memory.
static const int TOP_K_FASTEST =
    10;  // We should consider top-10 fastest models.

// For Inferentia
// TODO: we just assume each inferentia worker has 4 neuron cores. Maybe make
// it configurable.
static const int WORKER_NEURON_CORES = 4;
static int INFA_SLACK_SIZE = 2;
static int INFA_MAX_REPLICAS = 2;
static const int INFA_SCALE_DOWN_DELAY = 15;

static const int NLP_SCALE_DOWN_DELAY = 100;

namespace infaas {
namespace internal {
namespace {

// Get total physical memory on this machine
int64_t getTotalMemory() {
  struct sysinfo memInfo;
  sysinfo(&memInfo);
  int64_t totalMem = memInfo.totalram;
  totalMem *= memInfo.mem_unit;
  return totalMem;
}

// Get free physical memory
int64_t getFreeMemory() {
  struct sysinfo memInfo;
  sysinfo(&memInfo);
  int64_t freeMem = memInfo.freeram;
  freeMem *= memInfo.mem_unit;
  return freeMem;
}

// Static scaler: always keep a static number (SLACK_SIZE) of replicas for each
// model variant.
// This simulate what most of serving systems are doing: a fixed number of
// replicas.
void StaticScaler(const std::string& worker_name,
                  std::unique_ptr<RedisMetadata>& rmd, std::ofstream& logfile) {
  std::vector<std::string> running_modvars =
      rmd->get_variants_on_executor(worker_name);
  for (auto& modvar : running_modvars) {
    uint64_t curr_time = get_curr_timestamp();
    logfile << "[ " << curr_time << " ] Checking " << modvar << std::endl;
    auto hw = ChooseHardware(modvar, rmd);
    size_t num_replicas = 0;
    int count = 0;
    int slack_thresh = 0;
    if (hw == "GPU") {
      num_replicas = GpuModelManager::numReplicas(modvar);
      slack_thresh = GPU_SLACK_SIZE;
    } else if (hw == "CPU") {
      num_replicas = CpuModelManager::numReplicas(modvar);
      slack_thresh = CPU_SLACK_SIZE;
    } else if (hw == "INFA") {
      num_replicas = InfaModelManager::numReplicas(modvar);
      slack_thresh = INFA_SLACK_SIZE;
    } else {
      continue;
    }
    // Don't generate scaling request if the model is currently being loaded
    // (num_replicas == 0).
    if ((num_replicas < slack_thresh) && (num_replicas > 0)) {
      count = slack_thresh - num_replicas;
    }

    if (count) {
      logfile << "Generating scale request for model " << modvar
              << ", count = " << count << std::endl;
      int8_t res;
      if (hw == "GPU") {
        res = Autoscaler::setScaleRequestGpu(modvar, count);
      } else if (hw == "CPU") {
        res = Autoscaler::setScaleRequestCpu(modvar, count);
      } else if (hw == "INFA") {
        res = Autoscaler::setScaleRequestInfa(modvar, count);
      }
      if (res < 0) {
        logfile << "[StaticScaler] failed to set scaling request: " << modvar
                << ", count " << count << std::endl;
      }
    }
    logfile << "\n" << std::endl;
  }
}

// Check individual model variant. Return 0 on success; return -1 if it needs
// to scale but exceeds the resource limits, or encountered some errors.
//
// batch: average batch size for the variant. weighted_delta_qps: w_reqs -
// w_curr. scale_count: how many replicas we need to scale, negative number
// means scaling down.
int8_t checkModvar(const std::string& worker_name, const std::string& modvar,
                   std::unique_ptr<RedisMetadata>& rmd, std::ofstream& logfile,
                   int* avg_batch, double* weighted_delta_qps,
                   std::string* hardware, double* mod_load_lat,
                   std::string* framework, int* scale_count) {
  *framework = rmd->get_model_info(modvar, "framework");
  auto hw = ChooseHardware(modvar, rmd, *framework);
  *hardware = hw;
  size_t num_replicas = 0;
  int count = 0;
  *scale_count = 0;
  double mod_qps = rmd->get_model_qps(worker_name, modvar);
  double load_lat = std::stod(rmd->get_model_info(modvar, "load_latency"));
  *mod_load_lat = load_lat;
  double slope = std::stod(rmd->get_model_info(modvar, "slope"));
  double intercept = std::stod(rmd->get_model_info(modvar, "intercept"));
  // NOTE: this is the batch size we need to compute the w_curr, not the actual
  // batch size!
  int batch = 0;
  // The actual average batch size of requests.
  int actual_batch = Autoscaler::getAvgBatch(modvar);
  double peak_mem = std::stod(rmd->get_model_info(modvar, "peak_memory"));
  double free_cpu_mem = (double)(getFreeMemory());
  double free_gpu_mem =
      (100.0 - rmd->get_gpu_util(worker_name)) / 100.0 * total_gpu_memory;
  double memory_limit = (hw == "GPU") ? free_gpu_mem : free_cpu_mem;
  int used_neurons = InfaModelManager::numUsedCores();
  int free_neurons = WORKER_NEURON_CORES - used_neurons;
  if (hw == "GPU") {
    num_replicas = GpuModelManager::numReplicas(modvar);
    // If it is a GPU model, we should use the maximum batch size a model can
    // support. In this way, the fastest model we select will always have a
    // higher batch size than the currently loaded GPU model. Thus, we can
    // prevent downgrading to a lower batch model.
    batch = std::min(std::stoi(rmd->get_model_info(modvar, "max_batch")),
                     MAX_ONLINE_BATCH);
  } else if (hw == "CPU") {
    num_replicas = CpuModelManager::numReplicas(modvar);
    batch = actual_batch;
  } else if (hw == "INFA") {
    num_replicas = InfaModelManager::numReplicas(modvar);
    batch = actual_batch;
  } else {
    logfile << "Unrecognized hardware: " << hw << std::endl;
    return -1;
  }
  *avg_batch = batch;
  double inf_lat = slope * batch + intercept;
  logfile << "Actual batch: " << actual_batch << "; max_batch: " << batch
          << std::endl;
  // Don't generate scaling request if the model is currently being loaded
  if (num_replicas > 0) {
    double single_throughput = 1000.0 / inf_lat * batch;
    double weighted_qps = mod_qps * actual_batch;  // Use the actual batch!
    double weighted_curr = num_replicas * single_throughput;
    double wdelta_qps =
        weighted_qps - weighted_curr + loadHeuristic * weighted_curr * load_lat;
    // Prevent CPU from scaling too fast.
    if (hw == "CPU") {
      wdelta_qps = weighted_qps - weighted_curr * (1 - cpuHueristic);
    }
    *weighted_delta_qps = wdelta_qps;
    // Compute scale up & down
    // Scale up
    int delta_plus = (int)std::ceil(wdelta_qps / single_throughput);
    // Scale down
    double down_thresh = (num_replicas - 1) * single_throughput;
    logfile << "w_reqs: " << weighted_qps << ", w_curr: " << weighted_curr
            << ", load_lat: " << load_lat << " => +delta: " << delta_plus
            << "; down_thresh: " << down_thresh << std::endl;
    if (delta_plus > 0) {
      count = delta_plus;
      *scale_count = count;
      // Check resource usage
      if (hw == "INFA") {
        if (used_neurons + count > WORKER_NEURON_CORES) {
          logfile << "Exceed Neuron cores limit" << std::endl;
          return -1;
        }
      } else if (delta_plus * peak_mem > std::max(memory_limit - memorySlack, 0.0)) {
        logfile << "Exceed resource limit" << std::endl;
        return -1;
      } else if ((hw == "GPU") &&
                 (num_replicas + delta_plus > GPU_MAX_REPLICAS)) {
        logfile << "Exceed GPU maximum replicas: " << delta_plus << std::endl;
        return -1;
      }
    } else {
      if (weighted_qps <= std::max(0.0, down_thresh)) {
        count = -1;
      } else {
        // Don't do anything.
        count = 0;
      }
    }
  }

  *scale_count = count;
  return 0;
}

// Scale per model variant.
// This simulate the behavior of SageMaker and probably Clipper and other
// "state-of-the-art" systems. They only consider replicating the same
// container/instance and no migration to better hardware/model variants.
void IndividualScaler(const std::string& worker_name,
                      std::unique_ptr<RedisMetadata>& rmd,
                      std::ofstream& logfile) {
  std::vector<std::string> running_modvars =
      rmd->get_variants_on_executor(worker_name);
  for (auto& modvar : running_modvars) {
    uint64_t curr_time = get_curr_timestamp();
    int8_t res;
    int batch, count;
    double weighted_delta_qps;
    double load_lat;
    std::string hw, framework;
    logfile << "[ " << curr_time << " ] Checking " << modvar << std::endl;
    res = checkModvar(worker_name, modvar, rmd, logfile, &batch,
                      &weighted_delta_qps, &hw, &load_lat, &framework, &count);
    // Ignore the error, just scale anyway.
    if (count) {
      logfile << "Generating scale request for model " << modvar
              << ", count = " << count << std::endl;
      if (hw == "GPU") {
        res = Autoscaler::setScaleRequestGpu(modvar, count);
      } else if (hw == "CPU") {
        res = Autoscaler::setScaleRequestCpu(modvar, count);
      } else if (hw == "INFA") {
        res = Autoscaler::setScaleRequestInfa(modvar, count);
      }
      if (res < 0) {
        logfile << "[IndividualScaler] failed to set scaling request: "
                << modvar << ", count " << count << std::endl;
      }
    }
    logfile << "\n" << std::endl;
  }
}

struct ModvarScaleInfo {
  std::string modvar_name;
  int count;                    // how many to scale
  double load_lat;              // load latency
  std::string hw;               // hardware
  std::string down_modvar;  // For trt scaling down.
};

// Check how many we need to scale for the fastest model.
// Return 0 means we can scale this model. Return -1 means there will be more
// than 1 replica of GPU model or out of memory
int8_t checkFastest(const std::string& worker_name,
                    const std::string& fastest_var,
                    std::unique_ptr<RedisMetadata>& rmd, std::ofstream& logfile,
                    int max_batch, double sum_wdelta_qps,
                    std::string down_var, std::string* fastest_hw,
                    double* fastest_loadlat, int* fastest_count) {
  auto hw = ChooseHardware(fastest_var, rmd);
  *fastest_hw = hw;
  size_t num_replicas = 0;
  int count = 0;
  *fastest_count = 0;
  int8_t is_running = rmd->is_model_running(fastest_var, worker_name);
  double load_lat = std::stod(rmd->get_model_info(fastest_var, "load_latency"));
  *fastest_loadlat = load_lat;
  double slope = std::stod(rmd->get_model_info(fastest_var, "slope"));
  double intercept = std::stod(rmd->get_model_info(fastest_var, "intercept"));
  int actual_batch = Autoscaler::getAvgBatch(fastest_var);
  int batch =
      (hw == "GPU")
          ? std::min(std::stoi(rmd->get_model_info(fastest_var, "max_batch")),
                     MAX_ONLINE_BATCH)
          : actual_batch;
  logfile << "Actual batch: " << actual_batch << "; max_batch: " << batch
          << std::endl;
  double inf_lat = slope * batch + intercept;
  double peak_mem = std::stod(rmd->get_model_info(fastest_var, "peak_memory"));
  double free_cpu_mem = (double)(getFreeMemory());
  double free_gpu_mem = 0;
#ifdef INFAAS_GPU_WORKER
  free_gpu_mem = (100.0 - rmd->get_gpu_util(worker_name)) / 100.0 * total_gpu_memory;
  logfile << free_gpu_mem << std::endl;
#endif
  double memory_limit = (hw == "GPU") ? free_gpu_mem : free_cpu_mem;
  double mod_qps = 0.0;
  if (is_running) {
    mod_qps = rmd->get_model_qps(worker_name, fastest_var);
    if (hw == "GPU") {
      num_replicas = GpuModelManager::numReplicas(fastest_var);
    } else if (hw == "CPU") {
      num_replicas = CpuModelManager::numReplicas(fastest_var);
    } else if (hw == "INFA") {
      num_replicas = InfaModelManager::numReplicas(fastest_var);
    } else {
      logfile << "Unrecognized hardware: " << hw << std::endl;
      return -1;
    }
  }

  // Compute our formulas.
  double single_throughput = 1000.0 / inf_lat * batch;
  // Weighted qps needs to consider all other sum_weighted_delta_qps.
  double weighted_qps = mod_qps * actual_batch + sum_wdelta_qps;
  double weighted_curr = num_replicas * single_throughput;
  int delta_plus = 0;
  // Scale up formula
  double adjust_weighted =
      weighted_qps - weighted_curr + loadHeuristic * load_lat;
  if (weighted_qps > 0.0) {
    delta_plus = (int)std::ceil(adjust_weighted / single_throughput);
  }
  // Scale down, we need to consider the downgrade model if we have one.
  double down_thresh = ((double)num_replicas - 1) * single_throughput;
  if (!down_var.empty()) {
    auto down_hw = ChooseHardware(down_var, rmd);
    int down_batch = 1;
    // We will use batch-1 for CPU models.
    if (down_hw != "CPU") {
      down_batch =
          std::min(std::stoi(rmd->get_model_info(down_var, "max_batch")),
                   MAX_ONLINE_BATCH);
    }
    double down_slope = std::stod(rmd->get_model_info(down_var, "slope"));
    double down_intercept =
        std::stod(rmd->get_model_info(down_var, "intercept"));
    double down_inf_lat = down_slope * down_batch + down_intercept;
    double down_single_throughput = 1000.0 / down_inf_lat * down_batch;
    // Assume we will load exactly one downgrade model.
    down_thresh = down_single_throughput;
  }
  logfile << "adjusted_w_reqs: " << adjust_weighted
          << "; w_reqs: " << weighted_qps << ", w_curr: " << weighted_curr
          << "; single throughput: " << single_throughput
          << ", load_lat: " << load_lat << " => +delta: " << delta_plus
          << "; down_thresh: " << down_thresh << std::endl;
  if (delta_plus > 0) {
    count = delta_plus;
    *fastest_count = count;
    // Check resource usage
    if (delta_plus * peak_mem > std::max(memory_limit - memorySlack, 0.0)) {
      *fastest_count = 0;
      logfile << "Exceed resource limit" << std::endl;
      return -1;
    } else if ((hw == "GPU") &&
               (num_replicas + delta_plus > GPU_MAX_REPLICAS)) {
      // If this happens, we need to upgrade to a model that supports higher
      // batch sizes.
      logfile << "Exceed GPU maximum replicas: " << delta_plus << std::endl;
      return -1;
    } else if (batch < max_batch) {
      // We also need to upgrade to a higher batch model.
      logfile << "Fastest model cannot support this batch size: " << max_batch
              << std::endl;
      return -1;
    }
  } else {
    if ((num_replicas > 0) &&
        ((weighted_qps <= std::max(0.0, down_thresh)) || (mod_qps <= 0.0)) &&
        (actual_batch > 0)) {
      // If the model has 0 QPS, then scale down.
      // Only downgrade if the model is not newly loaded
      count = -1;
    } else {
      // Don't do anything.
      count = 0;
    }
  }

  *fastest_count = count;
  return 0;
}

// Scale per parent model.
void InfaasScaler(const std::string& worker_name,
                  std::unique_ptr<RedisMetadata>& rmd, std::ofstream& logfile) {
  std::vector<std::string> running_parents =
      rmd->get_parent_models_on_executor(worker_name);
  logfile << "INFaaS Scaler" << std::endl;
  for (auto& parent_name : running_parents) {
    uint64_t curr_time = get_curr_timestamp();
    logfile << "[ " << curr_time << " ] Checking " << parent_name << std::endl;
    // Get the top-10 fastest model.
    std::vector<std::string> fastest_models =
        rmd->tot_lat_bin(parent_name, 0, 10000, TOP_K_FASTEST);

    std::string fastest_var, curr_running_fastest, scaledown_fastest;
    if (fastest_models.empty()) {
      logfile << "Failed to find the fastest variant for " << parent_name
              << std::endl;
      continue;
    }

    std::vector<std::string> running_modvars =
        rmd->get_parents_variants_on_executor(parent_name, worker_name);

    // Now get the fast variant that is not running and has the next batch size
    // we can upgrade to. Capped at MAX_ONLINE_BATCH. If the fastest variant is
    // not TensorRT model, ignore this step.
    fastest_var = fastest_models[0];
    auto fastest_framework = rmd->get_model_info(fastest_var, "framework");
    int trt_max_running_batch = 0;
    // the minimum not running batch that is larger than max_running_batch.
    int trt_next_batch = MAX_ONLINE_BATCH;
    int trt_prev_batch = 0;
    std::vector<int> trt_batch;
    std::map<int, std::string> trt_batch2name;
    if (fastest_framework == "tensorrt") {
      logfile << "Checking batch sizes for tensorrt fastest models."
              << std::endl;
      // Check the running modvar and record the max_running batch.
      for (auto& modvar : running_modvars) {
        auto framework = rmd->get_model_info(modvar, "framework");
        if (framework != "tensorrt") { continue; }
        int mod_batch = std::stoi(rmd->get_model_info(modvar, "max_batch"));
        if (mod_batch > trt_max_running_batch) {
          fastest_var = modvar;
          curr_running_fastest = modvar;
          trt_max_running_batch = mod_batch;
        }
      }
      // Then find the fastest_var we should upgrade.
      // If we have no lower batch variant, set the downgrade var to the first
      // CPU model.
      std::string fastest_cpu_var = "";
      for (auto& modvar : fastest_models) {
        auto framework = rmd->get_model_info(modvar, "framework");
        if (framework != "tensorrt") {
          // Currently only consider TF CPU variant.
          if (((framework == "tensorflow-cpu") || (framework == "pytorch")) &&
              (fastest_cpu_var == "")) {
            fastest_cpu_var = modvar;
          }
          continue;
        }
        int mod_batch = std::stoi(rmd->get_model_info(modvar, "max_batch"));
        trt_batch.push_back(mod_batch);
        trt_batch2name[mod_batch] = modvar;
        // Check trt_max_running_batch < mod_batch <= trt_next_batch
        if (mod_batch <= trt_max_running_batch) { continue; }
        if (mod_batch <= trt_next_batch) {
          fastest_var = modvar;
          trt_next_batch = mod_batch;
        }
      }
      // Find the one to downgrade
      std::sort(trt_batch.begin(), trt_batch.end());
      auto idxp =
          std::find(trt_batch.begin(), trt_batch.end(), trt_max_running_batch);
      if ((idxp == trt_batch.begin()) || (idxp == trt_batch.end())) {
        trt_prev_batch = 0;
        scaledown_fastest = "";
      } else {
        trt_prev_batch = *(--idxp);
        scaledown_fastest = trt_batch2name[trt_prev_batch];
      }
      if ((scaledown_fastest == "") && (fastest_cpu_var != "")) {
        scaledown_fastest = fastest_cpu_var;
      }
    }
    logfile << "Curr running fastest variant is: " << curr_running_fastest
            << ", Next upgrade fastest variant is: " << fastest_var
            << ", Downgrade fastest variant is: " << scaledown_fastest
            << "; max running batch: " << trt_max_running_batch
            << "; next batch: " << trt_next_batch
            << "; prev batch: " << trt_prev_batch << std::endl;
    // 1) Get all running variants for this parent model. Then calculate the
    // strategy (a): scaling individually. Exclude the fastest mdoel.
    // If we found a running TensorRT that has lower batch size than the
    // current running largest batch size, then force to unload the model
    double sum_wdelta_qps = 0.0;
    double sum_cost = 0.0;
    int max_batch = 1;
    int8_t res;
    std::vector<ModvarScaleInfo> individual_scale_info;
    for (auto& modvar : running_modvars) {
      if (modvar == curr_running_fastest) {
        logfile << "\t Escaping " << modvar << std::endl;
        continue;
      }
      logfile << "\t Checking " << modvar << std::endl;
      int batch, count;
      double weighted_delta_qps;
      double load_lat;
      std::string hw;
      std::string framework;
      res =
          checkModvar(worker_name, modvar, rmd, logfile, &batch,
                      &weighted_delta_qps, &hw, &load_lat, &framework, &count);
      // Even res != 0, we still need to update sum_wdelta_qps and max_batch.
      // We just cannot actually add them to the queue.
      if (count > 0) {
        sum_wdelta_qps += weighted_delta_qps;
        max_batch = std::max(max_batch, batch);
      }
      if (res == 0) {
        // Sum of all variants that need to scale up.
        if (count > 0) {
          sum_cost += load_lat * (double)count;
          // Check it won't exceed max CPU replicas. Otherwise, add a large
          // cost.
          if (hw == "CPU") {
            int after_reps = (int)CpuModelManager::numReplicas(modvar) + count;
            if (after_reps > CPU_MAX_REPLICAS) {
              sum_cost += load_lat * 1000.0;
            }
          } else if (hw == "GPU") {
            int after_reps = (int)GpuModelManager::numReplicas(modvar) + count;
            if (after_reps > GPU_MAX_REPLICAS) {
              sum_cost += load_lat * 1000.0;
            }
          }
        }
        logfile << "\t count: " << count << "; hw: " << hw << std::endl;
      } else {
        logfile << "Failed to check modvar " << modvar << std::endl;
        if (framework != "tensorrt") {
          // For TensorRT, we still need to check whether the batch size is
          // lower than the current running batch.
          continue;
        }
      }
      if ((framework == "tensorrt") && (batch < trt_max_running_batch)) {
        logfile << "Force lower batch tensorrt model to unload: " << modvar
                << std::endl;
        // Note: what if the new variant is just loaded due to the
        // downgrading? Before the higher batch variant is unloaded, this part
        // will force the newly loaded variant to unload.
        count = -(GPU_SCALE_DOWN_DELAY * 100);
      }
      // if ((hw == "CPU") && (trt_max_running_batch > 0)) {
      //  logfile << "Force CPU model to unload: " << modvar << std::endl;
      //  count = -(CPU_SCALE_DOWN_DELAY * 100);
      //}
      if (count != 0) {
        individual_scale_info.push_back({modvar, count, load_lat, hw, ""});
      }

      logfile << std::endl;
    }

    // 2) Check the fastest model and calculate the strategy (b): scaling the
    // fastest model.
    std::string fastest_hw;
    int fastest_count;
    double fastest_loadlat;
    std::string scaled_fastest = curr_running_fastest;
    if (curr_running_fastest.empty()) { scaled_fastest = fastest_var; }
    logfile << "\t Checking scaled fastest modvar: " << scaled_fastest
            << "; sum_wdelta_qps: " << sum_wdelta_qps << std::endl;
    res = checkFastest(worker_name, scaled_fastest, rmd, logfile, max_batch,
                       sum_wdelta_qps, scaledown_fastest, &fastest_hw,
                       &fastest_loadlat, &fastest_count);
    logfile << "\t fastest_count: " << fastest_count
            << "; fastest_hw: " << fastest_hw << std::endl;
    // 0 means individual scaling up, 1 means scaling the fastest version.
    int strategy = 0;
    if (fastest_hw == "CPU") {
      strategy = 0;
    } else {
      if (res < 0) {
        // If fastest_count > 0 but res < 0, it indicates that we need to
        // upgrade a GPU model to higher batch size.
        if (fastest_count > 0) {
          logfile << "We need to upgrade to higher batch model: " << fastest_var
                  << std::endl;
          scaled_fastest = fastest_var;
          strategy = 1;
        } else {
          // Scale individually
          strategy = 0;
        }
      } else {
        // Compare two strategies. Choose one with lower cost.
        // It means currently we need to upgrade from CPU to GPU.
        double fastest_cost = fastest_loadlat * (double)fastest_count;
        logfile << "sum_cost: " << sum_cost
                << " vs. fastest cost: " << fastest_cost << std::endl;
        if ((fastest_count > 0) && (sum_cost > 1.5 * fastest_cost)) {
          strategy = 1;
        }
        // Downgrade to prev model
        if ((fastest_count < 0) && (scaledown_fastest != "")) {
          logfile << "We will downgrade to lower batch model: "
                  << scaledown_fastest << std::endl;
          // If we can downgrade, it means there is no need to upgrade CPU
          // models.
          strategy = 0;
        }
        // If we have a running fast model, no need to choose strategy a.
        if ((fastest_count == 0) && !curr_running_fastest.empty()) {
          strategy = 1;
        }
      }
    }
    logfile << "Selecting strategy " << strategy << std::endl;

    // Actual scaling. Still push all scaling down requests. Only differentiate
    // scaling up.
    if ((strategy == 1) || (fastest_count < 0)) {
      if (fastest_count != 0) {
        individual_scale_info.push_back({scaled_fastest, fastest_count,
                                         fastest_loadlat, fastest_hw,
                                         scaledown_fastest});
      }
    }

    for (auto& entry : individual_scale_info) {
      std::string modvar = entry.modvar_name;
      int count = entry.count;
      std::string hw = entry.hw;
      std::string down_var = entry.down_modvar;
      if ((strategy == 1) && (count > 0) && (modvar != fastest_var)) {
        logfile << "Escape individual scale up: " << modvar
                << "; count=" << count << std::endl;
        continue;
      }
      // If this variant is the downgrade target, then don't unload this
      // variant. Or if the variant is a CPU variant and the GPU batch-1 model
      // is downgrading.
      if ((count < -1) && (fastest_count < 0)) {
        if ((modvar == scaledown_fastest) ||
            ((hw == "CPU") && (trt_prev_batch < 1))) {
          logfile << "The fastest variant is downgrading to: " << modvar
                  << "; stop forcing scale down it. Set count=-1, ignore count="
                  << count << std::endl;
          count = -1;
        }
      }

      if (hw == "GPU") {
        res = Autoscaler::setScaleRequestGpu(modvar, count, down_var);
      } else if (hw == "CPU") {
        res = Autoscaler::setScaleRequestCpu(modvar, count);
      }
      if (res < 0) {
        logfile << "[StaticScaler] failed to set scaling request: " << modvar
                << ", count " << count << std::endl;
      }
    }

    logfile << "===============================================\n" << std::endl;
  }
}

// Scale per parent model, for Inferentia workers.
void InfaasNeuronScaler(const std::string& worker_name,
                  std::unique_ptr<RedisMetadata>& rmd, std::ofstream& logfile) {
  std::vector<std::string> running_parents =
      rmd->get_parent_models_on_executor(worker_name);
  logfile << "INFaaS Neuron Scaler" << std::endl;
  for (auto& parent_name : running_parents) {
    uint64_t curr_time = get_curr_timestamp();
    logfile << "[ " << curr_time << " ] Checking " << parent_name << std::endl;
    // Get the top-10 fastest model.
    std::vector<std::string> fastest_models =
        rmd->tot_lat_bin(parent_name, 0, 10000, TOP_K_FASTEST);

    std::string fastest_var, curr_running_fastest, scaledown_fastest;
    if (fastest_models.empty()) {
      logfile << "Failed to find the fastest variant for " << parent_name
              << std::endl;
      continue;
    } else {
      logfile << "Found " << fastest_models.size() << " fastest models." << std::endl;
    }

    std::vector<std::string> running_modvars =
        rmd->get_parents_variants_on_executor(parent_name, worker_name);

    // Now get the fast variant that is not running and has the next batch size
    // we can upgrade to. Capped at MAX_ONLINE_BATCH. If the fastest variant is
    // not Inferentia model, ignore this step.
    fastest_var = fastest_models[0];
    auto fastest_framework = rmd->get_model_info(fastest_var, "framework");
    int infa_max_running_batch = 0;
    // the minimum not running batch that is larger than max_running_batch.
    int infa_next_batch = MAX_ONLINE_BATCH;
    int infa_prev_batch = 0;
    std::vector<int> infa_batch;
    std::map<int, std::string> infa_batch2name;
    for (auto& modvar : fastest_models) {
      auto framework = rmd->get_model_info(modvar, "framework");
      if (framework == "inferentia") {
        fastest_framework = framework;
        fastest_var = modvar;
        break;
      }
    }
    if (fastest_framework == "inferentia") {
      logfile << "Checking batch sizes for inferentia fastest models."
              << std::endl;
      // Check the running modvar and record the max_running batch.
      for (auto& modvar : running_modvars) {
        auto framework = rmd->get_model_info(modvar, "framework");
        if (framework != "inferentia") { continue; }
        int mod_batch = std::stoi(rmd->get_model_info(modvar, "max_batch"));
        if (mod_batch > infa_max_running_batch) {
          fastest_var = modvar;
          curr_running_fastest = modvar;
          infa_max_running_batch = mod_batch;
        }
      }
      // Then find the fastest_var we should upgrade.
      // If we have no lower batch variant, set the downgrade var to the first
      // CPU model.
      std::string fastest_cpu_var = "";
      for (auto& modvar : fastest_models) {
        auto framework = rmd->get_model_info(modvar, "framework");
        if (framework != "inferentia") {
          // Currently only consider TF CPU variant.
          if (((framework == "tensorflow-cpu") || (framework == "pytorch")) &&
              (fastest_cpu_var == "")) {
            fastest_cpu_var = modvar;
          }
          continue;
        }
        int mod_batch = std::stoi(rmd->get_model_info(modvar, "max_batch"));
        infa_batch.push_back(mod_batch);
        infa_batch2name[mod_batch] = modvar;
        // Check infa_max_running_batch < mod_batch <= infa_next_batch
        if (mod_batch <= infa_max_running_batch) { continue; }
        if (mod_batch <= infa_next_batch) {
          fastest_var = modvar;
          infa_next_batch = mod_batch;
        }
      }
      // Find the one to downgrade
      std::sort(infa_batch.begin(), infa_batch.end());
      auto idxp =
          std::find(infa_batch.begin(), infa_batch.end(), infa_max_running_batch);
      if ((idxp == infa_batch.begin()) || (idxp == infa_batch.end())) {
        infa_prev_batch = 0;
        scaledown_fastest = "";
      } else {
        infa_prev_batch = *(--idxp);
        scaledown_fastest = infa_batch2name[infa_prev_batch];
      }
      if ((scaledown_fastest == "") && (fastest_cpu_var != "")) {
        scaledown_fastest = fastest_cpu_var;
      }
    }
    logfile << "Curr running fastest variant is: " << curr_running_fastest
            << ", Next upgrade fastest variant is: " << fastest_var
            << ", Downgrade fastest variant is: " << scaledown_fastest
            << "; max running batch: " << infa_max_running_batch
            << "; next batch: " << infa_next_batch
            << "; prev batch: " << infa_prev_batch << std::endl;
    // 1) Get all running variants for this parent model. Then calculate the
    // strategy (a): scaling individually. Exclude the fastest mdoel.
    // If we found a running Inferentia that has lower batch size than the
    // current running largest batch size, then force to unload the model
    double sum_wdelta_qps = 0.0;
    double sum_cost = 0.0;
    int max_batch = 1;
    int8_t res;
    std::vector<ModvarScaleInfo> individual_scale_info;
    for (auto& modvar : running_modvars) {
      if (modvar == curr_running_fastest) {
        logfile << "\t Escaping " << modvar << std::endl;
        continue;
      }
      logfile << "\t Checking " << modvar << std::endl;
      int batch, count;
      double weighted_delta_qps;
      double load_lat;
      std::string hw;
      std::string framework;
      res =
          checkModvar(worker_name, modvar, rmd, logfile, &batch,
                      &weighted_delta_qps, &hw, &load_lat, &framework, &count);
      // Even res != 0, we still need to update sum_wdelta_qps and max_batch.
      // We just cannot actually add them to the queue.
      if (count > 0) {
        sum_wdelta_qps += weighted_delta_qps;
        max_batch = std::max(max_batch, batch);
      }
      if (res == 0) {
        // Sum of all variants that need to scale up.
        if (count > 0) {
          sum_cost += load_lat * (double)count;
          // Check it won't exceed max CPU replicas. Otherwise, add a large
          // cost.
          if (hw == "CPU") {
            int after_reps = (int)CpuModelManager::numReplicas(modvar) + count;
            if (after_reps > CPU_MAX_REPLICAS) {
              sum_cost += load_lat * 1000.0;
            }
          } else if (hw == "INFA") {
            int after_reps = (int)InfaModelManager::numReplicas(modvar) + count;
            if (after_reps > INFA_MAX_REPLICAS) {
              sum_cost += load_lat * 1000.0;
            }
          }
        }
        logfile << "\t count: " << count << "; hw: " << hw << std::endl;
      } else {
        logfile << "Failed to check modvar " << modvar << std::endl;
        if (framework != "inferentia") {
          // For Inferentia, we still need to check whether the batch size is
          // lower than the current running batch.
          continue;
        }
      }
      if ((framework == "inferentia") && (batch < infa_max_running_batch)) {
        logfile << "Force lower batch inferentia model to unload: " << modvar
                << std::endl;
        // Note: what if the new variant is just loaded due to the
        // downgrading? Before the higher batch variant is unloaded, this part
        // will force the newly loaded variant to unload.
        count = -(INFA_SCALE_DOWN_DELAY * 100);
      }
      // if ((hw == "CPU") && (trt_max_running_batch > 0)) {
      //  logfile << "Force CPU model to unload: " << modvar << std::endl;
      //  count = -(CPU_SCALE_DOWN_DELAY * 100);
      //}
      if (count != 0) {
        individual_scale_info.push_back({modvar, count, load_lat, hw, ""});
      }

      logfile << std::endl;
    }

    // 2) Check the fastest model and calculate the strategy (b): scaling the
    // fastest model.
    std::string fastest_hw;
    int fastest_count;
    double fastest_loadlat;
    std::string scaled_fastest = curr_running_fastest;
    if (curr_running_fastest.empty()) { scaled_fastest = fastest_var; }
    logfile << "\t Checking scaled fastest modvar: " << scaled_fastest
            << "; sum_wdelta_qps: " << sum_wdelta_qps << std::endl;
    res = checkFastest(worker_name, scaled_fastest, rmd, logfile, max_batch,
                       sum_wdelta_qps, scaledown_fastest, &fastest_hw,
                       &fastest_loadlat, &fastest_count);
    logfile << "\t fastest_count: " << fastest_count
            << "; fastest_hw: " << fastest_hw << std::endl;
    // 0 means individual scaling up, 1 means scaling the fastest version.
    int strategy = 0;
    if (fastest_hw == "CPU") {
      strategy = 0;
    } else {
      if (res < 0) {
        // If fastest_count > 0 but res < 0, it indicates that we need to
        // upgrade a Inferentia model to higher batch size.
        if (fastest_count > 0) {
          logfile << "We need to upgrade to a better model: " << fastest_var
                  << std::endl;
          scaled_fastest = fastest_var;
          strategy = 1;
        } else {
          // Scale individually
          strategy = 0;
        }
      } else {
        // Compare two strategies. Choose one with lower cost.
        // It means currently we need to upgrade from CPU to GPU.
        double fastest_cost = fastest_loadlat * (double)fastest_count;
        logfile << "sum_cost: " << sum_cost
                << " vs. fastest cost: " << fastest_cost << std::endl;
        if ((fastest_count > 0) && (sum_cost > 1.5 * fastest_cost)) {
          strategy = 1;
        }
        // Downgrade to prev model
        if ((fastest_count < 0) && (scaledown_fastest != "")) {
          logfile << "We will downgrade to lower batch model: "
                  << scaledown_fastest << std::endl;
          // If we can downgrade, it means there is no need to upgrade CPU
          // models.
          strategy = 0;
        }
        // If we have a running fast model, no need to choose strategy a.
        if ((fastest_count == 0) && !curr_running_fastest.empty()) {
          strategy = 1;
        }
      }
    }
    logfile << "Selecting strategy " << strategy << std::endl;

    // Actual scaling. Still push all scaling down requests. Only differentiate
    // scaling up.
    if ((strategy == 1) || (fastest_count < 0)) {
      if (fastest_count != 0) {
        individual_scale_info.push_back({scaled_fastest, fastest_count,
                                         fastest_loadlat, fastest_hw,
                                         scaledown_fastest});
      }
    }

    for (auto& entry : individual_scale_info) {
      std::string modvar = entry.modvar_name;
      int count = entry.count;
      std::string hw = entry.hw;
      std::string down_var = entry.down_modvar;
      if ((strategy == 1) && (count > 0) && (modvar != fastest_var)) {
        logfile << "Escape individual scale up: " << modvar
                << "; count=" << count << std::endl;
        continue;
      }
      // If this variant is the downgrade target, then don't unload this
      // variant. Or if the variant is a Inferentia variant and the Inferentia
      // batch-1 model is downgrading.
      if ((count < -1) && (fastest_count < 0)) {
        if ((modvar == scaledown_fastest) ||
            ((hw == "CPU") && (infa_prev_batch < 1))) {
          logfile << "The fastest variant is downgrading to: " << modvar
                  << "; stop forcing scale down it. Set count=-1, ignore count="
                  << count << std::endl;
          count = -1;
        }
      }

      if (hw == "INFA") {
        res = Autoscaler::setScaleRequestInfa(modvar, count, down_var);
      } else if (hw == "CPU") {
        res = Autoscaler::setScaleRequestCpu(modvar, count);
      } else {
        logfile << "Don't support this hardware: " << hw << std::endl;
      }
      if (res < 0) {
        logfile << "[StaticScaler] failed to set scaling request: " << modvar
                << ", count " << count << std::endl;
      }
    }

    logfile << "===============================================\n" << std::endl;
  }
}
}  // namespace

std::mutex Autoscaler::gpu_mutex_;
std::mutex Autoscaler::cpu_mutex_;
std::mutex Autoscaler::infa_mutex_;
std::deque<ScaleRequest> Autoscaler::gpu_scale_reqs_;
std::deque<ScaleRequest> Autoscaler::cpu_scale_reqs_;
std::deque<ScaleRequest> Autoscaler::infa_scale_reqs_;
std::map<std::string, std::atomic<bool>> Autoscaler::model_available_;
std::map<std::string, std::atomic<int>> Autoscaler::model_num_scaledown_;
std::map<std::string, int> Autoscaler::model_avg_batch_;

int Autoscaler::getAvgBatch(const std::string& model_name) {
  return model_avg_batch_[model_name];
}

void Autoscaler::setAvgBatch(const std::string& model_name, int batch) {
  model_avg_batch_[model_name] = batch;
}

void Autoscaler::AutoscalerArbiter(const std::string& worker_name,
                                   const AutoscalerType& atype,
                                   std::unique_ptr<RedisMetadata>& rmd) {
  // Log to file "INFaaS/logs/worker/autoscaler_arbiter.log"
  std::ofstream logfile;
  logfile.open(infaas_log_dir + "/worker/autoscaler_arbiter.log");
  logfile << "AutoscalerArbiter " << worker_name << "; type " << atype
          << std::endl;
  int sleep_interval = 1000;
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
    switch (atype) {
      case AUTOSCALE_STATIC:
        StaticScaler(worker_name, rmd, logfile);
        break;
      case AUTOSCALE_INDIVIDUAL:
        // Enable 2 GPU instances.
        GPU_MAX_REPLICAS = 2;
        // GPU_SCALE_DOWN_DELAY = 60;
        IndividualScaler(worker_name, rmd, logfile);
        break;
      case AUTOSCALE_INFAAS:
#ifdef INFAAS_NEURON_WORKER
        // Special for Inferentia
        INFA_MAX_REPLICAS = 1;
        InfaasNeuronScaler(worker_name, rmd, logfile);
#else
        InfaasScaler(worker_name, rmd, logfile);
#endif
        break;
      case AUTOSCALE_NONE:
        // Don't do anything, never scale up or unload a model.
      default:
        break;
    }
  }
}

int8_t Autoscaler::setScaleRequestGpu(const std::string& model_name, int count,
                                      const std::string down_var) {
  // Don't generate request if the model is being scaled
  bool success = false;
  bool expected = false;
  success = model_available_[model_name].compare_exchange_strong(
      expected, true, std::memory_order_acq_rel);
  if (!success) { return -1; }

  {
    std::lock_guard<std::mutex> lock(gpu_mutex_);
    gpu_scale_reqs_.push_back(ScaleRequest{model_name, count, down_var});
  }
  return 0;
}

int8_t Autoscaler::setScaleRequestCpu(const std::string& model_name,
                                      int count) {
  // Don't generate request if the model is being scaled
  bool success = false;
  bool expected = false;
  success = model_available_[model_name].compare_exchange_strong(
      expected, true, std::memory_order_acq_rel);
  if (!success) { return -1; }

  {
    std::lock_guard<std::mutex> lock(cpu_mutex_);
    cpu_scale_reqs_.push_back(ScaleRequest{model_name, count});
  }
  return 0;
}

int8_t Autoscaler::setScaleRequestInfa(const std::string& model_name,
                                       int count, const std::string down_var) {
  // Don't generate request if the model is being scaled
  bool success = false;
  bool expected = false;
  success = model_available_[model_name].compare_exchange_strong(
      expected, true, std::memory_order_acq_rel);
  if (!success) { return -1; }

  {
    std::lock_guard<std::mutex> lock(infa_mutex_);
    infa_scale_reqs_.push_back(ScaleRequest{model_name, count, down_var});
  }
  return 0;
}


// Pop one scale request from the queue.
int8_t Autoscaler::popScaleRequestGpu(ScaleRequest* reqs) {
  int8_t res = 0;
  {
    std::lock_guard<std::mutex> lock(gpu_mutex_);
    if (gpu_scale_reqs_.size() == 0) {
      // failed to pop;
      return -1;
    } else {
      *reqs = gpu_scale_reqs_.front();
      gpu_scale_reqs_.pop_front();
    }
  }
  if (reqs->count < 0) {
    model_num_scaledown_[reqs->model_name].fetch_add(-reqs->count);
  }

  return res;
}

int8_t Autoscaler::popScaleRequestCpu(ScaleRequest* reqs) {
  int8_t res = 0;
  {
    std::lock_guard<std::mutex> lock(cpu_mutex_);
    if (cpu_scale_reqs_.size() == 0) {
      // failed to pop;
      return -1;
    } else {
      *reqs = cpu_scale_reqs_.front();
      cpu_scale_reqs_.pop_front();
    }
  }
  if (reqs->count < 0) {
    model_num_scaledown_[reqs->model_name].fetch_add(-reqs->count);
  }

  return res;
}

int8_t Autoscaler::popScaleRequestInfa(ScaleRequest* reqs) {
  int8_t res = 0;
  {
    std::lock_guard<std::mutex> lock(infa_mutex_);
    if (infa_scale_reqs_.size() == 0) {
      // failed to pop;
      return -1;
    } else {
      *reqs = infa_scale_reqs_.front();
      infa_scale_reqs_.pop_front();
    }
  }
  if (reqs->count < 0) {
    model_num_scaledown_[reqs->model_name].fetch_add(-reqs->count);
  }

  return res;
}

void Autoscaler::GpuAutoscalerDaemon(const std::string& worker_name,
                                     const AutoscalerType& atype,
                                     std::unique_ptr<RedisMetadata>& rmd,
                                     std::unique_ptr<Aws::S3::S3Client>& s3c) {
  // Log to the file "INFaaS/logs/gpu_autoscaler_daemon.log"
  std::ofstream logfile;
  logfile.open(infaas_log_dir + "/worker/gpu_autoscaler_daemon.log");
  // Set nice value = 10 to be a lower priority.
  int curr_nice = nice(10);
  logfile << "Set GpuAutoscalerDaemon thread nice = " << curr_nice << std::endl;
  int sleep_interval = 500;  // Sleep 500 msec.

  GpuModelManager manager(worker_name);
  while (true) {
    // Get one scaling request from the front, process one request at a time.
    int8_t res;
    ScaleRequest reqs;
    while (popScaleRequestGpu(&reqs) == 0) {
      std::string model_name = reqs.model_name;
      std::string trt_down_var = reqs.down_var;
      auto count = reqs.count;
      logfile << "Model name: " << model_name << "; count = " << reqs.count
              << std::endl;
      // numReplicas = -1 means the model is currently being loaded. Should
      // round to 0.
      int curr_reps = std::max(0, GpuModelManager::numReplicas(model_name));
      int after_reps = std::max(0, (int)(curr_reps + count));

      // Don't scale down until we reach the backed up threshold.
      int curr_backups = model_num_scaledown_[model_name].load();
      int scale_down_delay = GPU_SCALE_DOWN_DELAY;
      if (model_name.find("gnmt") != std::string::npos) {
        scale_down_delay = NLP_SCALE_DOWN_DELAY;
      }

      if ((count < 0) && (curr_backups <= scale_down_delay)) {
        logfile << "Scale down requests not enough: " << curr_backups
                << "; No change for " << model_name << std::endl;
        model_available_[model_name].store(false);
        continue;
      } else {
        // Clean up the backup counter
        model_num_scaledown_[model_name].store(0);
      }

      logfile << "Change from " << curr_reps << " to " << after_reps
              << std::endl;
      int8_t res = -1;
      // Unload the model if it reaches 0.
      if (after_reps == 0) {
        if (!trt_down_var.empty()) {
          logfile << "Downgrading to " << trt_down_var << std::endl;

          auto down_hw = ChooseHardware(trt_down_var, rmd);
          if (down_hw == "CPU") {
            logfile << "Will be handeled by parent scale down method."
                    << std::endl;
          } else {
            std::string model_url =
                bucket_prefix + infaas_bucket + "/" + trt_down_var;
            res = manager.LoadModel(model_url, trt_down_var, rmd, s3c);
            if (res >= 0) {
              logfile << "Loaded downgrade model " << trt_down_var << std::endl;
            } else {
              logfile << "Failed to load downgrade model " << trt_down_var
                      << std::endl;
            }
          }
        } else {
          logfile << "No model to downgrade to, unloading." << std::endl;
        }
        res = manager.UnloadModel(model_name, rmd);
      } else if (curr_reps == 0) {
        std::string model_url =
            bucket_prefix + infaas_bucket + "/" + model_name;
        res = manager.LoadModel(model_url, model_name, rmd, s3c);
      } else if (after_reps > 0) {
        // NOTE: multiple GPU replicas can cause bad performance.
        if (after_reps <= GPU_MAX_REPLICAS) {
          res = GpuModelManager::changeNumReplicas(model_name, after_reps);
        } else {
          logfile << "Exceeding GPU max number of replicas: "
                  << GPU_MAX_REPLICAS << std::endl;
        }
      }
      if (res >= 0) {
        logfile << "Finished scaling for model " << model_name << std::endl;
      } else {
        logfile << "Failed to scale for model " << model_name << std::endl;
      }
      // release the lock anyway.
      model_available_[model_name].store(false);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
  }
}

void Autoscaler::CpuAutoscalerDaemon(const std::string& worker_name,
                                     const AutoscalerType& atype,
                                     std::unique_ptr<RedisMetadata>& rmd,
                                     std::unique_ptr<Aws::S3::S3Client>& s3c) {
  // Log to the file "INFaaS/logs/cpu_autoscaler_daemon.log"
  std::ofstream logfile;
  logfile.open(infaas_log_dir + "/worker/cpu_autoscaler_daemon.log");

  // Set nice value = 10 to be a lower priority.
  int curr_nice = nice(10);
  logfile << "Set CpuAutoscalerDaemon thread nice = " << curr_nice << std::endl;
  int sleep_interval = 500;  // Sleep 500 msec.

  CpuModelManager manager(worker_name);
  while (true) {
    int8_t res;
    ScaleRequest reqs;
    // Get one scaling request from the front, process one request at a time.
    while (popScaleRequestCpu(&reqs) == 0) {
      std::string model_name = reqs.model_name;
      auto count = reqs.count;
      logfile << "Model name: " << model_name << "; count = " << reqs.count
              << std::endl;
      auto curr_reps = CpuModelManager::numReplicas(model_name);
      auto after_reps = (int)curr_reps + count;

      // Don't scale down until we reach the backed up threshold.
      int curr_backups = model_num_scaledown_[model_name].load();
      int scale_down_delay = CPU_SCALE_DOWN_DELAY;
      if (model_name.find("gnmt") != std::string::npos) {
        scale_down_delay = NLP_SCALE_DOWN_DELAY;
      }
      if ((count < 0) && (curr_backups <= scale_down_delay)) {
        logfile << "Scale down requests not enough: " << curr_backups
                << "; No change for " << model_name << std::endl;
        model_available_[model_name].store(false);
        continue;
      } else {
        // Clean up the backup counter
        model_num_scaledown_[model_name].store(0);
      }

      logfile << "Change from " << curr_reps << " to " << after_reps
              << std::endl;
      if (count > 0) {
        int after_reps = std::max(0, (int)(curr_reps + count));
        if (after_reps > CPU_MAX_REPLICAS) {
          logfile << "Exceeds maximum CPU replicas." << std::endl;
        } else {
          // Load count models
          for (int i = 0; i < count; ++i) {
            std::string model_url =
                bucket_prefix + infaas_bucket + "/" + model_name;
            std::string container_name =
                model_name + "_online_" + std::to_string(curr_reps + i);
            logfile << "Loading " << container_name << std::endl;
            auto res = manager.LoadModel(model_url, model_name, rmd, s3c,
                                         container_name, true);
            if (res < 0) {
              logfile << "Failed to load " << container_name << std::endl;
            }
          }
        }
      } else if (count < 0) {
        int to_unload = std::min(-count, (int)curr_reps);
        // Unload count models in reverse order
        for (int i = 1; i <= to_unload; ++i) {
          std::string container_name =
              model_name + "_online_" + std::to_string(curr_reps - i);
          logfile << "Unloading " << container_name << std::endl;
          auto res = manager.UnloadModel(model_name, rmd, container_name, true);
          if (res < 0) {
            logfile << "Failed to unload " << container_name << std::endl;
          }
        }
      }
      // release the lock.
      model_available_[model_name].store(false);

      logfile << "Finished scaling for model " << model_name << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
  }
}

void Autoscaler::InfaAutoscalerDaemon(const std::string& worker_name,
                                     const AutoscalerType& atype,
                                     std::unique_ptr<RedisMetadata>& rmd,
                                     std::unique_ptr<Aws::S3::S3Client>& s3c) {
  // Log to the file "INFaaS/logs/inferentia_autoscaler_daemon.log"
  std::ofstream logfile;
  logfile.open(infaas_log_dir + "/worker/inferentia_autoscaler_daemon.log");

  // Set nice value = 10 to be a lower priority.
  int curr_nice = nice(10);
  logfile << "Set InfaAutoscalerDaemon thread nice = " << curr_nice << std::endl;
  int sleep_interval = 500;  // Sleep 500 msec.

  InfaModelManager manager(worker_name);
  while (true) {
    int8_t res;
    ScaleRequest reqs;
    // Get one scaling request from the front, process one request at a time.
    while (popScaleRequestInfa(&reqs) == 0) {
      std::string model_name = reqs.model_name;
      std::string infa_down_var = reqs.down_var;
      auto count = reqs.count;
      logfile << "Model name: " << model_name << "; count = " << reqs.count
              << std::endl;
      auto curr_reps = InfaModelManager::numReplicas(model_name);
      auto after_reps = (int)curr_reps + count;

      // Don't scale down until we reach the backed up threshold.
      int curr_backups = model_num_scaledown_[model_name].load();
      int scale_down_delay = INFA_SCALE_DOWN_DELAY;
      if (model_name.find("gnmt") != std::string::npos) {
        scale_down_delay = NLP_SCALE_DOWN_DELAY;
      }

      if ((count < 0) && (curr_backups <= scale_down_delay)) {
        logfile << "Scale down requests not enough: " << curr_backups
                << "; No change for " << model_name << std::endl;
        model_available_[model_name].store(false);
        continue;
      } else {
        // Clean up the backup counter
        model_num_scaledown_[model_name].store(0);
      }

      logfile << "Change from " << curr_reps << " to " << after_reps
              << std::endl;
      if (count > 0) {
        int after_reps = std::max(0, (int)(curr_reps + count));
        if (after_reps > INFA_MAX_REPLICAS) {
          logfile << "Exceeds maximum Inferentia replicas." << std::endl;
        } else {
          // Load count models
          for (int i = 0; i < count; ++i) {
            std::string model_url =
                bucket_prefix + infaas_bucket + "/" + model_name;
            std::string container_name =
                model_name + "_online_" + std::to_string(curr_reps + i);
            logfile << "Loading " << container_name << std::endl;
            auto res = manager.LoadModel(model_url, model_name, rmd, s3c,
                                         container_name);
            if (res < 0) {
              logfile << "Failed to load " << container_name << std::endl;
            }
          }
        }
      } else if (count < 0) {
        int after_reps = std::max(0, (int)(curr_reps + count));
        // TODO: for now, we assume Inferentia will only downgrade to CPU.
        if ((after_reps == 0) && (!infa_down_var.empty())) {
          logfile << "Downgrading to " << infa_down_var << std::endl;
          auto down_hw = ChooseHardware(infa_down_var, rmd);
          if (down_hw == "CPU") {
            logfile << "Will be handled by parent scale down method." << std::endl;
          } else {
            logfile << "Currently dont't support downgrading to "
                    << infa_down_var << std::endl;
          }
        }
        int to_unload = std::min(-count, (int)curr_reps);
        // Unload count models in reverse order
        for (int i = 1; i <= to_unload; ++i) {
          std::string container_name =
              model_name + "_online_" + std::to_string(curr_reps - i);
          logfile << "Unloading " << container_name << std::endl;
          auto res = manager.UnloadModel(model_name, rmd, container_name);
          if (res < 0) {
            logfile << "Failed to unload " << container_name << std::endl;
          }
        }
      }
      // release the lock.
      model_available_[model_name].store(false);

      logfile << "Finished scaling for model " << model_name << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_interval));
  }
}

}  // namespace internal
}  // namespace infaas
