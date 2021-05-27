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

#ifndef REDIS_METADATA_H
#define REDIS_METADATA_H

#include <cstdint>
#include <set>
#include <string>
#include <utility>  // pair
#include <vector>

#include <redox.hpp>

// INFaaS metadata suffixes and set names
#define VMSCALE_KEY "vmscale"
#define SLACKSCALE_KEY "slackscale"
#define CPUEXEC_KEY "numcpuexec"
#define INFERENTIAEXEC_KEY "numinferentiaexec"
#define ALLEXEC_SET "allexecutors"
#define GMOD_SET "gmod_set"
#define MODEL_SET "model_set"
#define MODELVAR_SET "modelvar_set"
#define CPUUTIL_SET "cpuutil_set"
#define GPUUTIL_SET "gpuutil_set"
#define INFERENTIAUTIL_SET "inferentiautil_set"
#define RUNMODS_SET "allrunning"
#define CPUEXEC_SUFF "cpuexec"
#define INFERENTIAEXEC_SUFF "inferentiaexec"
#define PTONLY_SUFF "pytorch_only"
#define SDOWN_SUFF "scaledown"  // executor_name + parent_model + SDOWN_SUFF
#define LOADUNL_SUFF "scaledown"
#define LOADLAT_SUFF "loadlat"
#define INFLAT_SUFF "b1inflat"
#define TOTLAT_SUFF "b1totlat"
#define ACCURACY_SUFF "accuracy"
#define GPARACC_SUFF "gparacc"  // gparent_model_name + <acc_bin> + GPARACC_SUFF
#define GPARACCBIN_SUFF "gparaccbin"
#define MODQPS_SUFF "modqps"
#define MODAVGLAT_SUFF "modavglat"
#define EXECMOD_SUFF "models"
#define EXECMVAR_SUFF "modvar"
#define MODINFO_SUFF "info"
#define MODVAR_SUFF "modvariants"
#define GPARENT_SUFF "gparent"
#define PARENT_SUFF "parent"
#define RUNCHILD_SUFF "runningchild"
#define RUNCHIEX_SUFF \
  "runningexecchild"  // executor_name + parent_model + RUNCHILD_SUFF
#define RUNMVARS_SUFF "runningmvars"
#define RUNMODS_SUFF "runningmods"
#define INSTID_SUFF "instid"
#define BLIST_SUFF "blist"
#define BLISTMOD_SUFF "blistmod"  // executor_name + model_name + BLISTMOD_SUFF
#define SLACK_SUFF "slack"

struct Address {
  std::string ip;
  std::string port;
};

// Accuracy bins for grandparent models.
// Bins are allocated as {n,n+1} pairs (so there is one less bin than #values)
static const double gpar_accuracy_bins[] = {0.0, 50.0, 70.0, 75.0, 78.0, 100.0};
static const int8_t num_gpar_bins = 5;

class RedisMetadata {
public:
  RedisMetadata(struct Address redis_server);

  // Add executor-address
  int8_t add_executor_addr(const std::string& executor_name,
                           const struct Address& addr);

  // Retrieve address of executor
  const struct Address get_executor_addr(const std::string& executor_name);

  // Add executor-InstanceId for AWS
  int8_t add_executor_instid(const std::string& executor_name,
                             const std::string& instid);

  // Retrieve InstanceId of executor
  std::string get_executor_instid(const std::string& executor_name);

  // Designate an executor as being CPU only
  int8_t set_exec_onlycpu(const std::string& executor_name);

  // Check if an executor is CPU only
  int8_t is_exec_onlycpu(const std::string& executor_name);

  // Designate an executor as supporting Inferentia
  int8_t set_exec_inferentia(const std::string& executor_name);

  // Check if an executor supports Inferentia
  int8_t is_exec_inferentia(const std::string& executor_name);

  // Designate an executor as being slack for GPU
  int8_t set_exec_slack(const std::string& executor_name,
                        const std::string& model_variant = "0");

  // Check if an executor is slack for GPU.
  // If it is, return the variant running, otherwise return 0.
  // If it isn't, return NS
  std::string is_exec_slack(const std::string& executor_name);

  // Check if executor exists
  int8_t executor_exists(const std::string& executor_name);

  // Blacklist an executor
  int8_t blacklist_executor(const std::string& executor_name,
                            const int16_t& expire_time = 30);

  // Check if executor is blacklisted
  bool is_blacklisted(const std::string& executor_name);

  // Set VM scale out request
  int8_t set_vm_scale();

  // Unset VM scale out request
  int8_t unset_vm_scale();

  // Check value of VM scale out flag
  int8_t vm_scale_status();

  // Set slack scale out request
  int8_t set_slack_scale();

  // Unset slack scale out request
  int8_t unset_slack_scale();

  // Check value of slack scale out flag
  int8_t slack_scale_status();

  // Delete executor.
  int8_t delete_executor(const std::string& executor_name);

  // Get number of executors
  int16_t get_num_executors();

  // Get number of CPU-only executors
  int16_t get_num_cpu_executors();

  // Get number of Inferentia executors
  int16_t get_num_inferentia_executors();

  // Get all executors
  std::vector<std::string> get_all_executors();

  // Add grandparent model. Model variants will be grouped by bins
  int8_t add_gparent_model(const std::string& gparent_model_name);

  // Add parent model. All model variants will fan out from here
  int8_t add_parent_model(const std::string& parent_model_name);

  // Add model variant with some metadata. The rest of the metadata requires
  // profiling
  int8_t add_model(const std::string& model_name,
                   const std::string& parent_model_name,
                   const std::string& gparent_model_name,
                   const double& comp_size, const double& accuracy,
                   const std::string& dataset, const std::string& submitter,
                   const std::string& framework, const std::string& task,
                   const int16_t& img_dimensions, const int16_t& max_batch,
                   const double& load_latency, const double& inf_latency,
                   const double& peak_memory, const double& slope,
                   const double& intercept);

  // Get parent model of a model variant
  std::string get_parent_model(const std::string& model_name);

  // Get all model variants for a "parent model"
  std::vector<std::string> get_all_model_variants(
      const std::string& parent_model_name);

  // Get all parent models for a particular task and dataset
  std::vector<std::string> get_all_parent_models(const std::string& task,
                                                 const std::string& dataset);

  // Check if grandparent model exists/is registered
  bool gparent_model_registered(const std::string& gparent_model_name);

  // Check if parent model exists/is registered
  bool parent_model_registered(const std::string& parent_model_name);

  // Check if model variant exists/is registered
  bool model_registered(const std::string& model_name);

  // Get all running models
  std::vector<std::string> get_all_running_models();

  // Get model variant load latency
  double get_load_lat(const std::string& model_name);

  // Get model variant batch-1 inference latency
  double get_inf_lat(const std::string& model_name);

  // Get model variant accuracy
  double get_accuracy(const std::string& model_name);

  // Get batch-1 inference latency bin
  std::vector<std::string> inf_lat_bin(const std::string& parent_model_name,
                                       const double& min_lat,
                                       const double& max_lat,
                                       const int8_t& max_results = 5);

  // Get total latency bin (batch-1 for inference)
  std::vector<std::string> tot_lat_bin(const std::string& parent_model_name,
                                       const double& min_lat,
                                       const double& max_lat,
                                       const int8_t& max_results = 5);

  // Get parent accuracy bin
  std::vector<std::string> par_acc_bin(const std::string& parent_model_name,
                                       const double& min_acc,
                                       const int8_t& max_results = 5);

  // Get grandparent accuracy bin.
  // The main difference with this function and par_acc_bin is that
  //// we first find the appropriate bin before making the REDIS query
  std::vector<std::string> gpar_acc_bin(
      const std::string& grandparent_model_name, const double& min_acc,
      const int8_t& max_results = 5);

  // Add running model variant on an executor
  int8_t add_running_model(const std::string& executor_name,
                           const std::string& model_name);

  // Remove running model completely from an executor.
  // Note that this will remove all running instances on both
  //// CPU and GPU from the metadata store
  int8_t remove_running_model(const std::string& executor_name,
                              const std::string& model_name);

  // Checks if parent model is running on any executor.
  // If executor_name is left empty, will return true if model is running on any
  // instance
  int8_t is_parent_model_running(const std::string& parent_model_name,
                                 const std::string& executor_name = "");

  // Checks if model variant is running on any executor.
  // If executor_name is left empty, will return true if model is running on any
  // instance
  int8_t is_model_running(const std::string& model_name,
                          const std::string& executor_name = "");

  // Checks if parent has only PyTorch models
  int8_t is_pytorch_only(const std::string& parent_model_name);

  // Get all parent models running on a particular executor
  std::vector<std::string> get_parent_models_on_executor(
      const std::string& executor_name);

  // Get all model variants running on a particular executor
  std::vector<std::string> get_variants_on_executor(
      const std::string& executor_name);

  // Get all of a parent model's variants running on a particular executor
  std::vector<std::string> get_parents_variants_on_executor(
      const std::string& parent_model_name, const std::string& executor_name);

  // Get model variant information by name
  // info can be any of the following:
  //// comp_size, dataset, submitter, framework, task,
  //// max_batch, peak_memory, img_dim, slope, intercept
  std::string get_model_info(const std::string& model_name,
                             const std::string& info);

  // Update parent model QPS for a particular executor
  int8_t update_parentmodel_qps(
      const std::string& executor_name,
      const std::vector<std::pair<std::string, double>>& mod_qps);

  // Update model QPS for a particular executor
  int8_t update_model_qps(const std::string& executor_name,
                          const std::string& model_name, const double& qps);

  // Get model QPS for a particular executor
  double get_model_qps(const std::string& executor_name,
                       const std::string& model_name);

  // Get executor name with minimum QPS
  std::vector<std::string> min_qps_name(const std::string& model_name,
                                        const int8_t& max_results);

  // Get minimum overall QPS for a model
  double get_min_qps(const std::string& model_name);

  // Update model average latency for a particular executor
  int8_t update_model_avglat(const std::string& executor_name,
                             const std::string& model_name,
                             const double& avg_lat);

  // Get model average latency on a particular executor
  double get_model_avglat(const std::string& executor_name,
                          const std::string& model_name);

  // Set blacklist model on worker based on average latency
  int8_t set_model_avglat_blacklist(const std::string& executor_name,
                                    const std::string& model_name);

  // Unset blacklist model on worker based on average latency
  int8_t unset_model_avglat_blacklist(const std::string& executor_name,
                                      const std::string& model_name);

  // Check if model is blackisted on worker based on average latency
  int8_t get_model_avglat_blacklist(const std::string& executor_name,
                                    const std::string& model_name);

  // Set if parent model is servicing too many CPU requests with a GPU
  int8_t set_parent_scaledown(const std::string& executor_name,
                              const std::string& parent_model_name);

  // Unset if parent model is NOT servicing too many CPU requests with a GPU
  int8_t unset_parent_scaledown(const std::string& executor_name,
                                const std::string& parent_model_name);

  // Check if parent model is servicing too many CPU requests with a GPU
  int8_t get_parent_scaledown(const std::string& executor_name,
                              const std::string& parent_model_name);

  // Set if model is being loaded or unloaded on an executor
  int8_t set_model_load_unload(const std::string& model_name);

  // Set if model is NOT being loaded or unloaded on an executor
  int8_t unset_model_load_unload(const std::string& model_name);

  // Check if model is being loaded or unloaded on an executor
  int8_t get_model_load_unload(const std::string& model_name);

  // Remove a model variant from the metadata store
  int8_t delete_model(const std::string& model_name);

  // Update CPU utilization on executor
  int8_t update_cpu_util(const std::string& executor_name,
                         const double& utilization,
                         const int8_t& first_time = 0);

  // Update GPU utilization on executor
  int8_t update_gpu_util(const std::string& executor_name,
                         const double& utilization,
                         const int8_t& first_time = 0);

  // Update Inferentia utilization on executor
  int8_t update_inferentia_util(const std::string& executor_name,
                                const double& utilization,
                                const int8_t& first_time = 0);

  // Get CPU utilization on executor
  double get_cpu_util(const std::string& executor_name);

  // Get GPU utilization on executor
  double get_gpu_util(const std::string& executor_name);

  // Get Inferentia utilization on executor
  double get_inferentia_util(const std::string& executor_name);

  // Get executor with the maximum CPU utilization
  std::vector<std::string> max_cpu_util_name(const double& max_thresh = 100,
                                             const int8_t& max_results = 3);

  // Get executor with the minimum CPU utilization
  std::vector<std::string> min_cpu_util_name(const int8_t& max_results = 3);

  // Get minimum CPU utilization across all executors
  double get_min_cpu_util();

  // Get executor with the maximum GPU utilization
  std::vector<std::string> max_gpu_util_name(const double& max_thresh = 100,
                                             const int8_t& max_results = 3);

  // Get executor with the minimum GPU utilization
  std::vector<std::string> min_gpu_util_name(const int8_t& max_results = 3);

  // Get minimum GPU utilization across all executors
  double get_min_gpu_util();

  // Get executor with the maximum Inferentia utilization
  std::vector<std::string> max_inferentia_util_name(const double& max_thresh = 100,
                                                    const int8_t& max_results = 3);

  // Get executor with the minimum Inferentia utilization
  std::vector<std::string> min_inferentia_util_name(const int8_t& max_results = 3);

  // Get minimum Inferentia utilization across all executors
  double get_min_inferentia_util();

  // Convenient function to check if Address is empty Address
  static bool is_empty_address(const struct Address& addr);

  // Convenient function to turn Address struct into string
  static const std::string Address_to_str(const struct Address& addr);

private:
  bool key_exists(const std::string& key);
  bool set_member(const std::string& key, const std::string& field);
  bool gmodel_exists(const std::string& model);
  bool model_exists(const std::string& model);
  bool modelvar_exists(const std::string& model);
  bool hash_exists(const std::string& key, const std::string& field);

  int8_t check_pytorch_status(const std::string& model);

  static const struct Address empty_addr;

  struct Address redis_server_;
  redox::Redox rdx_;
};

#endif
