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

#include <algorithm>  // find
#include <exception>  // If connection to Redis fails in constructor
#include <iostream>
#include <map>
#include <sstream>

#include "redis_metadata.h"

using namespace redox;

const struct Address RedisMetadata::empty_addr = {"0", "0"};

RedisMetadata::RedisMetadata(struct Address redis_server)
    : redis_server_(redis_server) {
  // Initialize connection to Redis server
  uint16_t redis_port = stoi(redis_server_.port);
  if (!rdx_.connect(redis_server_.ip, redis_port)) {
    throw std::runtime_error("Failed to connect to Redis server");
  }
  std::cout << "[Redis Metadata]: Successfully connected" << std::endl;
}

int8_t RedisMetadata::add_executor_addr(const std::string& executor_name,
                                        const struct Address& addr) {
  const std::string exec_addr = addr.ip + ":" + addr.port;
  Command<std::string>& c_exec_addr =
      rdx_.commandSync<std::string>({"SET", executor_name, exec_addr});
  if (!c_exec_addr.ok()) { return -1; }

  // Add to all executor set
  Command<int>& c_add_exec =
      rdx_.commandSync<int>({"SADD", ALLEXEC_SET, executor_name});
  if (!c_add_exec.ok()) { return -1; }

  // Call update_cpu_util and update_gpu_util for initialization to 0
  //// For inferentia, always set to 101.0. It will be set to 0.0 if supported
  if ((update_cpu_util(executor_name, 0.0, 1) == -1) ||
      (update_gpu_util(executor_name, 0.0, 1) == -1) ||
      (update_inferentia_util(executor_name, 101.0, 1) == -1)) {
    return -1;
  }

  return 0;
}

const struct Address RedisMetadata::get_executor_addr(
    const std::string& executor_name) {
  // Check that executor exists
  if (!key_exists(executor_name)) {
    std::cout << "[Redis Metadata]: " << executor_name << " does not exist"
              << std::endl;
    return empty_addr;
  }

  std::string return_ip, return_port;
  Command<std::string>& c_exec_addr =
      rdx_.commandSync<std::string>({"GET", executor_name});
  if (!c_exec_addr.ok()) { return empty_addr; }
  std::string reply = c_exec_addr.reply();

  std::istringstream ss(reply);
  std::getline(ss, return_ip, ':');
  std::getline(ss, return_port, ':');
  return {return_ip, return_port};
}

int8_t RedisMetadata::add_executor_instid(const std::string& executor_name,
                                          const std::string& instid) {
  const std::string instid_name = executor_name + "-" + INSTID_SUFF;
  Command<std::string>& c_exec_instid =
      rdx_.commandSync<std::string>({"SET", instid_name, instid});
  if (!c_exec_instid.ok()) { return -1; }

  return 0;
}

std::string RedisMetadata::get_executor_instid(
    const std::string& executor_name) {
  // Check that executor exists
  if (!key_exists(executor_name)) {
    std::cout << "[Redis Metadata]: " << executor_name << " does not exist"
              << std::endl;
    return "FAIL";
  }

  const std::string instid_name = executor_name + "-" + INSTID_SUFF;
  Command<std::string>& c_exec_instid =
      rdx_.commandSync<std::string>({"GET", instid_name});
  if (!c_exec_instid.ok()) { return "FAIL"; }

  std::string reply = c_exec_instid.reply();

  return reply;
}

int8_t RedisMetadata::set_exec_onlycpu(const std::string& executor_name) {
  // Check that executor exists
  if (!key_exists(executor_name)) {
    std::cout << "[Redis Metadata]: " << executor_name << " does not exist"
              << std::endl;
    return -1;
  }
  const std::string exec_cpu = executor_name + "-" + CPUEXEC_SUFF;

  // The actual value of exec_cpu doesn't matter; it's simply a flag
  Command<std::string>& c_exec_cpu =
      rdx_.commandSync<std::string>({"SET", exec_cpu, "1"});
  if (!c_exec_cpu.ok()) { return -1; }

  // Update the gpu and inferentia utilization to be over 100 to
  //// make it get skipped during decision-making
  if (update_gpu_util(executor_name, 101.0, 0) == -1) { return -1; }
  if (update_inferentia_util(executor_name, 101.0, 0) == -1) { return -1; }

  // Increment the CPU executor counter
  Command<int>& c_numcpuexec = rdx_.commandSync<int>({"INCR", CPUEXEC_KEY});
  if (!c_numcpuexec.ok()) { return -1; }

  return 0;
}

int8_t RedisMetadata::is_exec_onlycpu(const std::string& executor_name) {
  // Check that executor exists
  if (!key_exists(executor_name)) {
    std::cout << "[Redis Metadata]: " << executor_name << " does not exist"
              << std::endl;
    return -1;
  }

  // Since this is only set for some executors, simply check if the key exists
  const std::string exec_cpu = executor_name + "-" + CPUEXEC_SUFF;
  if (key_exists(exec_cpu)) {
    return 1;
  } else {
    return 0;
  }
}

int8_t RedisMetadata::set_exec_inferentia(const std::string& executor_name) {
  // Check that executor exists
  if (!key_exists(executor_name)) {
    std::cout << "[Redis Metadata]: " << executor_name << " does not exist"
              << std::endl;
    return -1;
  }
  const std::string exec_inferentia = executor_name + "-" + INFERENTIAEXEC_SUFF;

  // The actual value of exec_inferentia doesn't matter; it's simply a flag
  Command<std::string>& c_exec_inferentia =
      rdx_.commandSync<std::string>({"SET", exec_inferentia, "1"});
  if (!c_exec_inferentia.ok()) { return -1; }

  // Update the gpu utilization to be over 100 to make it get skipped
  //// during decision-making. Also set inferentia util to 0.0.
  if (update_gpu_util(executor_name, 101.0, 0) == -1) { return -1; }
  if (update_inferentia_util(executor_name, 0.0, 0) == -1) { return -1; }

  // Increment the Inferentia executor counter
  Command<int>& c_numinferentiaexec = rdx_.commandSync<int>({"INCR", INFERENTIAEXEC_KEY});
  if (!c_numinferentiaexec.ok()) { return -1; }

  return 0;
}

int8_t RedisMetadata::is_exec_inferentia(const std::string& executor_name) {
  // Check that executor exists
  if (!key_exists(executor_name)) {
    std::cout << "[Redis Metadata]: " << executor_name << " does not exist"
              << std::endl;
    return -1;
  }

  // Since this is only set for some executors, simply check if the key exists
  const std::string exec_inferentia = executor_name + "-" + INFERENTIAEXEC_SUFF;
  if (key_exists(exec_inferentia)) {
    return 1;
  } else {
    return 0;
  }
}

int8_t RedisMetadata::set_exec_slack(const std::string& executor_name,
                                     const std::string& model_variant) {
  // Check that executor exists
  if (!key_exists(executor_name)) {
    std::cout << "[Redis Metadata]: " << executor_name << " does not exist"
              << std::endl;
    return -1;
  }
  const std::string exec_slack = executor_name + "-" + SLACK_SUFF;

  // Default is 0. Otherwise, set to the variant that is running on it
  Command<std::string>& c_exec_slack =
      rdx_.commandSync<std::string>({"SET", exec_slack, model_variant});
  if (!c_exec_slack.ok()) { return -1; }

  return 0;
}

std::string RedisMetadata::is_exec_slack(const std::string& executor_name) {
  // Check that executor exists
  if (!key_exists(executor_name)) {
    std::cout << "[Redis Metadata]: " << executor_name << " does not exist"
              << std::endl;
    return "FAIL";
  }

  const std::string exec_slack = executor_name + "-" + SLACK_SUFF;

  // If key doesn't exist, it is not slack
  if (!key_exists(exec_slack)) { return "NS"; }

  Command<std::string>& c_exec_slack =
      rdx_.commandSync<std::string>({"GET", exec_slack});
  if (!c_exec_slack.ok()) { return "FAIL"; }

  std::string reply = c_exec_slack.reply();

  return reply;
}

int8_t RedisMetadata::executor_exists(const std::string& executor_name) {
  return key_exists(executor_name);
}

int8_t RedisMetadata::blacklist_executor(const std::string& executor_name,
                                         const int16_t& expire_time) {
  // Check that executor exists
  if (!key_exists(executor_name)) {
    std::cout << "[Redis Metadata]: " << executor_name << " does not exist"
              << std::endl;
    return -1;
  }
  const std::string blist_name = executor_name + "-" + BLIST_SUFF;

  // The actual value of blist_name doesn't matter; it's simply a flag
  Command<std::string>& c_exec_blist =
      rdx_.commandSync<std::string>({"SET", blist_name, "1"});
  if (!c_exec_blist.ok()) { return -1; }

  // Now set a TTL
  Command<int>& c_exec_expire = rdx_.commandSync<int>(
      {"EXPIRE", blist_name, std::to_string(expire_time)});
  if (!c_exec_expire.ok()) { return -1; }

  return 0;
}

bool RedisMetadata::is_blacklisted(const std::string& executor_name) {
  // Check that executor exists.
  // The input should never fail this check because it should come from what the
  // storage itself has recorded.
  if (!key_exists(executor_name)) {
    std::cout << "[Redis Metadata]: " << executor_name << " does not exist"
              << std::endl;
    return true;
  }

  const std::string blist_name = executor_name + "-" + BLIST_SUFF;
  Command<int>& c_exec_exists = rdx_.commandSync<int>({"EXISTS", blist_name});
  if (!c_exec_exists.ok()) { return -1; }

  int reply = c_exec_exists.reply();

  if (reply == 1) {
    return true;
  } else {
    return false;
  }
}

int8_t RedisMetadata::set_vm_scale() {
  Command<std::string>& c_vmscale =
      rdx_.commandSync<std::string>({"SET", VMSCALE_KEY, "1"});
  if (!c_vmscale.ok()) { return -1; }
  return 0;
}

int8_t RedisMetadata::unset_vm_scale() {
  Command<std::string>& c_vmscale =
      rdx_.commandSync<std::string>({"SET", VMSCALE_KEY, "0"});
  if (!c_vmscale.ok()) { return -1; }
  return 0;
}

int8_t RedisMetadata::vm_scale_status() {
  Command<std::string>& c_vmscale =
      rdx_.commandSync<std::string>({"GET", VMSCALE_KEY});
  if (!c_vmscale.ok()) { return -1; }

  int8_t reply = std::stoi(c_vmscale.reply());

  return reply;
}

int8_t RedisMetadata::set_slack_scale() {
  Command<std::string>& c_slackscale =
      rdx_.commandSync<std::string>({"SET", SLACKSCALE_KEY, "1"});
  if (!c_slackscale.ok()) { return -1; }
  return 0;
}

int8_t RedisMetadata::unset_slack_scale() {
  Command<std::string>& c_slackscale =
      rdx_.commandSync<std::string>({"SET", SLACKSCALE_KEY, "0"});
  if (!c_slackscale.ok()) { return -1; }
  return 0;
}

int8_t RedisMetadata::slack_scale_status() {
  Command<std::string>& c_slackscale =
      rdx_.commandSync<std::string>({"GET", SLACKSCALE_KEY});
  if (!c_slackscale.ok()) { return -1; }

  int8_t reply = std::stoi(c_slackscale.reply());

  return reply;
}

int8_t RedisMetadata::delete_executor(const std::string& executor_name) {
  // Delete CPU only key if applicable
  const std::string exec_cpu = executor_name + "-" + CPUEXEC_SUFF;
  if (is_exec_onlycpu(executor_name)) {
    Command<int>& c_exec_cpu_del = rdx_.commandSync<int>({"DEL", exec_cpu});
    if (!c_exec_cpu_del.ok()) { return -1; }

    // Decrement the CPU executor counter
    Command<int>& c_numcpuexec = rdx_.commandSync<int>({"DECR", CPUEXEC_KEY});
    if (!c_numcpuexec.ok()) { return -1; }
  }

  // Delete Inferentia only key if applicable
  const std::string exec_inferentia = executor_name + "-" + INFERENTIAEXEC_SUFF;
  if (is_exec_inferentia(executor_name)) {
    Command<int>& c_exec_inferentia_del = rdx_.commandSync<int>({"DEL", exec_inferentia});
    if (!c_exec_inferentia_del.ok()) { return -1; }

    // Decrement the Inferentia executor counter
    Command<int>& c_numinferentiaexec = rdx_.commandSync<int>({"DECR", INFERENTIAEXEC_KEY});
    if (!c_numinferentiaexec.ok()) { return -1; }
  }

  // Delete slack key if applicable
  const std::string exec_slack = executor_name + "-" + SLACK_SUFF;
  if (key_exists(executor_name)) {
    Command<int>& c_exec_slack_del = rdx_.commandSync<int>({"DEL", exec_slack});
    if (!c_exec_slack_del.ok()) { return -1; }
  }

  // Delete from CPU utilization sorted set
  Command<int>& c_exec_cpu_del =
      rdx_.commandSync<int>({"ZREM", CPUUTIL_SET, executor_name});
  if (!c_exec_cpu_del.ok()) { return -1; }

  // Delete from GPU utilization sorted set
  Command<int>& c_exec_gpu_del =
      rdx_.commandSync<int>({"ZREM", GPUUTIL_SET, executor_name});
  if (!c_exec_gpu_del.ok()) { return -1; }

  // Delete from Inferentia utilization sorted set
  Command<int>& c_exec_inferentia_del =
      rdx_.commandSync<int>({"ZREM", INFERENTIAUTIL_SET, executor_name});
  if (!c_exec_inferentia_del.ok()) { return -1; }

  // Delete from all executor set
  Command<int>& c_all_exec_del =
      rdx_.commandSync<int>({"SREM", ALLEXEC_SET, executor_name});
  if (!c_all_exec_del.ok()) { return -1; }

  // Set all models that were running on it to be no longer running
  const std::string exec_mvar_name = executor_name + "-" + EXECMVAR_SUFF;
  Command<std::set<std::string>>& c_exec_models =
      rdx_.commandSync<std::set<std::string>>({"SMEMBERS", exec_mvar_name});
  if (!c_exec_models.ok()) { return -1; }

  std::set<std::string> reply = c_exec_models.reply();
  for (auto mod : reply) { remove_running_model(executor_name, mod); }

  // Delete executor-to-model set
  const std::string exec_mod_name = executor_name + "-" + EXECMOD_SUFF;
  Command<int>& c_exec_mod_del = rdx_.commandSync<int>({"DEL", exec_mod_name});
  if (!c_exec_mod_del.ok()) { return -1; }

  // Delete executor-to-model variant set
  Command<int>& c_exec_mvar_del =
      rdx_.commandSync<int>({"DEL", exec_mvar_name});
  if (!c_exec_mvar_del.ok()) { return -1; }

  // Delete executor-address (now safe)
  Command<int>& c_exec_del = rdx_.commandSync<int>({"DEL", executor_name});
  if (!c_exec_del.ok()) { return -1; }

  return 0;
}

int16_t RedisMetadata::get_num_executors() {
  // Check if CPU utilization set exists. If not, there are no executors
  if (!key_exists(CPUUTIL_SET)) { return 0; }

  // Check length of CPU utilization set.
  Command<long long int>& c_numexec =
      rdx_.commandSync<long long int>({"ZCARD", CPUUTIL_SET});
  if (!c_numexec.ok()) { return -1; }

  long long int numexec_reply = c_numexec.reply();

  return (int16_t)numexec_reply;
}

int16_t RedisMetadata::get_num_cpu_executors() {
  // Check if CPUEXEC_KEY exists. If not, there are no CPU executors
  if (!key_exists(CPUEXEC_KEY)) { return 0; }

  Command<std::string>& c_numcpuexec =
      rdx_.commandSync<std::string>({"GET", CPUEXEC_KEY});
  if (!c_numcpuexec.ok()) { return -1; }

  int8_t reply = std::stoi(c_numcpuexec.reply());

  return reply;
}

int16_t RedisMetadata::get_num_inferentia_executors() {
  // Check if INFERENTIAEXEC_KEY exists. If not, there are no Inferentia executors
  if (!key_exists(INFERENTIAEXEC_KEY)) { return 0; }

  Command<std::string>& c_numinferentiaexec =
      rdx_.commandSync<std::string>({"GET", INFERENTIAEXEC_KEY});
  if (!c_numinferentiaexec.ok()) { return -1; }

  int8_t reply = std::stoi(c_numinferentiaexec.reply());

  return reply;
}

std::vector<std::string> RedisMetadata::get_all_executors() {
  Command<std::vector<std::string>>& c_allexec_set =
      rdx_.commandSync<std::vector<std::string>>({"SMEMBERS", ALLEXEC_SET});
  if (!c_allexec_set.ok()) { return {}; }

  std::vector<std::string> reply = c_allexec_set.reply();

  return reply;
}

int8_t RedisMetadata::add_gparent_model(const std::string& gparent_model_name) {
  // Check that grandparent model DOESN'T exists
  if (gmodel_exists(gparent_model_name)) {
    std::cout << "[Redis Metadata]: " << gparent_model_name << " already exists"
              << std::endl;
    return -1;
  }

  // Add to grandparent model set
  Command<int>& c_add_model =
      rdx_.commandSync<int>({"SADD", GMOD_SET, gparent_model_name});
  if (!c_add_model.ok()) { return -1; }
}

int8_t RedisMetadata::add_parent_model(const std::string& parent_model_name) {
  // Check that model DOESN'T exists
  if (model_exists(parent_model_name)) {
    std::cout << "[Redis Metadata]: " << parent_model_name << " already exists"
              << std::endl;
    return -1;
  }

  // Add to model set
  Command<int>& c_add_model =
      rdx_.commandSync<int>({"SADD", MODEL_SET, parent_model_name});
  if (!c_add_model.ok()) { return -1; }
}

int8_t RedisMetadata::add_model(
    const std::string& model_name, const std::string& parent_model_name,
    const std::string& gparent_model_name, const double& comp_size,
    const double& accuracy, const std::string& dataset,
    const std::string& submitter, const std::string& framework,
    const std::string& task, const int16_t& img_dimensions,
    const int16_t& max_batch, const double& load_latency,
    const double& inf_latency, const double& peak_memory, const double& slope,
    const double& intercept) {
  // Check that model variant DOESN'T exists
  if (modelvar_exists(model_name)) {
    std::cout << "[Redis Metadata]: " << model_name << " already registered"
              << std::endl;
    return -1;
  }

  // Add to all model variant set
  Command<int>& c_add_model =
      rdx_.commandSync<int>({"SADD", MODELVAR_SET, model_name});
  if (!c_add_model.ok()) { return -1; }

  // Link grandparent model->model variant
  const std::string var_gpar_name = model_name + "-" + GPARENT_SUFF;
  Command<std::string>& c_var_gpar =
      rdx_.commandSync<std::string>({"SET", var_gpar_name, gparent_model_name});
  if (!c_var_gpar.ok()) { return -1; }

  // Link model->model variant
  const std::string var_par_name = model_name + "-" + PARENT_SUFF;
  Command<std::string>& c_var_par =
      rdx_.commandSync<std::string>({"SET", var_par_name, parent_model_name});
  if (!c_var_par.ok()) { return -1; }

  // Add to model->model variant set that is sorted by inference latency
  const std::string par_child_name = parent_model_name + "-" + MODVAR_SUFF;
  Command<int>& c_var_par_sset = rdx_.commandSync<int>(
      {"ZADD", par_child_name, std::to_string(inf_latency), model_name});
  if (!c_var_par_sset.ok()) { return -1; }

  // Create model info hash lookup table
  const std::string model_info_name = model_name + "-" + MODINFO_SUFF;
  Command<std::string>& c_modinfo_htable = rdx_.commandSync<std::string>(
      {"HMSET",        model_info_name,
       "comp_size",    std::to_string(comp_size),
       "dataset",      dataset,
       "submitter",    submitter,
       "framework",    framework,
       "task",         task,
       "max_batch",    std::to_string(max_batch),
       "load_latency", std::to_string(load_latency),
       "inf_latency",  std::to_string(inf_latency),
       "peak_memory",  std::to_string(peak_memory),
       "img_dim",      std::to_string(img_dimensions),
       "slope",        std::to_string(slope),
       "intercept",    std::to_string(intercept)});
  if (!c_modinfo_htable.ok()) { return -1; }

  // Add to parent model's load latency set
  const std::string load_lat_name = parent_model_name + LOADLAT_SUFF;
  Command<int>& c_load_lat_sset = rdx_.commandSync<int>(
      {"ZADD", load_lat_name, std::to_string(load_latency), model_name});
  if (!c_load_lat_sset.ok()) { return -1.0; }

  // Add to parent model's inference latency set
  const std::string inf_lat_name = parent_model_name + INFLAT_SUFF;
  Command<int>& c_inf_lat_sset = rdx_.commandSync<int>(
      {"ZADD", inf_lat_name, std::to_string(inf_latency), model_name});
  if (!c_inf_lat_sset.ok()) { return -1.0; }

  // Add to parent model's total latency set
  const std::string tot_lat_name = parent_model_name + TOTLAT_SUFF;
  Command<int>& c_tot_lat_sset = rdx_.commandSync<int>(
      {"ZADD", tot_lat_name, std::to_string(load_latency + inf_latency),
       model_name});
  if (!c_tot_lat_sset.ok()) { return -1.0; }

  // Add to parent model's accuracy set
  const std::string model_acc_name = parent_model_name + ACCURACY_SUFF;
  Command<int>& c_acc_sset = rdx_.commandSync<int>(
      {"ZADD", model_acc_name, std::to_string(accuracy), model_name});
  if (!c_acc_sset.ok()) { return -1; }

  // Find which grandparent accuracy bin it belongs to
  double min_acc, max_acc;
  int i = -1;
  for (i = 0; i < (num_gpar_bins - 1); ++i) {
    min_acc = gpar_accuracy_bins[i];
    max_acc = gpar_accuracy_bins[i + 1];
    if (accuracy >= min_acc && accuracy <= max_acc) { break; }
  }
  // If i is invalid, it means the accuracy submitted was invalid
  if (i < 0) {
    throw std::runtime_error("Accuracy passed to add_model is invalid");
  }

  std::string bin_num = std::to_string(i);

  const std::string gpar_acc_name =
      gparent_model_name + "-" + bin_num + "-" + GPARACC_SUFF;

  // Now add it to the respective accuracy bin set
  Command<int>& c_gpar_acc_sset = rdx_.commandSync<int>(
      {"ZADD", gpar_acc_name, std::to_string(accuracy), model_name});
  if (!c_gpar_acc_sset.ok()) { return -1; }

  // Save the bin number for removing during model deletion
  const std::string gpar_bin_name = model_name + "-" + GPARACCBIN_SUFF;
  Command<std::string>& c_bin_num =
      rdx_.commandSync<std::string>({"SET", gpar_bin_name, bin_num});
  if (!c_bin_num.ok()) { return -1; }

  // Initialize model load/unload
  if (unset_model_load_unload(model_name) < 0) { return -1; }

  // Check if the parent model contains only PyTorch models
  if (check_pytorch_status(parent_model_name) < 0) { return -1; }

  return 0;
}

std::string RedisMetadata::get_parent_model(const std::string& model_name) {
  const std::string var_par_name = model_name + "-" + PARENT_SUFF;
  Command<std::string>& c_var_par =
      rdx_.commandSync<std::string>({"GET", var_par_name});
  if (!c_var_par.ok()) { return "FAIL"; }
  std::string reply = c_var_par.reply();

  return reply;
}

std::vector<std::string> RedisMetadata::get_all_model_variants(
    const std::string& parent_model_name) {
  const std::string var_par_name = parent_model_name + "-" + MODVAR_SUFF;
  Command<std::vector<std::string>>& c_all_var =
      rdx_.commandSync<std::vector<std::string>>(
          {"ZRANGE", var_par_name, "0", "-1"});
  if (!c_all_var.ok()) { return {}; }

  std::vector<std::string> reply = c_all_var.reply();

  return reply;
}

std::vector<std::string> RedisMetadata::get_all_parent_models(
    const std::string& task, const std::string& dataset) {
  std::vector<std::string> valid_parent_models;
  Command<std::vector<std::string>>& c_parmod_set =
      rdx_.commandSync<std::vector<std::string>>({"SMEMBERS", MODEL_SET});
  if (!c_parmod_set.ok()) { return {}; }

  std::vector<std::string> reply = c_parmod_set.reply();

  // Iterate through all parents and find the ones with matching task and
  // dataset
  for (std::string all_par : reply) {
    std::vector<std::string> amv = get_all_model_variants(all_par);
    if (amv.size() > 0) {
      std::string curr_task = get_model_info(amv[0], "task");
      if (curr_task != task) { continue; }
      std::string curr_dataset = get_model_info(amv[0], "dataset");
      if (curr_dataset != dataset) { continue; }
      // Add to valid_parent_models
      valid_parent_models.push_back(all_par);
    }
  }

  return valid_parent_models;
}

bool RedisMetadata::gparent_model_registered(
    const std::string& gparent_model_name) {
  return gmodel_exists(gparent_model_name);
}

bool RedisMetadata::parent_model_registered(
    const std::string& parent_model_name) {
  return model_exists(parent_model_name);
}

bool RedisMetadata::model_registered(const std::string& model_name) {
  return modelvar_exists(model_name);
}

std::vector<std::string> RedisMetadata::get_all_running_models() {
  Command<std::vector<std::string>>& c_runmod_set =
      rdx_.commandSync<std::vector<std::string>>({"SMEMBERS", RUNMODS_SET});
  if (!c_runmod_set.ok()) {
    return {};  // TODO: return more valid error
  }

  std::vector<std::string> reply = c_runmod_set.reply();

  return reply;
}

double RedisMetadata::get_load_lat(const std::string& model_name) {
  // Check if model variant exists
  if (!modelvar_exists(model_name)) { return -1.0; }

  // Get parent model
  std::string parent_model = get_parent_model(model_name);
  const std::string load_lat_name = parent_model + LOADLAT_SUFF;

  Command<std::string>& c_load_lat_sset =
      rdx_.commandSync<std::string>({"ZSCORE", load_lat_name, model_name});
  if (!c_load_lat_sset.ok()) { return -1.0; }
  std::string reply = c_load_lat_sset.reply();

  return std::stod(reply);
}

double RedisMetadata::get_inf_lat(const std::string& model_name) {
  // Check if model variant exists
  if (!modelvar_exists(model_name)) { return -1.0; }

  // Get parent model
  std::string parent_model = get_parent_model(model_name);
  const std::string inf_lat_name = parent_model + INFLAT_SUFF;

  Command<std::string>& c_inf_lat_sset =
      rdx_.commandSync<std::string>({"ZSCORE", inf_lat_name, model_name});
  if (!c_inf_lat_sset.ok()) { return -1.0; }
  std::string reply = c_inf_lat_sset.reply();

  return std::stod(reply);
}

double RedisMetadata::get_accuracy(const std::string& model_name) {
  // Check if model variant exists
  if (!modelvar_exists(model_name)) { return -1.0; }

  // Get parent model
  std::string parent_model = get_parent_model(model_name);
  const std::string model_acc_name = parent_model + ACCURACY_SUFF;

  Command<std::string>& c_acc_sset =
      rdx_.commandSync<std::string>({"ZSCORE", model_acc_name, model_name});
  if (!c_acc_sset.ok()) { return -1.0; }
  std::string reply = c_acc_sset.reply();

  return std::stod(reply);
}

std::vector<std::string> RedisMetadata::inf_lat_bin(
    const std::string& parent_model_name, const double& min_lat,
    const double& max_lat, const int8_t& max_results) {
  const std::string inf_lat_name = parent_model_name + INFLAT_SUFF;

  Command<std::vector<std::string>>& c_lat_bin =
      rdx_.commandSync<std::vector<std::string>>(
          {"ZRANGEBYSCORE", inf_lat_name, std::to_string(min_lat),
           std::to_string(max_lat), "LIMIT", "0", std::to_string(max_results)});
  if (!c_lat_bin.ok()) { return {}; }

  std::vector<std::string> reply = c_lat_bin.reply();

  return reply;
}

std::vector<std::string> RedisMetadata::tot_lat_bin(
    const std::string& parent_model_name, const double& min_lat,
    const double& max_lat, const int8_t& max_results) {
  const std::string tot_lat_name = parent_model_name + TOTLAT_SUFF;

  Command<std::vector<std::string>>& c_lat_bin =
      rdx_.commandSync<std::vector<std::string>>(
          {"ZRANGEBYSCORE", tot_lat_name, std::to_string(min_lat),
           std::to_string(max_lat), "LIMIT", "0", std::to_string(max_results)});
  if (!c_lat_bin.ok()) { return {}; }

  std::vector<std::string> reply = c_lat_bin.reply();

  return reply;
}

std::vector<std::string> RedisMetadata::par_acc_bin(
    const std::string& parent_model_name, const double& min_acc,
    const int8_t& max_results) {
  const std::string model_acc_name = parent_model_name + ACCURACY_SUFF;

  Command<std::vector<std::string>>& c_acc_bin =
      rdx_.commandSync<std::vector<std::string>>(
          {"ZRANGEBYSCORE", model_acc_name, std::to_string(min_acc), "+inf",
           "LIMIT", "0", std::to_string(max_results)});
  if (!c_acc_bin.ok()) { return {}; }

  std::vector<std::string> reply = c_acc_bin.reply();

  return reply;
}

std::vector<std::string> RedisMetadata::gpar_acc_bin(
    const std::string& grandparent_model_name, const double& min_acc,
    const int8_t& max_results) {
  // Search for the smallest bin to satisfy the minimum accuracy
  double next_min_acc, next_max_acc;
  int i = -1;
  for (i = 0; i < (num_gpar_bins - 1); ++i) {
    next_min_acc = gpar_accuracy_bins[i];
    next_max_acc = gpar_accuracy_bins[i + 1];
    if (min_acc >= next_min_acc && min_acc <= next_max_acc) { break; }
  }
  // If i is invalid, it means the accuracy submitted was invalid. Return empty
  // vector
  if (i < 0) { return {}; }

  // At this point, we know the smallest bin that satisfies the minimum
  // accuracy. Find the first bin that 1) satisfies the minimum accuracy
  // requirement and
  //// 2) contains at least one model. If none exist, return an empty vector
  std::string gpar_acc_name;
  std::vector<std::string> reply;

  while (i < num_gpar_bins) {
    gpar_acc_name =
        grandparent_model_name + "-" + std::to_string(i) + "-" + GPARACC_SUFF;

    Command<std::vector<std::string>>& c_gpar_acc_bin =
        rdx_.commandSync<std::vector<std::string>>(
            {"ZRANGEBYSCORE", gpar_acc_name, std::to_string(min_acc), "+inf",
             "LIMIT", "0", std::to_string(max_results)});
    if (!c_gpar_acc_bin.ok()) { return {}; }

    reply = c_gpar_acc_bin.reply();
    if (reply.size() != 0) { return reply; }

    i++;
  }

  // No model in any bin to satisfy the request. Return empty vector
  return {};
}

int8_t RedisMetadata::add_running_model(const std::string& executor_name,
                                        const std::string& model_name) {
  // Check that model variant exists
  if (!modelvar_exists(model_name)) {
    std::cout << "[Redis Metadata]: " << model_name << " is not registered"
              << std::endl;
    return -1;
  }

  // Get parent model
  std::string parent_model = get_parent_model(model_name);

  // Initialize model-QPS-executor set
  int8_t umq_rc = update_model_qps(executor_name, model_name, 0.0);
  if (umq_rc < 0) { return -1; }

  const std::string model_qps_name = model_name + "-" + MODQPS_SUFF;
  Command<int>& c_modqps_sset =
      rdx_.commandSync<int>({"ZADD", model_qps_name, "0.0", executor_name});
  if (!c_modqps_sset.ok()) { return -1; }

  // Update executor-to-model variant set
  const std::string exec_mvar_name = executor_name + "-" + EXECMVAR_SUFF;
  Command<int>& c_exec_mvar_set =
      rdx_.commandSync<int>({"SADD", exec_mvar_name, model_name});
  if (!c_exec_mvar_set.ok()) { return -1; }

  // Update executor-to-model set
  const std::string exec_mod_name = executor_name + "-" + EXECMOD_SUFF;
  Command<int>& c_exec_mod_set =
      rdx_.commandSync<int>({"SADD", exec_mod_name, parent_model});
  if (!c_exec_mod_set.ok()) { return -1; }

  // Update running model variants
  const std::string running_modvar = model_name + "-" + RUNMVARS_SUFF;
  Command<int>& c_runningmvar_set =
      rdx_.commandSync<int>({"INCR", running_modvar});
  if (!c_runningmvar_set.ok()) { return -1; }

  // Add to all running model variants list
  Command<int>& c_allrunning_set =
      rdx_.commandSync<int>({"SADD", RUNMODS_SET, model_name});
  if (!c_allrunning_set.ok()) { return -1; }

  // Update running model
  const std::string running_mods = parent_model + "-" + RUNMODS_SUFF;
  Command<int>& c_runningmod_set =
      rdx_.commandSync<int>({"INCR", running_mods});
  if (!c_runningmod_set.ok()) { return -1; }

  // Update parent-model variant running set
  const std::string parent_child_name = parent_model + "-" + RUNCHILD_SUFF;
  Command<std::string>& c_parent_child_set = rdx_.commandSync<std::string>(
      {"ZINCRBY", parent_child_name, "1", model_name});
  if (!c_parent_child_set.ok()) { return -1; }

  // Update executor-parent-model variant running set
  const std::string exec_parent_child_name =
      executor_name + "-" + parent_model + "-" + RUNCHIEX_SUFF;
  if (key_exists(exec_parent_child_name)) {
    Command<int>& c_exec_parent_child_set =
        rdx_.commandSync<int>({"INCR", exec_parent_child_name});
    if (!c_exec_parent_child_set.ok()) { return -1; }
  } else {
    Command<std::string>& c_set_exec_par_child_set =
        rdx_.commandSync<std::string>({"SET", exec_parent_child_name, "1"});
    if (!c_set_exec_par_child_set.ok()) { return -1; }
  }

  // Initialize model-AvgLat-executor set
  int8_t uml_rc = update_model_avglat(executor_name, model_name, 0.0);
  if (uml_rc < 0) { return -1; }

  // Initialize/unset blacklist model on worker
  int8_t umab_rc = unset_model_avglat_blacklist(executor_name, model_name);
  if (umab_rc < 0) { return -1; }

  // Initialize/unset scaledown mode on worker
  int8_t ups_rc = unset_parent_scaledown(executor_name, parent_model);
  if (ups_rc < 0) { return -1; }

  return 0;
}

int8_t RedisMetadata::remove_running_model(const std::string& executor_name,
                                           const std::string& model_name) {
  // Check that model variant exists
  if (!modelvar_exists(model_name)) {
    std::cout << "[Redis Metadata]: " << model_name << " is not registered"
              << std::endl;
    return -1;
  }

  // Check that the model is actually running before deleting
  if (!is_model_running(model_name, executor_name)) {
    std::cout << "[Redis Metadata]: " << model_name
              << " is not running, cannot delete " << std::endl;
    return -1;
  }

  // Unset the blacklist before removing.
  if (unset_model_avglat_blacklist(executor_name, model_name) != 0) {
    std::cout << "[Redis Metadata]: failed to unset blacklist for "
              << model_name << std::endl;
    return -1;
  }

  // Remove from executor's running model variant list
  const std::string exec_mvar_name = executor_name + "-" + EXECMVAR_SUFF;
  Command<int>& c_exec_mvar_set =
      rdx_.commandSync<int>({"SREM", exec_mvar_name, model_name});
  if (!c_exec_mvar_set.ok()) { return -1; }

  // Get parent model
  std::string parent_model = get_parent_model(model_name);

  // Decrement the number of running children on the executor
  const std::string exec_parent_child_name =
      executor_name + "-" + parent_model + "-" + RUNCHIEX_SUFF;
  Command<int>& c_exec_parent_child_set =
      rdx_.commandSync<int>({"DECR", exec_parent_child_name});
  if (!c_exec_parent_child_set.ok()) { return -1; }

  // Remove executor's running parent model set and counter if no children are
  // running on it
  int16_t epc_reply = c_exec_parent_child_set.reply();
  if (epc_reply == 0) {
    // Remove executor's running parent model list
    const std::string exec_mod_name = executor_name + "-" + EXECMOD_SUFF;
    Command<int>& c_execmod_set =
        rdx_.commandSync<int>({"SREM", exec_mod_name, parent_model});
    if (!c_execmod_set.ok()) { return -1; }

    // Remove counter
    Command<int>& c_exec_par_mod_count =
        rdx_.commandSync<int>({"DEL", exec_parent_child_name});
    if (!c_exec_par_mod_count.ok()) { return -1; }
  }

  // Remove executor from model's QPS set
  const std::string model_qps_name = model_name + "-" + MODQPS_SUFF;
  Command<int>& c_mod_qps_del =
      rdx_.commandSync<int>({"ZREM", model_qps_name, executor_name});
  if (!c_mod_qps_del.ok()) { return -1; }

  // Remove executor from model's average latency set
  const std::string model_avglat_name = model_name + "-" + MODAVGLAT_SUFF;
  Command<int>& c_mod_avglat_del =
      rdx_.commandSync<int>({"ZREM", model_avglat_name, executor_name});
  if (!c_mod_avglat_del.ok()) { return -1; }

  // Delete model variant-executor avglat key
  const std::string blist_mod_name =
      executor_name + "-" + model_name + "-" + BLISTMOD_SUFF;
  Command<int>& c_blist_mod = rdx_.commandSync<int>({"DEL", blist_mod_name});
  if (!c_blist_mod.ok()) { return -1; }

  // Decrement running models
  const std::string running_mods = parent_model + "-" + RUNMODS_SUFF;
  Command<int>& c_runningmod = rdx_.commandSync<int>({"DECR", running_mods});
  if (!c_runningmod.ok()) { return -1; }

  int16_t runmod_reply = c_runningmod.reply();
  if (runmod_reply == 0) {  // Not running anywhere, delete
    Command<int>& c_del_rmod = rdx_.commandSync<int>({"DEL", running_mods});
    if (!c_del_rmod.ok()) { return -1; }

    Command<int>& c_allrunning_set =
        rdx_.commandSync<int>({"SREM", RUNMODS_SET, model_name});
    if (!c_allrunning_set.ok()) { return -1; }
  }

  // Decrement running model variants
  const std::string running_modvar = model_name + "-" + RUNMVARS_SUFF;
  Command<int>& c_runningmodvar =
      rdx_.commandSync<int>({"DECR", running_modvar});
  if (!c_runningmodvar.ok()) { return -1; }

  int16_t runmodvar_reply = c_runningmodvar.reply();
  if (runmodvar_reply == 0) {  // Not running anywhere, delete
    Command<int>& c_del_rmodvar =
        rdx_.commandSync<int>({"DEL", running_modvar});
    if (!c_del_rmodvar.ok()) { return -1; }
  }

  // Decrement parent-model variant running set
  const std::string parent_child_name = parent_model + "-" + RUNCHILD_SUFF;
  Command<std::string>& c_parent_child_set = rdx_.commandSync<std::string>(
      {"ZINCRBY", parent_child_name, "-1", model_name});
  if (!c_parent_child_set.ok()) { return -1; }

  int16_t pc_reply = std::stoi(c_parent_child_set.reply());
  if (pc_reply == 0) {  // No versions of this model are running.
    Command<int>& c_del_parent_child =
        rdx_.commandSync<int>({"ZREM", parent_child_name, model_name});
    if (!c_del_parent_child.ok()) { return -1; }
  }

  // If it was an exclusive model, reset
  if (is_exec_slack(executor_name) == model_name) {
    if (set_exec_slack(executor_name) != 0) {
      std::cout << "[Redis Metadata]: failed to reset slack for ";
      std::cout << executor_name << std::endl;
      return -1;
    }
  }

  return 0;
}

int8_t RedisMetadata::is_parent_model_running(
    const std::string& parent_model_name, const std::string& executor_name) {
  // Check that model exists
  if (!model_exists(parent_model_name)) {
    std::cout << "[Redis Metadata]: " << parent_model_name
              << " is not registered" << std::endl;
    return -1;
  }

  if (executor_name
          .empty()) {  // If executor name is empty, check all executors set
    const std::string running_parent_name =
        parent_model_name + "-" + RUNCHILD_SUFF;
    if (key_exists(running_parent_name)) {
      return 1;
    } else {
      return 0;
    }
  } else {
    const std::string exec_mod_name = executor_name + "-" + EXECMOD_SUFF;
    if (set_member(exec_mod_name, parent_model_name)) {
      return 1;
    } else {
      return 0;
    }
  }
}

int8_t RedisMetadata::is_model_running(const std::string& model_name,
                                       const std::string& executor_name) {
  // Check that model exists
  if (!modelvar_exists(model_name)) {
    std::cout << "[Redis Metadata]: " << model_name << " is not registered"
              << std::endl;
    return -1;
  }

  if (executor_name
          .empty()) {  // If executor name is empty, check all executors set
    const std::string running_modvar_name = model_name + "-" + RUNMVARS_SUFF;
    if (key_exists(running_modvar_name)) {
      return 1;
    } else {
      return 0;
    }
  } else {
    const std::string exec_mod_name = executor_name + "-" + EXECMVAR_SUFF;
    if (set_member(exec_mod_name, model_name)) {
      return 1;
    } else {
      return 0;
    }
  }
}

int8_t RedisMetadata::is_pytorch_only(const std::string& parent_model_name) {
  // Check that model exists
  if (!model_exists(parent_model_name)) {
    std::cout << "[Redis Metadata]: " << parent_model_name
              << " is not registered" << std::endl;
    return -1;
  }

  // Since this is only set for some parent models, simply check if the key
  // exists
  const std::string pt_parent = parent_model_name + "-" + PTONLY_SUFF;
  if (key_exists(pt_parent)) {
    return 1;
  } else {
    return 0;
  }
}

std::vector<std::string> RedisMetadata::get_parent_models_on_executor(
    const std::string& executor_name) {
  const std::string exec_mod_name = executor_name + "-" + EXECMOD_SUFF;
  Command<std::vector<std::string>>& c_execmod_set =
      rdx_.commandSync<std::vector<std::string>>({"SMEMBERS", exec_mod_name});
  if (!c_execmod_set.ok()) {
    return {};  // TODO: return more valid error
  }

  std::vector<std::string> reply = c_execmod_set.reply();

  return reply;
}

std::vector<std::string> RedisMetadata::get_variants_on_executor(
    const std::string& executor_name) {
  const std::string exec_mvar_name = executor_name + "-" + EXECMVAR_SUFF;
  Command<std::vector<std::string>>& c_execmvar_set =
      rdx_.commandSync<std::vector<std::string>>({"SMEMBERS", exec_mvar_name});
  if (!c_execmvar_set.ok()) {
    return {};  // TODO: return more valid error
  }

  std::vector<std::string> reply = c_execmvar_set.reply();

  return reply;
}

std::vector<std::string> RedisMetadata::get_parents_variants_on_executor(
    const std::string& parent_model_name, const std::string& executor_name) {
  // Check that model exists
  if (!model_exists(parent_model_name)) {
    std::cout << "[Redis Metadata]: " << parent_model_name
              << " is not registered" << std::endl;
    return {};
  }

  // Iterate through all variants and check if they are running. This is
  // necessary since variants can have multiple replicas
  std::vector<std::string> all_var = get_all_model_variants(parent_model_name);
  std::vector<std::string> running_variants;
  for (auto av : all_var) {
    if (is_model_running(av, executor_name)) { running_variants.push_back(av); }
  }

  return running_variants;
}

std::string RedisMetadata::get_model_info(const std::string& model_name,
                                          const std::string& info) {
  // Check that model variant exists
  if (!modelvar_exists(model_name)) {
    std::cout << "[Redis Metadata]: Tried to get info, but " << model_name;
    std::cout << " does not exist" << std::endl;
    return "FAIL";
  }

  // Check that metadata key is valid
  const std::string model_info_name = model_name + "-" + MODINFO_SUFF;
  if (!hash_exists(model_info_name, info)) {
    std::cout << "[Redis Metadata]: " << info
              << " is not a valid metadata entry" << std::endl;
    return "FAIL";
  }

  // Look-up based on metadata name
  Command<std::string>& c_md =
      rdx_.commandSync<std::string>({"HGET", model_info_name, info});
  if (!c_md.ok()) { return "FAIL"; }
  std::string reply = c_md.reply();

  return reply;
}

int8_t RedisMetadata::update_parentmodel_qps(
    const std::string& executor_name,
    const std::vector<std::pair<std::string, double>>& mod_qps) {
  // Keep track of aggregate QPS with a map
  std::map<std::string, int32_t> agg_map;

  // Iterate over mod_qps and aggregate to parent models
  for (auto mq : mod_qps) {
    std::string parent_model = get_parent_model(mq.first);
    if (agg_map.find(parent_model) != agg_map.end()) {
      agg_map[parent_model]++;
    } else {
      agg_map.insert(std::pair<std::string, int32_t>(parent_model, mq.second));
    }
  }

  // Update each parent model's sorted set
  for (auto am : agg_map) {
    const std::string model_qps_name = am.first + "-" + MODQPS_SUFF;
    Command<int>& c_mod_qps_sset = rdx_.commandSync<int>(
        {"ZADD", model_qps_name, std::to_string(am.second), executor_name});
    if (!c_mod_qps_sset.ok()) { return -1; }
  }

  return 0;
}

int8_t RedisMetadata::update_model_qps(const std::string& executor_name,
                                       const std::string& model_name,
                                       const double& qps) {
  // Check that model variant exists
  if (!modelvar_exists(model_name)) {
    std::cout << "[Redis Metadata]: " << model_name << " does not exist"
              << std::endl;
    return -1;
  }

  // Add to sorted set
  const std::string model_qps_name = model_name + "-" + MODQPS_SUFF;
  Command<int>& c_mod_qps_sset = rdx_.commandSync<int>(
      {"ZADD", model_qps_name, std::to_string(qps), executor_name});
  if (!c_mod_qps_sset.ok()) { return -1; }

  return 0;
}

double RedisMetadata::get_model_qps(const std::string& executor_name,
                                    const std::string& model_name) {
  // Check if model variant exists
  if (!modelvar_exists(model_name)) { return -1.0; }

  const std::string model_qps_name = model_name + "-" + MODQPS_SUFF;
  Command<std::string>& c_modqps_sset =
      rdx_.commandSync<std::string>({"ZSCORE", model_qps_name, executor_name});
  if (!c_modqps_sset.ok()) { return -1.0; }
  std::string reply = c_modqps_sset.reply();

  return std::stod(reply);
}

std::vector<std::string> RedisMetadata::min_qps_name(
    const std::string& model_name, const int8_t& max_results) {
  // Check if model variant exists
  if (!modelvar_exists(model_name)) {
    return {};  // TODO: return more valid error
  }

  // Check if model is running.
  if (!is_model_running(model_name)) { return {}; }

  const std::string model_qps_name = model_name + "-" + MODQPS_SUFF;
  // Set stays sorted, so we request the bottom element
  Command<std::vector<std::string>>& c_min_qps =
      rdx_.commandSync<std::vector<std::string>>(
          {"ZRANGEBYSCORE", model_qps_name, "-inf", "+inf", "LIMIT", "0",
           std::to_string(max_results)});
  if (!c_min_qps.ok()) { return {}; }

  std::vector<std::string> reply = c_min_qps.reply();

  return reply;
}

double RedisMetadata::get_min_qps(const std::string& model_name) {
  // Check if model variant exists
  if (!modelvar_exists(model_name)) {
    return -1.0;  // TODO: return more valid error
  }

  // Check if model is running. If not, return empty_addr
  if (!is_model_running(model_name)) { return -1.0; }

  const std::string model_qps_name = model_name + "-" + MODQPS_SUFF;
  // Set stays sorted, so we request the bottom element
  Command<std::vector<std::string>>& c_min_qps =
      rdx_.commandSync<std::vector<std::string>>(
          {"ZRANGEBYSCORE", model_qps_name, "-inf", "+inf", "LIMIT", "0", "1"});
  if (!c_min_qps.ok()) { return -1.0; }

  std::vector<std::string> reply = c_min_qps.reply();

  // Now get model's QPS
  double min_qps = get_model_qps(reply[0], model_name);

  return min_qps;
}

int8_t RedisMetadata::update_model_avglat(const std::string& executor_name,
                                          const std::string& model_name,
                                          const double& avg_lat) {
  // Check that model variant exists
  if (!modelvar_exists(model_name)) {
    std::cout << "[Redis Metadata]: " << model_name << " does not exist"
              << std::endl;
    return -1;
  }

  // Add to sorted set
  const std::string model_avglat_name = model_name + "-" + MODAVGLAT_SUFF;
  Command<int>& c_mod_avglat_sset = rdx_.commandSync<int>(
      {"ZADD", model_avglat_name, std::to_string(avg_lat), executor_name});
  if (!c_mod_avglat_sset.ok()) { return -1; }

  return 0;
}

double RedisMetadata::get_model_avglat(const std::string& executor_name,
                                       const std::string& model_name) {
  // Check if model variant exists
  if (!modelvar_exists(model_name)) { return -1.0; }

  const std::string model_avglat_name = model_name + "-" + MODAVGLAT_SUFF;
  Command<std::string>& c_mod_avglat_sset = rdx_.commandSync<std::string>(
      {"ZSCORE", model_avglat_name, executor_name});
  if (!c_mod_avglat_sset.ok()) { return -1.0; }
  std::string reply = c_mod_avglat_sset.reply();

  return std::stod(reply);
}

int8_t RedisMetadata::set_model_avglat_blacklist(
    const std::string& executor_name, const std::string& model_name) {
  // Ensure model is running on the worker
  if (is_model_running(model_name, executor_name) != 1) {
    std::cout << "[Redis Metadata]: " << model_name << " not running on ";
    std::cout << executor_name << ". Cannot blacklist!" << std::endl;
    return -1;
  }

  const std::string blist_mod_name =
      executor_name + "-" + model_name + "-" + BLISTMOD_SUFF;
  Command<std::string>& c_blist_mod =
      rdx_.commandSync<std::string>({"SET", blist_mod_name, "1"});
  if (!c_blist_mod.ok()) { return -1; }
  return 0;
}

int8_t RedisMetadata::unset_model_avglat_blacklist(
    const std::string& executor_name, const std::string& model_name) {
  // Ensure model is running on the worker
  if (is_model_running(model_name, executor_name) != 1) {
    std::cout << "[Redis Metadata]: " << model_name << " not running on ";
    std::cout << executor_name << ". Cannot un-blacklist!" << std::endl;
    return -1;
  }

  const std::string blist_mod_name =
      executor_name + "-" + model_name + "-" + BLISTMOD_SUFF;
  Command<std::string>& c_blist_mod =
      rdx_.commandSync<std::string>({"SET", blist_mod_name, "0"});
  if (!c_blist_mod.ok()) { return -1; }
  return 0;
}

int8_t RedisMetadata::get_model_avglat_blacklist(
    const std::string& executor_name, const std::string& model_name) {
  // Ensure model is running on the worker
  if (is_model_running(model_name, executor_name) != 1) {
    std::cout << "[Redis Metadata]: " << model_name << " not running on ";
    std::cout << executor_name << ". Cannot check blacklist!" << std::endl;
    return -1;
  }

  const std::string blist_mod_name =
      executor_name + "-" + model_name + "-" + BLISTMOD_SUFF;
  Command<std::string>& c_blist_mod =
      rdx_.commandSync<std::string>({"GET", blist_mod_name});
  if (!c_blist_mod.ok()) { return -1; }

  int8_t reply = std::stoi(c_blist_mod.reply());

  return reply;
}

int8_t RedisMetadata::set_parent_scaledown(
    const std::string& executor_name, const std::string& parent_model_name) {
  // Ensure parent model is running on the worker
  if (is_parent_model_running(parent_model_name, executor_name) != 1) {
    std::cout << "[Redis Metadata]: " << parent_model_name
              << " not running on ";
    std::cout << executor_name << ". Cannot set scaledown!" << std::endl;
    return -1;
  }

  const std::string scaledown_name =
      executor_name + "-" + parent_model_name + "-" + SDOWN_SUFF;
  Command<std::string>& c_sdown_pmod =
      rdx_.commandSync<std::string>({"SET", scaledown_name, "1"});
  if (!c_sdown_pmod.ok()) { return -1; }
  return 0;
}

int8_t RedisMetadata::unset_parent_scaledown(
    const std::string& executor_name, const std::string& parent_model_name) {
  // Ensure parent model is running on the worker
  if (is_parent_model_running(parent_model_name, executor_name) != 1) {
    std::cout << "[Redis Metadata]: " << parent_model_name
              << " not running on ";
    std::cout << executor_name << ". Cannot set scaledown!" << std::endl;
    return -1;
  }

  const std::string scaledown_name =
      executor_name + "-" + parent_model_name + "-" + SDOWN_SUFF;
  Command<std::string>& c_sdown_pmod =
      rdx_.commandSync<std::string>({"SET", scaledown_name, "0"});
  if (!c_sdown_pmod.ok()) { return -1; }
  return 0;
}

int8_t RedisMetadata::get_parent_scaledown(
    const std::string& executor_name, const std::string& parent_model_name) {
  // Ensure parent model is running on the worker
  if (is_parent_model_running(parent_model_name, executor_name) != 1) {
    std::cout << "[Redis Metadata]: " << parent_model_name
              << " not running on ";
    std::cout << executor_name << ". Cannot set scaledown!" << std::endl;
    return -1;
  }

  const std::string scaledown_name =
      executor_name + "-" + parent_model_name + "-" + SDOWN_SUFF;
  Command<std::string>& c_sdown_pmod =
      rdx_.commandSync<std::string>({"GET", scaledown_name});
  if (!c_sdown_pmod.ok()) { return -1; }

  int8_t reply = std::stoi(c_sdown_pmod.reply());

  return reply;
}

int8_t RedisMetadata::set_model_load_unload(const std::string& model_name) {
  // Check if model variant exists
  if (!modelvar_exists(model_name)) { return -1; }

  const std::string load_unl_name = model_name + "-" + LOADUNL_SUFF;
  Command<std::string>& c_loadunl_mod =
      rdx_.commandSync<std::string>({"SET", load_unl_name, "1"});
  if (!c_loadunl_mod.ok()) { return -1; }
  return 0;
}

int8_t RedisMetadata::unset_model_load_unload(const std::string& model_name) {
  // Check if model variant exists
  if (!modelvar_exists(model_name)) { return -1; }

  const std::string load_unl_name = model_name + "-" + LOADUNL_SUFF;
  Command<std::string>& c_loadunl_mod =
      rdx_.commandSync<std::string>({"SET", load_unl_name, "0"});
  if (!c_loadunl_mod.ok()) { return -1; }
  return 0;
}

int8_t RedisMetadata::get_model_load_unload(const std::string& model_name) {
  // Check if model variant exists
  if (!modelvar_exists(model_name)) { return -1; }

  const std::string load_unl_name = model_name + "-" + LOADUNL_SUFF;
  Command<std::string>& c_loadunl_mod =
      rdx_.commandSync<std::string>({"GET", load_unl_name});
  if (!c_loadunl_mod.ok()) { return -1; }

  int8_t reply = std::stoi(c_loadunl_mod.reply());

  return reply;
}

int8_t RedisMetadata::delete_model(const std::string& model_name) {
  // Check that model variant exists
  if (!modelvar_exists(model_name)) {
    std::cout << "[Redis Metadata]: Tried to delete, but " << model_name;
    std::cout << " does not exist" << std::endl;
    return -1;
  }

  // Get parent model
  std::string parent_model = get_parent_model(model_name);

  // Remove model info hash lookup table
  const std::string model_info_name = model_name + "-" + MODINFO_SUFF;
  Command<int>& c_del_modinfo_htable =
      rdx_.commandSync<int>({"DEL", model_info_name});
  if (!c_del_modinfo_htable.ok()) { return -1; }

  // Remove model load/unload key
  const std::string load_unl_name = model_name + "-" + LOADUNL_SUFF;
  Command<int>& c_del_loadunl = rdx_.commandSync<int>({"DEL", load_unl_name});
  if (!c_del_loadunl.ok()) { return -1; }

  // Check if the model was running
  const std::string model_qps_name = model_name + "-" + MODQPS_SUFF;
  Command<std::set<std::string>>& c_exec_qps =
      rdx_.commandSync<std::set<std::string>>(
          {"ZRANGE", model_qps_name, "0", "-1"});
  if (!c_exec_qps.ok()) { return -1; }

  std::set<std::string> reply = c_exec_qps.reply();
  for (auto exec : reply) { remove_running_model(exec, model_name); }

  // Remove model QPS set
  Command<int>& c_mod_qps_del = rdx_.commandSync<int>({"DEL", model_qps_name});
  if (!c_mod_qps_del.ok()) { return -1; }

  // Remove from model set
  Command<int>& c_del_model_set =
      rdx_.commandSync<int>({"SREM", MODELVAR_SET, model_name});
  if (!c_del_model_set.ok()) { return -1; }

  // Remove grandparent model->model variant key
  const std::string var_gpar_name = model_name + "-" + GPARENT_SUFF;
  Command<int>& c_var_gpar = rdx_.commandSync<int>({"DEL", var_gpar_name});
  if (!c_var_gpar.ok()) { return -1; }

  // Remove model->model variant key
  const std::string var_par_name = model_name + "-" + PARENT_SUFF;
  Command<int>& c_var_par = rdx_.commandSync<int>({"DEL", var_par_name});
  if (!c_var_par.ok()) { return -1; }

  // Remove from grandparent accuracy bin
  const std::string gpar_bin_name = model_name + "-" + GPARACCBIN_SUFF;
  Command<int>& c_bin_num = rdx_.commandSync<int>({"DEL", gpar_bin_name});
  if (!c_bin_num.ok()) { return -1; }

  // Remove from parent's list of children
  const std::string par_child_name = parent_model + "-" + MODVAR_SUFF;
  Command<int>& c_var_par_rem =
      rdx_.commandSync<int>({"ZREM", par_child_name, model_name});
  if (!c_var_par_rem.ok()) { return -1; }

  // Check if deleting this model causes the parent to only have PyTorch models
  if (check_pytorch_status(parent_model) < 0) { return -1; }

  return 0;
}

int8_t RedisMetadata::update_cpu_util(const std::string& executor_name,
                                      const double& utilization,
                                      const int8_t& first_time) {
  if (!first_time) {
    // Check if key exists
    if (!key_exists(executor_name)) { return -1; }
  }

  // Add to sorted set
  Command<int>& c_cpu_sset = rdx_.commandSync<int>(
      {"ZADD", CPUUTIL_SET, std::to_string(utilization), executor_name});
  if (!c_cpu_sset.ok()) { return -1; }

  return 0;
}

int8_t RedisMetadata::update_gpu_util(const std::string& executor_name,
                                      const double& utilization,
                                      const int8_t& first_time) {
  if (!first_time) {
    // Check if key exists
    if (!key_exists(executor_name)) { return -1; }
  }

  // Add to sorted set
  Command<int>& c_gpu_sset = rdx_.commandSync<int>(
      {"ZADD", GPUUTIL_SET, std::to_string(utilization), executor_name});
  if (!c_gpu_sset.ok()) { return -1; }

  return 0;
}

int8_t RedisMetadata::update_inferentia_util(const std::string& executor_name,
                                             const double& utilization,
                                             const int8_t& first_time) {
  if (!first_time) {
    // Check if key exists
    if (!key_exists(executor_name)) { return -1; }
  }

  // Add to sorted set
  Command<int>& c_inferentia_sset = rdx_.commandSync<int>(
      {"ZADD", INFERENTIAUTIL_SET, std::to_string(utilization), executor_name});
  if (!c_inferentia_sset.ok()) { return -1; }

  return 0;
}

double RedisMetadata::get_cpu_util(const std::string& executor_name) {
  // Check if key exists
  if (!key_exists(executor_name)) { return -1.0; }

  Command<std::string>& c_cpu_util =
      rdx_.commandSync<std::string>({"ZSCORE", CPUUTIL_SET, executor_name});
  if (!c_cpu_util.ok()) { return -1.0; }
  std::string reply = c_cpu_util.reply();

  return std::stod(reply);
}

double RedisMetadata::get_gpu_util(const std::string& executor_name) {
  // Check if key exists
  if (!key_exists(executor_name)) { return -1.0; }

  Command<std::string>& c_gpu_util =
      rdx_.commandSync<std::string>({"ZSCORE", GPUUTIL_SET, executor_name});
  if (!c_gpu_util.ok()) { return -1.0; }
  std::string reply = c_gpu_util.reply();

  return std::stod(reply);
}

double RedisMetadata::get_inferentia_util(const std::string& executor_name) {
  // Check if key exists
  if (!key_exists(executor_name)) { return -1.0; }

  Command<std::string>& c_inferentia_util =
      rdx_.commandSync<std::string>({"ZSCORE", INFERENTIAUTIL_SET, executor_name});
  if (!c_inferentia_util.ok()) { return -1.0; }
  std::string reply = c_inferentia_util.reply();

  return std::stod(reply);
}

std::vector<std::string> RedisMetadata::max_cpu_util_name(
    const double& max_thresh, const int8_t& max_results) {
  // Set stays sorted, so we request the top element
  Command<std::vector<std::string>>& c_cpu_util =
      rdx_.commandSync<std::vector<std::string>>(
          {"ZREVRANGEBYSCORE", CPUUTIL_SET, std::to_string(max_thresh), "-inf",
           "LIMIT", "0", std::to_string(max_results)});
  if (!c_cpu_util.ok()) { return {}; }

  std::vector<std::string> reply = c_cpu_util.reply();

  return reply;
}

std::vector<std::string> RedisMetadata::min_cpu_util_name(
    const int8_t& max_results) {
  // Set stays sorted, so we request the bottom element
  Command<std::vector<std::string>>& c_cpu_util =
      rdx_.commandSync<std::vector<std::string>>({"ZRANGEBYSCORE", CPUUTIL_SET,
                                                  "-inf", "+inf", "LIMIT", "0",
                                                  std::to_string(max_results)});
  if (!c_cpu_util.ok()) { return {}; }

  std::vector<std::string> reply = c_cpu_util.reply();

  return reply;
}

double RedisMetadata::get_min_cpu_util() {
  // Set stays sorted, so we request the bottom element
  Command<std::vector<std::string>>& c_cpu_util =
      rdx_.commandSync<std::vector<std::string>>(
          {"ZRANGEBYSCORE", CPUUTIL_SET, "-inf", "+inf", "LIMIT", "0", "1"});
  if (!c_cpu_util.ok()) { return -1.0; }

  std::vector<std::string> reply = c_cpu_util.reply();

  // Now get executor's utilization
  double min_util = get_cpu_util(reply[0]);

  return min_util;
}

std::vector<std::string> RedisMetadata::max_gpu_util_name(
    const double& max_thresh, const int8_t& max_results) {
  // Set stays sorted, so we request the top element
  Command<std::vector<std::string>>& c_gpu_util =
      rdx_.commandSync<std::vector<std::string>>(
          {"ZREVRANGEBYSCORE", GPUUTIL_SET, std::to_string(max_thresh), "-inf",
           "LIMIT", "0", std::to_string(max_results)});
  if (!c_gpu_util.ok()) { return {}; }

  std::vector<std::string> reply = c_gpu_util.reply();

  return reply;
}

std::vector<std::string> RedisMetadata::min_gpu_util_name(
    const int8_t& max_results) {
  // Set stays sorted, so we request the bottom element
  Command<std::vector<std::string>>& c_gpu_util =
      rdx_.commandSync<std::vector<std::string>>({"ZRANGEBYSCORE", GPUUTIL_SET,
                                                  "-inf", "+inf", "LIMIT", "0",
                                                  std::to_string(max_results)});
  if (!c_gpu_util.ok()) { return {}; }

  std::vector<std::string> reply = c_gpu_util.reply();

  return reply;
}

double RedisMetadata::get_min_gpu_util() {
  // Set stays sorted, so we request the bottom element
  Command<std::vector<std::string>>& c_gpu_util =
      rdx_.commandSync<std::vector<std::string>>(
          {"ZRANGEBYSCORE", GPUUTIL_SET, "-inf", "+inf", "LIMIT", "0", "1"});
  if (!c_gpu_util.ok()) { return -1.0; }

  std::vector<std::string> reply = c_gpu_util.reply();

  // Now get executor's utilization
  double min_util = get_gpu_util(reply[0]);

  return min_util;
}

std::vector<std::string> RedisMetadata::max_inferentia_util_name(
    const double& max_thresh, const int8_t& max_results) {
  // Set stays sorted, so we request the top element
  Command<std::vector<std::string>>& c_inferentia_util =
      rdx_.commandSync<std::vector<std::string>>(
          {"ZREVRANGEBYSCORE", INFERENTIAUTIL_SET, std::to_string(max_thresh), "-inf",
           "LIMIT", "0", std::to_string(max_results)});
  if (!c_inferentia_util.ok()) { return {}; }

  std::vector<std::string> reply = c_inferentia_util.reply();

  return reply;
}

std::vector<std::string> RedisMetadata::min_inferentia_util_name(
    const int8_t& max_results) {
  // Set stays sorted, so we request the bottom element
  Command<std::vector<std::string>>& c_inferentia_util =
      rdx_.commandSync<std::vector<std::string>>({"ZRANGEBYSCORE", INFERENTIAUTIL_SET,
                                                  "-inf", "+inf", "LIMIT", "0",
                                                  std::to_string(max_results)});
  if (!c_inferentia_util.ok()) { return {}; }

  std::vector<std::string> reply = c_inferentia_util.reply();

  return reply;
}

double RedisMetadata::get_min_inferentia_util() {
  // Set stays sorted, so we request the bottom element
  Command<std::vector<std::string>>& c_inferentia_util =
      rdx_.commandSync<std::vector<std::string>>(
          {"ZRANGEBYSCORE", INFERENTIAUTIL_SET, "-inf", "+inf", "LIMIT", "0", "1"});
  if (!c_inferentia_util.ok()) { return -1.0; }

  std::vector<std::string> reply = c_inferentia_util.reply();

  // Now get executor's utilization
  double min_util = get_inferentia_util(reply[0]);

  return min_util;
}

bool RedisMetadata::is_empty_address(const struct Address& addr) {
  return ((addr.ip == empty_addr.ip) && (addr.port == empty_addr.port));
}

const std::string RedisMetadata::Address_to_str(const struct Address& addr) {
  return addr.ip + ":" + addr.port;
}

/*********************** Private Functions ***********************/

bool RedisMetadata::key_exists(const std::string& key) {
  Command<int>& c_exists = rdx_.commandSync<int>({"EXISTS", key});
  int reply = c_exists.reply();
  if (reply == 1) {
    return true;
  } else {
    return false;
  }
}

bool RedisMetadata::set_member(const std::string& key,
                               const std::string& field) {
  Command<int>& c_exists = rdx_.commandSync<int>({"SISMEMBER", key, field});
  int reply = c_exists.reply();
  if (reply == 1) {
    return true;
  } else {
    return false;
  }
}

bool RedisMetadata::gmodel_exists(const std::string& model) {
  return set_member(GMOD_SET, model);
}

bool RedisMetadata::model_exists(const std::string& model) {
  return set_member(MODEL_SET, model);
}

bool RedisMetadata::modelvar_exists(const std::string& model) {
  return set_member(MODELVAR_SET, model);
}

bool RedisMetadata::hash_exists(const std::string& key,
                                const std::string& field) {
  Command<int>& c_exists = rdx_.commandSync<int>({"HEXISTS", key, field});
  int reply = c_exists.reply();
  if (reply == 1) {
    return true;
  } else {
    return false;
  }
}

int8_t RedisMetadata::check_pytorch_status(const std::string& model) {
  // Walk through all variants and check if they are all PyTorch.
  // If so, set the flag. If not, delete it
  std::vector<std::string> all_var = get_all_model_variants(model);

  bool all_pt = true;
  for (std::string av : all_var) {
    if (get_model_info(av, "framework") != "pytorch") {
      all_pt = false;
      break;
    }
  }

  const std::string pt_parent = model + "-" + PTONLY_SUFF;
  if (all_pt) {
    Command<std::string>& c_ptonly_set =
        rdx_.commandSync<std::string>({"SET", pt_parent, "1"});
    if (!c_ptonly_set.ok()) { return -1; }
  } else {
    Command<int>& c_ptonly_del = rdx_.commandSync<int>({"DEL", pt_parent});
    if (!c_ptonly_del.ok()) { return -1; }
  }

  return 0;
}
