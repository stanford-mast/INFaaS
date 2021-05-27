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

#include <unistd.h>  // sleep()
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "redis_metadata.h"

#define FAIL(x) printf("[FAIL]: " #x "\n")
#define PASS(x) printf("[PASS]: " #x "\n")

struct test_mod_variant_info {
  std::string model_name;
  double comp_size;
  double acc;
  std::string dataset;
  std::string submitter;
  std::string framework;
  double load_lat;
  double inf_lat;
  int16_t batch;
  double peak_memory;
  std::string task;
  int16_t img_dim;
  double slope;
  double intercept;
};

static int8_t num_execs = 5;
static const std::string sample_exec[] = {"iw-0", "iw-1", "iw-2", "iw-3", "iw-4"};
static const struct Address sample_addr[] = {{"123.1.0.1", "4374"},
                                             {"123.4.5.6", "9876"},
                                             {"123.9.8.7", "6767"},
                                             {"123.65.64.63", "0123"},
                                             {"123.7.65.32", "2468"}};
static const double sample_util[] = {50.1, 48.3, 74.7, 60.4, 70.1};
static const double sample_qps[] = {200.3, 101.6, 500.4, 300.9, 220.8};
static const std::string parent_model = "mymodel";
static const std::string model_task = "classification";
static const std::string dataset = "imagenet";
static const int16_t img_dim = 299;
static const std::string gparent_model =
    model_task + "imagenet" + std::to_string(img_dim);
static struct test_mod_variant_info test_mod_variant = {
    "mymodel_trt", 100, 71.2, dataset,    "me",    "tensorrt", 1000,
    500,           4,   2000, model_task, img_dim, 100.1,      50.2};
static struct test_mod_variant_info test_pt_mod_variant = {
    "mymodel_pt", 10, 30.6, dataset,    "me",    "pytorch", 2000,
    2000,         1,  3000, model_task, img_dim, 20.1,      25.7};
static struct test_mod_variant_info test_inferentia_mod_variant = {
    "mymodel_inf", 1000, 10.6, dataset,    "me",    "inferentia", 3000,
    1000,         1,  5000, model_task, img_dim, 50.1,      10.8};

// IMPORTANT: need to launch a redis-server on port 6379 (default) to run this
// test
int main(int argc, char** argv) {
  int64_t total_us = 0;
  RedisMetadata rmd({"localhost", "6379"});
  int8_t rc = -1;
  int8_t is_running = -1;
  std::string res_str;
  std::chrono::microseconds duration;
  std::chrono::high_resolution_clock::time_point start, stop;

  // Test adding executors
  for (int i = 0; i < num_execs; ++i) {
    if (i == 0) {
      start = std::chrono::high_resolution_clock::now();
      rc = rmd.add_executor_addr(sample_exec[i], sample_addr[i]);
      stop = std::chrono::high_resolution_clock::now();
      duration =
          std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      std::cout << "Time taken by add_executor_addr: " << duration.count()
                << " microseconds" << std::endl;
      total_us += duration.count();
    } else {
      rc = rmd.add_executor_addr(sample_exec[i], sample_addr[i]);
    }
    if (rc) {
      FAIL("Add executors and addresses");
      return 1;
    }
  }
  PASS("Add executors and addresses");

  // Test retrieving the number of executors
  start = std::chrono::high_resolution_clock::now();
  int16_t number_of_executors = rmd.get_num_executors();
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time taken by get_num_executors: " << duration.count()
            << " microseconds" << std::endl;
  total_us += duration.count();
  if (number_of_executors == num_execs) {
    PASS("Number of executors");
  } else {
    FAIL("Number of executors");
    return 1;
  }

  // Test retrieving number of Inferentia executors
  int16_t number_of_inferentia_executors = rmd.get_num_inferentia_executors();
  if (number_of_inferentia_executors == 0) {
    PASS("Number of Inferentia executors (not yet set)");
  } else {
    FAIL("Number of Inferentia executors (not yet set)");
    return 1;
  }

  // Test retrieving number of CPU executors
  int16_t number_of_cpu_executors = rmd.get_num_cpu_executors();
  if (number_of_cpu_executors == 0) {
    PASS("Number of CPU executors (not yet set)");
  } else {
    FAIL("Number of CPU executors (not yet set)");
    return 1;
  }

  // Test getting a list of all the executors
  std::vector<std::string> all_exec = rmd.get_all_executors();
  if (all_exec.size() > 0) {
    for (const std::string ae : all_exec) {
      bool found = false;
      for (int i = 0; i < num_execs; ++i) {
        if (sample_exec[i] == ae) {
          found = true;
          break;
        }
      }
      if (!found) {
        FAIL("List of all executors");
        return 1;
      }
    }
    PASS("List of all executors");
  } else {
    FAIL("List of all executors");
    return 1;
  }

  // Test trying to get an executor that does not exist
  const struct Address bad_addr = rmd.get_executor_addr("BadKey");
  if (RedisMetadata::is_empty_address(bad_addr)) {
    PASS("BadKey");
  } else {
    FAIL("BadKey");
    return 1;
  }

  // Test getting an executor that does exist
  const struct Address good_addr = rmd.get_executor_addr(sample_exec[0]);
  if (good_addr.ip == sample_addr[0].ip &&
      good_addr.port == sample_addr[0].port) {
    PASS("Executor address retrieval");
  } else {
    FAIL("Executor address retrieval (got " + good_addr.ip + ":" +
         good_addr.port + ")");
    return 1;
  }

  // Test blacklisting an executor
  rc = rmd.blacklist_executor(sample_exec[0], 3);
  if (!rc) {
    PASS("Blacklist executor");
  } else {
    FAIL("Blacklist executor");
    return 1;
  }

  // Check if executor is still blacklisted (it should be)
  bool is_blisted = rmd.is_blacklisted(sample_exec[0]);
  if (is_blisted) {
    PASS("Still blacklisted");
  } else {
    PASS("Still blacklisted");
    return 1;
  }

  // Wait 4 seconds, then try again. Executor should no longer be blacklisted
  sleep(4);
  is_blisted = rmd.is_blacklisted(sample_exec[0]);
  if (!is_blisted) {
    PASS("No longer blacklisted");
  } else {
    FAIL("No longer blacklisted");
    return 1;
  }

  // Check if executor is slack before setting it
  std::string not_slack = rmd.is_exec_slack(sample_exec[0]);
  if (not_slack == "NS") {
    PASS("Not slack executor");
  } else {
    FAIL("Not slack executor");
    return 1;
  }

  // Test setting an executor as slack with no running variant
  rc = rmd.set_exec_slack(sample_exec[0]);
  if (!rc) {
    PASS("Slack executor, no running variant");
  } else {
    FAIL("Slack executor, no running variant");
    return 1;
  }

  // Check if executor is slack, should return 0 indicating no running variant
  std::string empty_variant = rmd.is_exec_slack(sample_exec[0]);
  if (empty_variant == "0") {
    PASS("Slack executor check, no running variant");
  } else {
    FAIL("Slack executor check, no running variant");
    return 1;
  }

  // Set executor to be slack with a running variant.
  // Then check to see that it is slack with "testmodel" running on it
  rc = rmd.set_exec_slack(sample_exec[0], "testmodel");
  std::string actual_variant = rmd.is_exec_slack(sample_exec[0]);
  if (actual_variant == "testmodel") {
    PASS("Slack executor check, with running variant");
  } else {
    FAIL("Slack executor check, with running variant");
    return 1;
  }

  // Test getting min utilization when no utilizations have explicitly been
  // added
  std::vector<std::string> min_gpu_noadd = rmd.min_gpu_util_name(num_execs);
  if (min_gpu_noadd.size() == num_execs) {
    PASS("Minimum utilization no add");
  } else {
    FAIL("Minimum utilization no add");
    return 1;
  }

  // Test adding parent model
  rc = rmd.add_parent_model(parent_model);
  if (!rc) {
    PASS("Add a parent model");
  } else {
    FAIL("Add a parent model");
    return 1;
  }

  // Test adding a PyTorch variant
  rc =
      rmd.add_model(test_pt_mod_variant.model_name, parent_model, gparent_model,
                    test_pt_mod_variant.comp_size, test_pt_mod_variant.acc,
                    test_pt_mod_variant.dataset, test_pt_mod_variant.submitter,
                    test_pt_mod_variant.framework, test_pt_mod_variant.task,
                    test_pt_mod_variant.img_dim, test_pt_mod_variant.batch,
                    test_pt_mod_variant.load_lat, test_pt_mod_variant.inf_lat,
                    test_pt_mod_variant.peak_memory, test_pt_mod_variant.slope,
                    test_pt_mod_variant.intercept);
  if (!rc) {
    PASS("Add a PyTorch model");
  } else {
    FAIL("Add a PyTorch model");
    return 1;
  }

  // Check if the parent only has PyTorch variants
  int8_t only_pt = rmd.is_pytorch_only(parent_model);
  if (only_pt) {
    PASS("Parent is only PyTorch");
  } else {
    FAIL("Parent is only PyTorch");
    return 1;
  }

  // Test adding a non-PyTorch model variant
  rc = rmd.add_model(test_mod_variant.model_name, parent_model, gparent_model,
                     test_mod_variant.comp_size, test_mod_variant.acc,
                     test_mod_variant.dataset, test_mod_variant.submitter,
                     test_mod_variant.framework, test_mod_variant.task,
                     test_mod_variant.img_dim, test_mod_variant.batch,
                     test_mod_variant.load_lat, test_mod_variant.inf_lat,
                     test_mod_variant.peak_memory, test_mod_variant.slope,
                     test_mod_variant.intercept);
  if (!rc) {
    PASS("Add a TRT model");
  } else {
    FAIL("Add a TRT model");
    return 1;
  }

  // Try to add the same model again. Should fail
  rc = rmd.add_model(test_mod_variant.model_name, parent_model, gparent_model,
                     test_mod_variant.comp_size, test_mod_variant.acc,
                     test_mod_variant.dataset, test_mod_variant.submitter,
                     test_mod_variant.framework, test_mod_variant.task,
                     test_mod_variant.img_dim, test_mod_variant.batch,
                     test_mod_variant.load_lat, test_mod_variant.inf_lat,
                     test_mod_variant.peak_memory, test_mod_variant.slope,
                     test_mod_variant.intercept);
  if (rc) {
    PASS("Model not added");
  } else {
    FAIL("Model not added");
    return 1;
  }

  // Test checking if parent is PyTorch-only (it should not be).
  only_pt = rmd.is_pytorch_only(parent_model);
  if (!only_pt) {
    PASS("Parent is not only PyTorch");
  } else {
    FAIL("Parent is not only PyTorch");
    return 1;
  }

  // Test getting model variant's parent
  std::string parent_result = rmd.get_parent_model(test_mod_variant.model_name);
  if (parent_result == parent_model) {
    PASS("Parent model");
  } else {
    FAIL("Parent model");
    return 1;
  }

  // Test the stored info is correct.
  res_str = rmd.get_model_info(test_mod_variant.model_name, "dataset");
  if (res_str == test_mod_variant.dataset) {
    PASS("Model dataset stored");
  } else {
    FAIL("Model dataset stored");
    return 1;
  }

  res_str = rmd.get_model_info(test_mod_variant.model_name, "framework");
  if (res_str == test_mod_variant.framework) {
    PASS("Model framework stored");
  } else {
    FAIL("Model framework stored");
    return 1;
  }

  res_str = rmd.get_model_info(test_mod_variant.model_name, "img_dim");
  if (std::stoi(res_str) == test_mod_variant.img_dim) {
    PASS("Model dimensions stored");
  } else {
    FAIL("Model dimensions stored");
    return 1;
  }

  res_str = rmd.get_model_info(test_mod_variant.model_name, "slope");
  if (std::stod(res_str) == test_mod_variant.slope) {
    PASS("Model slope stored");
  } else {
    FAIL("Model slope stored");
    return 1;
  }

  start = std::chrono::high_resolution_clock::now();
  res_str = rmd.get_model_info(test_mod_variant.model_name, "task");
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by get_model_info: " << duration.count()
            << " microseconds" << std::endl;
  total_us += duration.count();
  if (res_str == test_mod_variant.task) {
    PASS("Model task stored");
  } else {
    FAIL("Model task stored");
    return 1;
  }

  // Test getting a model's accuracy
  double model_md = rmd.get_accuracy(test_mod_variant.model_name);
  if (model_md == test_mod_variant.acc) {
    PASS("Accuracy");
  } else {
    FAIL("Accuracy (got " + model_md + "; expected " + test_mod_variant.acc +
         ")");
    return 1;
  }

  // Test getting a model's loading latency
  double retrieve_load_lat = rmd.get_load_lat(test_mod_variant.model_name);
  if (retrieve_load_lat == test_mod_variant.load_lat) {
    PASS("Retrieve loading latency");
  } else {
    FAIL("Retrieve loading latency");
    return 1;
  }

  // Test getting a model's inference latency
  start = std::chrono::high_resolution_clock::now();
  double retrieve_inf_lat = rmd.get_inf_lat(test_mod_variant.model_name);
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by get_inf_lat: " << duration.count()
            << " microseconds" << std::endl;
  total_us += duration.count();
  if (retrieve_inf_lat == test_mod_variant.inf_lat) {
    PASS("Retrieve inference latency");
  } else {
    FAIL("Retrieve inference latency");
    return 1;
  }

  // Test retrieving models with inference latency constraints. Should return
  // empty vector
  start = std::chrono::high_resolution_clock::now();
  std::vector<std::string> inf_lat_pool =
      rmd.inf_lat_bin(parent_model, 1.0, test_mod_variant.inf_lat - 50.0);
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by inf_lat_bin: " << duration.count()
            << " microseconds" << std::endl;
  if (inf_lat_pool.empty()) {
    PASS("Inference latency bucket");
  } else {
    FAIL("Inference latency bucket");
    return 1;
  }

  // Test retrieving models with total latency constraints. Should return
  // test_mod_variant
  double tot_lat = test_mod_variant.load_lat + test_mod_variant.inf_lat + 1.0;
  std::vector<std::string> tot_lat_pool =
      rmd.tot_lat_bin(parent_model, 1.0, tot_lat);
  if (tot_lat_pool.size() > 0 &&
      tot_lat_pool[0] == test_mod_variant.model_name) {
    PASS("Total latency bucket");
  } else {
    FAIL("Total latency bucket");
    return 1;
  }

  // Test retrieving models with accuracy constraints. Should return empty
  // vector
  std::vector<std::string> acc_pool =
      rmd.par_acc_bin(parent_model, test_mod_variant.acc + 1.0);
  if (acc_pool.empty()) {
    PASS("Accuracy bucket");
  } else {
    FAIL("Accuracy bucket");
    return 1;
  }

  // Test retrieving grandparent model variant with accuracy constraints.
  // Accuracy constraint will fall in the same bin as originally registered.
  // Should return test_mod_variant
  std::vector<std::string> gpar_acc_pool =
      rmd.gpar_acc_bin(gparent_model, test_mod_variant.acc);
  if (gpar_acc_pool.size() > 0 &&
      gpar_acc_pool[0] == test_mod_variant.model_name) {
    PASS("Grandparent accuracy bucket, same bucket");
  } else {
    FAIL("Grandparent accuracy bucket, same bucket");
    return 1;
  }

  // Now make the accuracy constraint much smaller, which forces it to look
  // through a couple
  //// of other bins first.
  // Should still return test_mod_variant
  start = std::chrono::high_resolution_clock::now();
  gpar_acc_pool = rmd.gpar_acc_bin(gparent_model, 42.1);
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by gpar_acc_bin: " << duration.count()
            << " microseconds" << std::endl;
  total_us += duration.count();
  if (gpar_acc_pool.size() > 0 &&
      gpar_acc_pool[0] == test_mod_variant.model_name) {
    PASS("Grandparent accuracy bucket, different bucket");
  } else {
    FAIL("Grandparent accuracy bucket, different bucket");
    return 1;
  }

  // Test setting an executor to CPU only
  rc = rmd.set_exec_onlycpu(sample_exec[3]);
  if (!rc) {
    PASS("Set executor to CPU only");
  } else {
    FAIL("Set executor to CPU only");
    return 1;
  }

  // Now check if it is a CPU-only executor
  int8_t is_onlycpu = rmd.is_exec_onlycpu(sample_exec[3]);
  if (is_onlycpu) {
    PASS("CPU-only worker check");
  } else {
    FAIL("CPU-only worker check");
    return 1;
  }

  // Check that the CPU-only executor has a GPU and Inferentia utilization of 101
  double cpuonly_gpuutil = rmd.get_gpu_util(sample_exec[3]);
  double cpuonly_inferentiautil = rmd.get_inferentia_util(sample_exec[3]);
  if (cpuonly_gpuutil == 101.0 && cpuonly_inferentiautil == 101.0) {
    PASS("CPU-only worker utilization");
  } else {
    FAIL("CPU-only worker utilization");
    return 1;
  }

  // Now check the number of CPU executors
  int16_t number_of_cpu_executors_afterset = rmd.get_num_cpu_executors();
  if (number_of_cpu_executors_afterset == 1) {
    PASS("Number of CPU executors (after set)");
  } else {
    FAIL("Number of CPU executors (after set)");
    return 1;
  }

  // Test setting an executor to support Inferentia
  rc = rmd.set_exec_inferentia(sample_exec[4]);
  if (!rc) {
    PASS("Set executor to support Inferentia");
  } else {
    FAIL("Set executor to support Inferentia");
    return 1;
  }

  // Now check if it supports Inferentia
  int8_t is_inferentia = rmd.is_exec_inferentia(sample_exec[4]);
  if (is_inferentia) {
    PASS("Inferentia worker check");
  } else {
    FAIL("Inferentia worker check");
    return 1;
  }

  // Check that the inferentia worker has a GPU utilization of 101
  double inferentia_gpuutil = rmd.get_gpu_util(sample_exec[4]);
  if (inferentia_gpuutil== 101.0) {
    PASS("Inferentia worker utilization");
  } else {
    FAIL("Inferentia worker utilization");
    return 1;
  }

  // Now check the number of Inferentia executors
  int16_t number_of_inferentia_executors_afterset = rmd.get_num_inferentia_executors();
  if (number_of_inferentia_executors_afterset == 1) {
    PASS("Number of Inferentia executors (after set)");
  } else {
    FAIL("Number of Inferentia executors (after set)");
    return 1;
  }

  // Test updating the GPU utilization
  for (int i = 0; i < num_execs - 2; ++i) {
    if (i == 0) {
      start = std::chrono::high_resolution_clock::now();
      rc = rmd.update_gpu_util(sample_exec[i], sample_util[i]);
      stop = std::chrono::high_resolution_clock::now();
      duration =
          std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      std::cout << "Time taken by update_gpu_util: " << duration.count()
                << " microseconds" << std::endl;
      total_us += duration.count();
    } else {
      rc = rmd.update_gpu_util(sample_exec[i], sample_util[i]);
    }
    if (rc) {
      FAIL("Update utilization (" + sample_exec[i] + ")");
      return 1;
    }
  }
  PASS("Update utilization");

  // Test getting the GPU utilization
  double util = rmd.get_gpu_util(sample_exec[0]);
  if (util == sample_util[0]) {
    PASS("Retrieve utilization");
  } else {
    FAIL("Retrieve utilization (got " + util + "; expected " + sample_util[0] +
         ")");
    return 1;
  }

  // Test getting min utilization
  std::vector<std::string> min_name = rmd.min_gpu_util_name();
  if (min_name[0] == sample_exec[1] && min_name[1] == sample_exec[0] &&
      min_name[2] == sample_exec[2]) {
    PASS("Minimum utilization");
  } else {
    FAIL("Minimum utilization");
    return 1;
  }

  // Test getting max utilization with multiple thresholds
  start = std::chrono::high_resolution_clock::now();
  std::vector<std::string> max_name_nothresh = rmd.max_gpu_util_name(100, 1);
  const struct Address max_addr_nothresh =
      rmd.get_executor_addr(max_name_nothresh[0]);
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by max_gpu_util_name: " << duration.count()
            << " microseconds" << std::endl;
  total_us += duration.count();
  if (max_addr_nothresh.ip == sample_addr[2].ip &&
      max_addr_nothresh.port == sample_addr[2].port) {
    PASS("Maximum utilization (no threshold)");
  } else {
    FAIL("Maximum utilization (no threshold)");
    return 1;
  }

  std::vector<std::string> max_name_thresh = rmd.max_gpu_util_name(70, 1);
  if (max_name_thresh[0] == sample_exec[0]) {
    PASS("Maximum utilization (with threshold)");
  } else {
    FAIL("Maximum utilization (with threshold)");
    return 1;
  }

  // Test getting min overall utilization
  double min_overall_util = rmd.get_min_gpu_util();
  if (min_overall_util == sample_util[1]) {
    PASS("Minimum overall utilization");
  } else {
    FAIL("Minimum overall utilization");
    return 1;
  }

  // Test seeing if model is running
  is_running = rmd.is_model_running(test_mod_variant.model_name);
  if (!is_running) {
    PASS("Model variant running");
  } else {
    FAIL("Model variant running");
    return 1;
  }

  // Test seeing if parent model is running
  is_running = rmd.is_parent_model_running(parent_model);
  if (!is_running) {
    PASS("Parent model running");
  } else {
    FAIL("Parent model running");
    return 1;
  }

  // Test adding running model
  start = std::chrono::high_resolution_clock::now();
  rc = rmd.add_running_model(sample_exec[0], test_mod_variant.model_name);
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time taken by add_running_model: " << duration.count()
            << " microseconds" << std::endl;
  total_us += duration.count();
  if (!rc) {
    PASS("Add a running model");
  } else {
    FAIL("Add a running model");
    return 1;
  }

  // Test if model is running on a particular executor
  start = std::chrono::high_resolution_clock::now();
  int8_t running_check =
      rmd.is_model_running(test_mod_variant.model_name, sample_exec[0]);
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by is_model_running: " << duration.count()
            << " microseconds" << std::endl;
  total_us += duration.count();
  if (running_check) {
    PASS("Is running");
  } else {
    FAIL("Is running");
    return 1;
  }

  // Test blacklisting a model
  rc = rmd.set_model_avglat_blacklist(sample_exec[0],
                                      test_mod_variant.model_name);
  if (!rc) {
    PASS("Blacklist a model");
  } else {
    FAIL("Blacklist a model");
    return 1;
  }

  // Check if the model is blacklisted (it should be)
  int8_t is_mod_blisted = rmd.get_model_avglat_blacklist(
      sample_exec[0], test_mod_variant.model_name);
  if (is_mod_blisted == 1) {
    PASS("Model is blacklisted");
  } else {
    FAIL("Model is blacklisted");
    return 1;
  }

  // Un-blacklist a model and check if it is blacklisted
  rmd.unset_model_avglat_blacklist(sample_exec[0], test_mod_variant.model_name);
  is_mod_blisted = rmd.get_model_avglat_blacklist(sample_exec[0],
                                                  test_mod_variant.model_name);
  if (!is_mod_blisted) {
    PASS("Model is not blacklisted");
  } else {
    FAIL("Model is not blacklisted");
    return 1;
  }

  // Test setting parent scaling
  rc = rmd.set_parent_scaledown(sample_exec[0], parent_model);
  if (!rc) {
    PASS("Parent model scaledown set");
  } else {
    FAIL("Parent model scaledown set");
    return 1;
  }

  // Check if the model is in scaledown mode (it should be)
  int8_t is_scaledown_mode =
      rmd.get_parent_scaledown(sample_exec[0], parent_model);
  if (is_scaledown_mode == 1) {
    PASS("Parent model is in scaledown mode");
  } else {
    FAIL("Parent model is in scaledown mode");
    return 1;
  }

  // Turn off scaledown mode and check if it is still blacklisted
  rmd.unset_parent_scaledown(sample_exec[0], parent_model);
  is_scaledown_mode = rmd.get_parent_scaledown(sample_exec[0], parent_model);
  if (!is_scaledown_mode) {
    PASS("Parent model not in scaledown mode");
  } else {
    FAIL("Parent model not in scaledown mode");
    return 1;
  }

  // Test adding model QPS
  rc = rmd.update_model_qps(sample_exec[0], test_mod_variant.model_name,
                            sample_qps[0]);
  if (!rc) {
    PASS("Add QPS");
  } else {
    FAIL("Add QPS");
    return 1;
  }

  // Test getting model QPS
  double mod_qps =
      rmd.get_model_qps(sample_exec[0], test_mod_variant.model_name);
  if (mod_qps == sample_qps[0]) {
    PASS("Retrieve QPS");
  } else {
    FAIL("Retrieve QPS (got " + mod_qps + "; expected " + sample_qps[0] + ")");
    return 1;
  }

  // Test getting minimum QPS
  rmd.add_running_model(sample_exec[1], test_mod_variant.model_name);
  rmd.update_model_qps(sample_exec[1], test_mod_variant.model_name,
                       sample_qps[1]);
  std::vector<std::string> min_qps_name =
      rmd.min_qps_name(test_mod_variant.model_name, 1);
  const struct Address min_qps_addr = rmd.get_executor_addr(min_qps_name[0]);
  if (min_qps_addr.ip == sample_addr[1].ip &&
      min_qps_addr.port == sample_addr[1].port) {
    PASS("Minimum QPS");
  } else {
    FAIL("Minimum QPS");
    return 1;
  }

  // Get all parent models running on sample_exec[1]
  std::vector<std::string> running_par_mods =
      rmd.get_parent_models_on_executor(sample_exec[1]);
  if (running_par_mods.size() > 0 && running_par_mods[0] == parent_model) {
    PASS("All parent models on executor");
  } else {
    FAIL("All parent models on executor");
    return 1;
  }

  // Get all model variants running on sample_exec[1]
  std::vector<std::string> running_modvars =
      rmd.get_variants_on_executor(sample_exec[1]);
  if (running_modvars.size() > 0 &&
      running_modvars[0] == test_mod_variant.model_name) {
    PASS("All model variants on executor");
  } else {
    FAIL("All model variants on executor");
    return 1;
  }

  // Get all running models based on parent name
  start = std::chrono::high_resolution_clock::now();
  std::vector<std::string> running_child =
      rmd.get_parents_variants_on_executor(parent_model, sample_exec[1]);
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by get_parents_variants_on_executor: "
            << duration.count() << " microseconds" << std::endl;
  total_us += duration.count();
  if (running_child.size() > 0 &&
      running_child[0] == test_mod_variant.model_name) {
    PASS("Running child on executor from parent");
  } else {
    FAIL("Running child on executor from parent");
    return 1;
  }

  // Remove model by calling remove_running_model explicitly
  rc = rmd.remove_running_model(sample_exec[1], test_mod_variant.model_name);
  if (!rc) {
    PASS("Remove a running model");
  } else {
    FAIL("Remove a running model");
    return 1;
  }

  // Model should not be running on sample_exec[1]
  rc = rmd.is_model_running(test_mod_variant.model_name, sample_exec[1]);
  if (!rc) {
    PASS("Model not running on specific executor");
  } else {
    FAIL("Model not running on specific executor");
    return 1;
  }

  // Now check if parent model is running
  is_running = rmd.is_parent_model_running(parent_model, sample_exec[2]);
  if (!is_running) {
    PASS("Removed from running");
  } else {
    FAIL("Removed from running");
    return 1;
  }

  // Check if parent model has any children running on sample_exec[1]; it
  // shouldn't
  running_child =
      rmd.get_parents_variants_on_executor(parent_model, sample_exec[1]);
  if (running_child.size() == 0) {
    PASS("No running child on executor from parent");
  } else {
    FAIL("No running child on executor from parent");
    return 1;
  }

  // Add running model and remove executor. Model should no longer be running
  rmd.add_running_model(sample_exec[2], test_mod_variant.model_name);
  rc = rmd.delete_executor(sample_exec[2]);
  num_execs--;
  if (!rc) {
    PASS("Remove executor");
  } else {
    FAIL("Remove executor");
    return 1;
  }

  // Test retrieving all models on a deleted executor
  start = std::chrono::high_resolution_clock::now();
  std::vector<std::string> all_running =
      rmd.get_variants_on_executor(sample_exec[2]);
  stop = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by get_variants_on_executor: " << duration.count()
            << " microseconds" << std::endl;
  total_us += duration.count();
  if (all_running.size() == 0) {
    PASS("All running models on executor");
  } else {
    FAIL("All running models on executor");
    return 1;
  }

  // Test retrieving the number of executors after removing one.
  number_of_executors = rmd.get_num_executors();
  if (number_of_executors == num_execs) {
    PASS("Number of executors after deletion");
  } else {
    FAIL("Number of executors after deletion");
    return 1;
  }

  // Parent model should still be running
  is_running = rmd.is_parent_model_running(parent_model);
  if (is_running) {
    PASS("Parent still running");
  } else {
    FAIL("Parent still running");
    return 1;
  }

  // Delete model from the original is_model_running test.
  // Model should then not be running at all.
  rmd.remove_running_model(sample_exec[0], test_mod_variant.model_name);
  is_running = rmd.is_model_running(test_mod_variant.model_name);
  if (!is_running) {
    PASS("Not running on deleted executor");
  } else {
    FAIL("Not running on deleted executor");
    return 1;
  }

  is_running = rmd.is_parent_model_running(parent_model);
  if (!is_running) {
    PASS("Parent not running after deleted executor");
  } else {
    FAIL("Parent not running after deleted executor");
    return 1;
  }

  // Try to set scaledown for parent that is not running. Should fail.
  rc = rmd.set_parent_scaledown(sample_exec[0], parent_model);
  if (rc) {
    PASS("Parent model scaledown properly not set");
  } else {
    FAIL("Parent model scaledown properly not set");
    return 1;
  }

  // Test deleting a model
  rc = rmd.delete_model(test_mod_variant.model_name);
  if (!rc) {
    PASS("Delete TRT model");
  } else {
    FAIL("Delete TRT model");
    return 1;
  }

  // Test checking if parent is PyTorch-only after deleting the TRT model
  only_pt = rmd.is_pytorch_only(parent_model);
  if (only_pt) {
    PASS("Parent is again only PyTorch");
  } else {
    FAIL("Parent is again only PyTorch");
    return 1;
  }

  // Test adding Inferentia variant
  rc =
      rmd.add_model(test_inferentia_mod_variant.model_name, parent_model, gparent_model,
                    test_inferentia_mod_variant.comp_size, test_inferentia_mod_variant.acc,
                    test_inferentia_mod_variant.dataset, test_inferentia_mod_variant.submitter,
                    test_inferentia_mod_variant.framework, test_inferentia_mod_variant.task,
                    test_inferentia_mod_variant.img_dim, test_inferentia_mod_variant.batch,
                    test_inferentia_mod_variant.load_lat, test_inferentia_mod_variant.inf_lat,
                    test_inferentia_mod_variant.peak_memory, test_inferentia_mod_variant.slope,
                    test_inferentia_mod_variant.intercept);
  if (!rc) {
    PASS("Add an Inferentia model");
  } else {
    FAIL("Add an Inferentia model");
    return 1;
  }

  // Test setting an inferentia model as running.
  // Also check that it is seen as running and that the parent is seen as running
  rmd.add_running_model(sample_exec[4], test_inferentia_mod_variant.model_name);
  int8_t is_inferentia_running = rmd.is_model_running(test_inferentia_mod_variant.model_name);
  if (is_inferentia_running) {
    PASS("Inferentia model running");
  } else {
    FAIL("Inferentia model running");
    return 1;
  }
  int8_t is_inferentia_parent_running = rmd.is_parent_model_running(parent_model);
  if (is_inferentia_parent_running) {
    PASS("Inferentia parent still running");
  } else {
    FAIL("Inferentia parent still running");
    return 1;
  }

  // Remove running inferentia model
  rc = rmd.remove_running_model(sample_exec[4], test_inferentia_mod_variant.model_name);
  if (!rc) {
    PASS("Remove running Inferentia model");
  } else {
    FAIL("Remove running Inferentia model");
    return 1;
  }

  // Delete inferentia executor and check how many inferentia executors are left
  // Also check that the number of CPU executors is still 1
  rc = rmd.delete_executor(sample_exec[4]);
  num_execs--;
  int8_t number_of_inferentia_executors_afterdelete = rmd.get_num_inferentia_executors();
  if (number_of_inferentia_executors_afterdelete == 0) {
    PASS("Number of Inferentia executors (after delete)");
  } else {
    FAIL("Number of Inferentia executors (after delete)");
    return 1;
  }
  int8_t number_of_cpu_executors_afterdelete = rmd.get_num_cpu_executors();
  if (number_of_cpu_executors_afterdelete == 1) {
    PASS("Number of CPU executors (after deleting Inferentia executor)");
  } else {
    FAIL("Number of CPU executors (after deleting Inferentia executor)");
    return 1;
  }

  std::cout << "All tests passed!!" << std::endl;
  std::cout << "Average time to complete a transaction: " << total_us / 12.0;
  std::cout << " microseconds" << std::endl;
  return 0;
}
