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

#include <ctype.h>
#include <unistd.h>
#include <array>
#include <chrono>  // time_since_epoch
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <aws/core/Aws.h>
#include <aws/ec2/EC2Client.h>
#include <aws/ec2/model/DescribeInstancesRequest.h>
#include <aws/ec2/model/DescribeInstancesResponse.h>
#include <aws/ec2/model/StopInstancesRequest.h>
#include <aws/ec2/model/StopInstancesResponse.h>
#include <aws/ec2/model/TerminateInstancesRequest.h>

#include "include/constants.h"
#include "metadata-store/redis_metadata.h"
#include "worker/query_client.h"

/*
The master's VM scaling daemon calls out to the shell to start a VM because
1) we want to be consistent in the way we start VMs between the start_infaas
script and the daemon and 2) starting a VM also involves checking its state
and making ssh calls with commands attached, which is already done in the
start_vm script.
*/

enum INSTANCETYPE { CPU = 0, GPU = 1, INFERENTIA = 2 };

// Constants
static const int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;

const int32_t sleep_seconds = 2e6;          // 1e6us = 1sec
const int32_t respond_sleep_seconds = 1e6;  // 1e6us = 1sec
const int16_t num_iter = 15;
const double cpu_shutdown_min = 5.0;
const double gpu_shutdown_min = 8.0;
const double avg_cpu_shutdown_min = 30.0;
const double avg_gpu_shutdown_min = 8.0;
const double avg_inferentia_shutdown_min = 25.0;
const double max_blacklist = 80.0;

int16_t vm_backoff_counter;
int16_t vm_shutdown_thresh = 15;

std::map<INSTANCETYPE, int16_t> inst_type_map;
std::map<INSTANCETYPE, int16_t> inst_type_min;

std::string stop_script = "scripts/stop_worker.sh";
std::string start_vm_script = "scripts/start_vm.sh";

int main(int argc, char** argv) {
  if (argc < 24) {
    std::cout << "Usage: ./master_vm_daemon <redis_ip> <redis_port> ";
    std::cout << "<cpugpu-util-thresh> <inferentia-util-thresh> ";
    std::cout << "<zone> <key-name> <worker-image> ";
    std::cout << "<machine-type-gpu> <machine-type-cpu> ";
    std::cout << "<machine-type-inferentia> <startup-script> ";
    std::cout << "<security-group> <max-try> <iam-role> <exec-port> ";
    std::cout << "<exec-prefix> <min-workers> <max-cpu-workers> ";
    std::cout << "<max-gpu-workers> <max-inferentia-workers> ";
    std::cout << "<master-ip> <worker-autoscaler> <delete-machines>";
    std::cout << std::endl;
    return 1;
  }

  const struct Address redis_addr = {argv[1], argv[2]};
  if (RedisMetadata::is_empty_address(redis_addr)) {
    std::cout << "Invalid redis server address: "
              << RedisMetadata::Address_to_str(redis_addr) << std::endl;
    return 1;
  }

  // Set variables
  const double cpugpu_util_thresh = std::stod(argv[3]);
  const double inferentia_util_thresh = std::stod(argv[4]);
  const std::string zone = argv[5];
  const std::string key_name = argv[6];
  const std::string worker_image = argv[7];
  const std::string machine_type_gpu = argv[8];
  const std::string machine_type_cpu = argv[9];
  const std::string machine_type_inferentia = argv[10];
  const std::string startup_script = argv[11];
  const std::string security_group = argv[12];
  const std::string max_try = argv[13];
  const std::string iam_role = argv[14];
  const std::string exec_port = argv[15];
  const std::string exec_prefix = argv[16];
  const int16_t min_workers = std::stoi(argv[17]);
  const int16_t max_cpu_workers = std::stoi(argv[18]);
  const int16_t max_gpu_workers = std::stoi(argv[19]);
  const int16_t max_inferentia_workers = std::stoi(argv[20]);
  const std::string master_ip = argv[21];
  const std::string worker_autoscaler = argv[22];
  const int8_t delete_machines = std::stoi(argv[23]);

  if (delete_machines == 2) {
    std::cout << "[LOG]: Scaled down machines will be persisted, ";
    std::cout << "but removed from INFaaS's view" << std::endl;
  } else if (delete_machines == 1) {
    std::cout << "[LOG]: Scaled down machines will be deleted" << std::endl;
  } else {
    std::cout << "[LOG]: Scaled down machines will be stopped" << std::endl;
  }

  std::cout << "[LOG]: Max number of CPU workers: " << max_cpu_workers;
  std::cout << std::endl;
  std::cout << "[LOG]: Max number of GPU workers: " << max_gpu_workers;
  std::cout << std::endl;
  std::cout << "[LOG]: Max number of Inferentia workers: " << max_inferentia_workers;
  std::cout << std::endl;

  RedisMetadata rm_({redis_addr.ip, redis_addr.port});

  // Unset VM scale flag
  if (rm_.unset_vm_scale() < 0) {
    std::cerr << "Error resetting VM scale flag!" << std::endl;
    throw std::runtime_error("Error resetting VM scale flag");
  }

  // Unset slack scale flag
  if (rm_.unset_slack_scale() < 0) {
    std::cerr << "Error resetting slack scale flag!" << std::endl;
    throw std::runtime_error("Error resetting slack scale flag");
  }

  // Go through initial workers and set them as: CPU, GPU, or Inferentia
  // Also create a second map that never changes (for scaledown)
  inst_type_map.insert(std::pair<INSTANCETYPE, int16_t>(CPU, 0));
  inst_type_map.insert(std::pair<INSTANCETYPE, int16_t>(GPU, 0));
  inst_type_map.insert(std::pair<INSTANCETYPE, int16_t>(INFERENTIA, 0));

  inst_type_min.insert(std::pair<INSTANCETYPE, int16_t>(CPU, 0));
  inst_type_min.insert(std::pair<INSTANCETYPE, int16_t>(GPU, 0));
  inst_type_min.insert(std::pair<INSTANCETYPE, int16_t>(INFERENTIA, 0));

  std::vector<std::string> all_workers = rm_.get_all_executors();
  int check_min = 0;
  for (std::string aw : all_workers) {
    if (rm_.is_exec_onlycpu(aw)) {
      inst_type_map[CPU]++;
      inst_type_min[CPU]++;
    } else if (rm_.is_exec_inferentia(aw)) {
      inst_type_map[INFERENTIA]++;
      inst_type_min[INFERENTIA]++;
    } else {
      inst_type_map[GPU]++;
      inst_type_min[GPU]++;
    }
    check_min++;
  }

  if (check_min != min_workers) {
    throw std::runtime_error("Minimum workers at start is wrong");
  }

  std::cout << "[LOG]: Starting with " << inst_type_map[CPU];
  std::cout << " CPU-only workers, " << inst_type_map[INFERENTIA];
  std::cout << " Inferentia/CPU workers, and " << inst_type_map[GPU];
  std::cout << " GPU/CPU workers" << std::endl;

  int16_t vm_shutdown_counter = 0;
  vm_backoff_counter = num_iter;

  // Set up AWS SDK
  Aws::SDKOptions options;
  Aws::InitAPI(options);
  Aws::Client::ClientConfiguration clientConfig;
  clientConfig.region = Aws::String(region.c_str());
  Aws::EC2::EC2Client ec2(clientConfig);

  while (1) {
    std::chrono::microseconds us_epoch =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch());
    std::cout << "[LOG]: Time since epoch (us): " << us_epoch.count()
              << std::endl;

    // Print for logging
    std::cout << "[LOG]: CPU/GPU utilization threshold is " << cpugpu_util_thresh;
    std::cout << "; Inferentia utilization threshold is " << inferentia_util_thresh << std::endl;

    // Print number of executors:
    int16_t num_exec = rm_.get_num_executors();
    int16_t num_cpu_exec = rm_.get_num_cpu_executors();
    int16_t num_inferentia_exec = rm_.get_num_inferentia_executors();
    int16_t num_gpu_exec = num_exec - num_cpu_exec - num_inferentia_exec;
    std::cout << "[LOG]: There are " << num_exec << " executors => ";
    std::cout << num_cpu_exec << " CPU; ";
    std::cout << num_inferentia_exec << " Inferentia; ";
    std::cout << num_gpu_exec << " GPU" << std::endl;

    // Print per-worker utilization for logging
    // -- CPU --
    std::vector<std::string> all_min_cpu = rm_.min_cpu_util_name(num_exec);

    double avg_cpu_util = 0.0;
    for (std::string min_cpu : all_min_cpu) {
      double curr_cpu_util = rm_.get_cpu_util(min_cpu);
      // If the utilization is too high, blacklist the executor
      if (curr_cpu_util > max_blacklist) {
        int rc = rm_.blacklist_executor(min_cpu, 2);
        if (rc < 0) {
          std::cerr << "Failed to blacklist " << min_cpu << std::endl;
          throw std::runtime_error("Failure to blacklist model");
        }
      }
      bool is_blisted = rm_.is_blacklisted(min_cpu);
      std::cout << "[LOG]: CPU => " << min_cpu << " has utilization ";
      std::cout << curr_cpu_util
                << "; Blacklist status: " << (int16_t)is_blisted << std::endl;
      // Note: this is a little hacky but does not affect the overall behavior
      // of the scaler.
      if (curr_cpu_util < 0.0) {
        rm_.delete_executor(min_cpu);
      } else {
        avg_cpu_util += curr_cpu_util;
      }
    }

    avg_cpu_util /= all_min_cpu.size();

    // -- GPU --
    std::vector<std::string> all_min_gpu = rm_.min_gpu_util_name(num_exec);

    double avg_gpu_util = 0.0;
    double avg_gpu_counter = 0.0;
    for (std::string min_gpu : all_min_gpu) {
      double curr_gpu_util = rm_.get_gpu_util(min_gpu);
      // If the utilization is too high, blacklist the executor
      if ((curr_gpu_util < 100.0) && (curr_gpu_util > max_blacklist)) {
        int rc = rm_.blacklist_executor(min_gpu, 2);
        if (rc < 0) {
          std::cerr << "Failed to blacklist " << min_gpu << std::endl;
          throw std::runtime_error("Failure to blacklist model");
        }
      }
      bool is_blisted = rm_.is_blacklisted(min_gpu);
      std::cout << "[LOG]: GPU => " << min_gpu << " has utilization ";
      std::cout << curr_gpu_util
                << "; Blacklist status: " << (int16_t)is_blisted << std::endl;
      // Note: this is a little hacky but does not affect the overall behavior
      // of the scaler.
      if (curr_gpu_util < 0.0) {
        rm_.delete_executor(min_gpu);
      } else if (curr_gpu_util < 100.0) {
        avg_gpu_util += curr_gpu_util;
        avg_gpu_counter++;
      }
    }

    if (avg_gpu_counter > 0) { avg_gpu_util /= avg_gpu_counter; }

    // -- Inferentia --
    std::vector<std::string> all_min_inferentia = rm_.min_inferentia_util_name(num_exec);

    double avg_inferentia_util = 0.0;
    double avg_inferentia_counter = 0.0;
    for (std::string min_inferentia : all_min_inferentia) {
      double curr_inferentia_util = rm_.get_inferentia_util(min_inferentia);
      // Currently, inferentia workers don't get blacklisted due to their utilization

      bool is_blisted = rm_.is_blacklisted(min_inferentia);
      std::cout << "[LOG]: Inferentia => " << min_inferentia << " has utilization ";
      std::cout << curr_inferentia_util
                << "; Blacklist status: " << (int16_t)is_blisted << std::endl;
      // Note: this is a little hacky but does not affect the overall behavior
      // of the scaler.
      if (curr_inferentia_util < 0.0) {
        rm_.delete_executor(min_inferentia);
      } else if (curr_inferentia_util < 100.0) {
        avg_inferentia_util += curr_inferentia_util;
        avg_inferentia_counter++;
      }
    }

    if (avg_inferentia_counter > 0) { avg_inferentia_util /= avg_inferentia_counter; }

    std::cout << "[LOG]: Avg CPU Util: " << avg_cpu_util;
    std::cout << "; Avg Inferentia Util: " << avg_inferentia_util;
    std::cout << "; Avg GPU Util: " << avg_gpu_util << std::endl;

    // Avoid oscillating (either starting too many instances or killing them too
    // fast)
    if (vm_backoff_counter == num_iter) {
      // Get minimum GPU utilization
      double gpu_util = rm_.get_min_gpu_util();

      // Get minimum CPU utilization
      double cpu_util = rm_.get_min_cpu_util();

      // Get minimum Inferentia utilization
      double inferentia_util = rm_.get_min_inferentia_util();

      // Check VM scale flag
      int8_t vm_scale_flag = rm_.vm_scale_status();

      // Check slack scale flag
      int8_t slack_scale_flag = rm_.slack_scale_status();

      std::cout << "[LOG]: Min overall GPU Util: " << gpu_util;
      std::cout << "; Min overall CPU Util: " << cpu_util;
      std::cout << "; Min overall Inferentia Util: " << inferentia_util;
      std::cout << "; VM Scale Flag: " << (int16_t)vm_scale_flag;
      std::cout << "; Slack Scale Flag: ";
      std::cout << (int16_t)slack_scale_flag << std::endl;

      // Check if machines need to be started
      if ((gpu_util > cpugpu_util_thresh && gpu_util < 100.0 &&
           (num_gpu_exec < max_gpu_workers)) ||
          (cpu_util > cpugpu_util_thresh && (num_cpu_exec < max_cpu_workers)) ||
          (inferentia_util > inferentia_util_thresh &&
           (num_inferentia_exec < max_inferentia_workers)) ||
          (vm_scale_flag == 1 && (num_gpu_exec < max_gpu_workers)) ||
          (slack_scale_flag == 1 && (num_gpu_exec < max_gpu_workers))) {
        std::cout << "[LOG]: Scaling triggered" << std::endl;

        std::string next_worker;
        std::string next_machine_type;
        bool is_cpuinstance = false;
        bool is_inferentiainstance = false;
        if (cpu_util > cpugpu_util_thresh) {
          next_machine_type = machine_type_cpu;
          next_worker =
              exec_prefix + "-cpu-" + std::to_string(inst_type_map[CPU]);
          inst_type_map[CPU]++;
          is_cpuinstance = true;
          std::cout << "[LOG]: Adding a new CPU instance: " << next_worker
                    << std::endl;
        } else if (inferentia_util > inferentia_util_thresh) {
          next_machine_type = machine_type_inferentia;
          next_worker =
              exec_prefix + "-inf-" + std::to_string(inst_type_map[INFERENTIA]);
          inst_type_map[INFERENTIA]++;
          is_inferentiainstance = true;
          std::cout << "[LOG]: Adding a new Inferentia instance: " << next_worker
                    << std::endl;
        } else {
          next_machine_type = machine_type_gpu;
          next_worker =
              exec_prefix + "-gpu-" + std::to_string(inst_type_map[GPU]);
          inst_type_map[GPU]++;
          std::cout << "[LOG]: Adding a new GPU instance: " << next_worker
                    << std::endl;
        }

        std::string cmd = start_vm_script + " " + region + " " + zone + " " +
                          key_name + " " + next_worker + " " + worker_image +
                          " " + next_machine_type + " " + startup_script + " " +
                          security_group + " " + max_try + " " + iam_role +
                          " " + master_ip + " " + worker_autoscaler + " " +
                          infaas_bucket;

        std::array<char, 128> buffer;
        std::string result;
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                      pclose);
        if (!pipe) { std::cout << "popen failed" << std::endl; }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
          result += buffer.data();
        }

        // Parse IP address out of the returned string
        std::string exec_ip = "";
        int8_t num_period = 0;
        for (int i = (result.size() - 1); i >= 0; --i) {
          // Skip newline at the very end
          if (i == (result.size() - 1) && !std::isdigit(result[i])) {
            continue;
          }

          // Break if we stop seeing numbers after the third period
          if (num_period == 3 && !std::isdigit(result[i])) { break; }
          std::string next_string(1, result[i]);
          if (next_string == ".") { num_period++; }

          exec_ip = next_string + exec_ip;
        }

        std::cout << "[LOG]: Exec IP: " << exec_ip << std::endl;

        if (exec_ip == "FAIL") {
          std::cout << "Worker failed to start" << std::endl;
          return 1;
        }

        // Wait until the worker is ready before sending requests
        grpc::ChannelArguments arguments;
        arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
        arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);

        infaas::internal::QueryClient query_client(grpc::CreateCustomChannel(
            RedisMetadata::Address_to_str({exec_ip, exec_port}),
            grpc::InsecureChannelCredentials(), arguments));
        auto worker_reply = query_client.Heartbeat();
        while (worker_reply.status() !=
               infaas::internal::InfaasRequestStatusEnum::SUCCESS) {
          std::cout << "[LOG]: Waiting for " << next_worker << " to respond..."
                    << std::endl;
          worker_reply = query_client.Heartbeat();
          usleep(respond_sleep_seconds);
        }
        std::cout << "[LOG]: Heartbeat received from " << next_worker
                  << std::endl;

        // Add worker's IP to the metadata store
        std::cout << "[LOG]: " << next_worker << " has IP: " << exec_ip
                  << std::endl;
        int8_t rc = rm_.add_executor_addr(next_worker, {exec_ip, exec_port});
        if (rc == -1) {
          std::cerr << "Failure to add " << next_worker << " to metadata store!"
                    << std::endl;
          throw std::runtime_error("Failure to add worker to metadata store");
        }

        // Mark as CPU only instance if applicable
        if (is_cpuinstance) {
          if (rm_.set_exec_onlycpu(next_worker) == -1) {
            std::cerr << "Failure to set " << next_worker << " as CPU only!";
            std::cerr << std::endl;
            throw std::runtime_error("Failure to set worker as CPU only");
          }
        }

        // Mark as Inferentia instance if applicable
        if (is_inferentiainstance) {
          if (rm_.set_exec_inferentia(next_worker) == -1) {
            std::cerr << "Failure to set " << next_worker << " as supporting Inferentia!";
            std::cerr << std::endl;
            throw std::runtime_error("Failure to set worker as supporting Inferentia");
          }
        }

        // If slack scale flag was set, set slack flag
        if (slack_scale_flag) {
          if (rm_.set_exec_slack(next_worker) == -1) {
            std::cerr << "Failure to set " << next_worker << " as slack!";
            std::cerr << std::endl;
            throw std::runtime_error("Failure to set worker slack");
          }
        }

        // Get worker's instance-id and add it to the metadata store
        Aws::EC2::Model::DescribeInstancesRequest request;
        auto outcome = ec2.DescribeInstances(request);
        bool done = false;
        // For now, it looks through all instances, find the instance with a
        // matching name, and records its instance-id. Could eventually use a
        // filter and a query pattern
        if (outcome.IsSuccess()) {
          const auto& reservations = outcome.GetResult().GetReservations();
          for (const auto& reservation : reservations) {
            if (done) { break; }
            const auto& instances = reservation.GetInstances();
            for (const auto& instance : instances) {
              Aws::String name = "Unknown";
              Aws::String worker_name(next_worker.c_str(), next_worker.size());

              const auto& tags = instance.GetTags();
              auto nameIter = std::find_if(tags.cbegin(), tags.cend(),
                                           [](const Aws::EC2::Model::Tag& tag) {
                                             return tag.GetKey() == "Name";
                                           });

              if (nameIter != tags.cend()) { name = nameIter->GetValue(); }
              if (name == worker_name) {
                std::cout << "[LOG]: " << worker_name
                          << " found from describe-instances";
                std::cout << std::endl;
                // Get instance-id
                Aws::String inst_id = instance.GetInstanceId();
                std::string inst_id_str(inst_id.c_str(), inst_id.size());

                int8_t rc_i = rm_.add_executor_instid(next_worker, inst_id_str);
                if (rc_i == -1) {
                  std::cerr << "Failure to add " << next_worker
                            << " instance-id to metadata store!" << std::endl;
                  throw std::runtime_error(
                      "Failure to add worker's instid to metadata store");
                }

                done = true;
                break;
              }
            }
          }
        } else {
          std::cerr << "Failed to describe instances" << std::endl;
          throw std::runtime_error("Describe instances failure");
        }

        // Reset vm_backoff_counter
        vm_backoff_counter = 0;
      }

      if (!vm_scale_flag &&
          (((gpu_util <= gpu_shutdown_min) && (cpu_util <= cpu_shutdown_min)) ||
           ((avg_gpu_util <= avg_gpu_shutdown_min) &&
            (avg_cpu_util <= avg_cpu_shutdown_min)) ||
           ((avg_inferentia_util <= avg_inferentia_shutdown_min) &&
            (avg_cpu_util <= avg_cpu_shutdown_min)))) {
        // Give a machine vm_shutdown_thresh chances before killing it
        vm_shutdown_counter++;
        if (vm_shutdown_counter == vm_shutdown_thresh) {
          // Check that the min is the same for both CPU and GPU
          std::vector<std::string> min_cpu_name = rm_.min_cpu_util_name(1);
          std::vector<std::string> min_gpu_name = rm_.min_gpu_util_name(3);

          // Also check with Inferentia workers
          std::vector<std::string> min_inferentia_name = rm_.min_inferentia_util_name(3);

          bool gpu_check = false;
          auto it =
              find(min_gpu_name.begin(), min_gpu_name.end(), min_cpu_name[0]);
          if (it != min_gpu_name.end()) { gpu_check = true; }
          bool gpu_scaledown = gpu_check & (inst_type_map[GPU] > inst_type_min[GPU]);

          bool inferentia_check = false;
          auto it_inf =
              find(min_inferentia_name.begin(), min_inferentia_name.end(), min_cpu_name[0]);
          if (it_inf != min_inferentia_name.end()) { inferentia_check = true; }
          bool inferentia_scaledown = inferentia_check & 
            (inst_type_map[INFERENTIA] > inst_type_min[INFERENTIA]);

          // Make sure the first workers are not deleted
          if ((gpu_scaledown || inferentia_scaledown) && (num_exec > min_workers)) {
            std::string victim_worker = "";
            if (gpu_scaledown) {
              victim_worker = min_gpu_name[0];
            } else {
              victim_worker = min_inferentia_name[0];
            }
            std::cout << "[LOG]: Shutdown triggered, victim worker: ";
            std::cout << victim_worker << std::endl;

            // First get the worker's instance-id
            std::string inst_id = rm_.get_executor_instid(victim_worker);
            if (inst_id == "FAIL") {
              std::cerr << "Failed to get instance-id of " << victim_worker;
              std::cerr << std::endl;
              throw std::runtime_error("Failed to get instance-id");
            }

            // Get the worker's IP: useful if the machine will be persisted.
            std::string del_exec_ip =
                (rm_.get_executor_addr(victim_worker)).ip;

            // Decrement the appropriate machine's counter
            if (rm_.is_exec_onlycpu(victim_worker)) {
              inst_type_map[CPU]--;
              std::cout << "[LOG]: Now there's " << inst_type_map[CPU]
                        << " CPU-only workers";
              std::cout << std::endl;
            } else if (rm_.is_exec_inferentia(victim_worker)) {
              inst_type_map[INFERENTIA]--;
              std::cout << "[LOG]: Now there's " << inst_type_map[INFERENTIA]
                        << " CPU/Inferentia workers";
              std::cout << std::endl;
            } else {
              inst_type_map[GPU]--;
              std::cout << "[LOG]: Now there's " << inst_type_map[GPU]
                        << " CPU/GPU workers";
              std::cout << std::endl;
            }

            // Remove the worker from the metadata store
            int8_t rc = rm_.delete_executor(victim_worker);
            if (rc == -1) {
              std::cerr << "Failure to delete " << victim_worker;
              std::cerr << " from metadata store!" << std::endl;
              throw std::runtime_error(
                  "Failure to delete worker from metadata store");
            }

            // Persist/kill/stop the machine
            if (delete_machines == 2) {
              // Send stop worker command
              std::string stop_cmd =
                  stop_script + " " + del_exec_ip + " " + key_name;
              std::cout << "[LOG]: Calling " << stop_cmd << " to stop worker"
                        << std::endl;
              if (system(stop_cmd.c_str()) == -1) {
                std::cerr << "Failure to call stop worker for "
                          << min_cpu_name[0] << std::endl;
                throw std::runtime_error("Failure to call stop worker");
              }
            } else if (delete_machines == 1) {
              Aws::String inst_id_aws(inst_id.c_str(), inst_id.size());
              Aws::EC2::Model::TerminateInstancesRequest request;
              request.AddInstanceIds(inst_id_aws);
              request.SetDryRun(false);
              auto term_outcome = ec2.TerminateInstances(request);
              if (!term_outcome.IsSuccess()) {
                std::cerr << "Failure to delete " << victim_worker;
                std::cerr << " instance!" << std::endl;
                throw std::runtime_error("Failure to delete worker instance");
              }
            } else {
              Aws::String inst_id_aws(inst_id.c_str(), inst_id.size());
              Aws::EC2::Model::StopInstancesRequest request;
              request.AddInstanceIds(inst_id_aws);
              request.SetDryRun(false);
              auto term_outcome = ec2.StopInstances(request);
              if (!term_outcome.IsSuccess()) {
                std::cerr << "Failure to stop " << victim_worker;
                std::cerr << " instance!" << std::endl;
                throw std::runtime_error("Failure to stop worker instance");
              }
            }

            // Reset vm_backoff_counter and shutdown_counter
            vm_backoff_counter = 0;
            vm_shutdown_counter = 0;
          } else {
            vm_shutdown_counter = 0;
          }
        } else {
          std::cout << "[LOG]: " << vm_shutdown_counter << " out of ";
          std::cout << vm_shutdown_thresh << " shutdown chances counted";
          std::cout << std::endl;
        }
      } else {
        vm_shutdown_counter = 0;  // Reset to zero
      }
    }

    if (vm_backoff_counter != num_iter) {
      vm_backoff_counter++;
      std::cout << "[LOG]: backoff counter: " << vm_backoff_counter;
      std::cout << " out of " << num_iter << std::endl;
      if (vm_backoff_counter == num_iter) {
        // Unset VM scale flag
        if (rm_.unset_vm_scale() < 0) {
          std::cerr << "Error resetting VM scale flag!" << std::endl;
          throw std::runtime_error("Error resetting VM scale flag");
        }
        // Unset slack scale flag
        if (rm_.unset_slack_scale() < 0) {
          std::cerr << "Error resetting slack scale flag!" << std::endl;
          throw std::runtime_error("Error resetting slack scale flag");
        }
      }
    }

    std::cout << "===============================================" << std::endl;
    usleep(sleep_seconds);
  }

  Aws::ShutdownAPI(options);
  return 0;
}
