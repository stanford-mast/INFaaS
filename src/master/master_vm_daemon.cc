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

enum INSTANCETYPE { CPU = 0, GPU = 1 };

// Constants
static const int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;

const int32_t sleep_seconds = 2e6;  // 1e6 = 1sec
const int16_t num_iter = 15;
const double cpu_shutdown_min = 5.0;
const double gpu_shutdown_min = 10.0;
const double avg_cpu_shutdown_min = 30;
const double avg_gpu_shutdown_min = 10;
const double max_blacklist = 80.0;

int16_t vm_backoff_counter;
int16_t vm_shutdown_thresh = 15;

std::map<INSTANCETYPE, int16_t> inst_type_map;

std::string stop_script = "scripts/stop_worker.sh";
std::string start_vm_script = "scripts/start_vm.sh";

int main(int argc, char** argv) {
  if (argc < 20) {
    std::cout
        << "Usage: ./master_vm_daemon <redis_ip> <redis_port> <util-thresh> ";
    std::cout << "<zone> <key-name> <worker-image> <machine-type-gpu> "
                 "<machine-type-cpu> ";
    std::cout << "<startup-script> <security-group> <max-try> <iam-role> "
                 "<exec-port> ";
    std::cout << "<exec-prefix> <min-workers> <max-workers> <master-ip> "
                 "<worker-autoscaler> ";
    std::cout << "<delete-machines>" << std::endl;
    return 1;
  }

  const struct Address redis_addr = {argv[1], argv[2]};
  if (RedisMetadata::is_empty_address(redis_addr)) {
    std::cout << "Invalid redis server address: "
              << RedisMetadata::Address_to_str(redis_addr) << std::endl;
    return 1;
  }

  // Set variables
  const double util_thresh = std::stod(argv[3]);
  const std::string zone = argv[4];
  const std::string key_name = argv[5];
  const std::string worker_image = argv[6];
  const std::string machine_type_gpu = argv[7];
  const std::string machine_type_cpu = argv[8];
  const std::string startup_script = argv[9];
  const std::string security_group = argv[10];
  const std::string max_try = argv[11];
  const std::string iam_role = argv[12];
  const std::string exec_port = argv[13];
  const std::string exec_prefix = argv[14];
  const int16_t min_workers = std::stoi(argv[15]);
  const int16_t max_workers = std::stoi(argv[16]);
  const std::string master_ip = argv[17];
  const std::string worker_autoscaler = argv[18];
  const int8_t delete_machines = std::stoi(argv[19]);

  if (delete_machines == 2) {
    std::cout << "[LOG]: Scaled down machines will be persisted, ";
    std::cout << "but removed from INFaaS's view" << std::endl;
  } else if (delete_machines == 1) {
    std::cout << "[LOG]: Scaled down machines will be deleted" << std::endl;
  } else {
    std::cout << "[LOG]: Scaled down machines will be stopped" << std::endl;
  }

  std::cout << "[LOG]: The maximum number of machines permitted is "
            << max_workers << std::endl;

  RedisMetadata rm_({redis_addr.ip, redis_addr.port});

  // Unset scale flag
  if (rm_.unset_vm_scale() < 0) {
    std::cerr << "Error resetting VM scale flag!" << std::endl;
    throw std::runtime_error("Error resetting VM scale flag");
  }

  // Go through all initial workers and categorize them as being CPU or GPU
  inst_type_map.insert(std::pair<INSTANCETYPE, int16_t>(CPU, 0));
  inst_type_map.insert(std::pair<INSTANCETYPE, int16_t>(GPU, 0));
  std::vector<std::string> all_workers = rm_.get_all_executors();
  for (std::string aw : all_workers) {
    if (rm_.is_exec_onlycpu(aw)) {
      inst_type_map[CPU]++;
    } else {
      inst_type_map[GPU]++;
    }
  }
  std::cout << "[LOG]: Starting with " << inst_type_map[CPU]
            << " CPU-only workers and ";
  std::cout << inst_type_map[GPU] << " GPU/CPU workers" << std::endl;

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

    // Print all of this for logging purposes
    std::cout << "[LOG]: Utilization threshold is " << util_thresh << std::endl;

    // Print number of executors:
    int16_t num_exec = rm_.get_num_executors();
    std::cout << "[LOG]: There are " << num_exec << " executors" << std::endl;

    // Print per-worker utilization for logging
    std::vector<std::string> all_min_cpu = rm_.min_cpu_util_name(num_exec);
    std::vector<std::string> all_min_gpu = rm_.min_gpu_util_name(num_exec);

    double avg_cpu_util = 0.0;
    for (std::string min_cpu : all_min_cpu) {
      double curr_cpu_util = rm_.get_cpu_util(min_cpu);
      // If the utilization is too high, blacklist the model
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
    std::cout << "[LOG]: Avg CPU Util: " << avg_cpu_util;
    std::cout << "; Avg GPU Util: " << avg_gpu_util << std::endl;

    // Avoid oscillating (either starting too many instances or killing them too
    // fast)
    if (vm_backoff_counter == num_iter) {
      // Get minimum GPU utilization
      double gpu_util = rm_.get_min_gpu_util();

      // Get minimum CPU utilization
      double cpu_util = rm_.get_min_cpu_util();

      // Check VM scale flag
      int8_t vm_scale_flag = rm_.vm_scale_status();

      std::cout << "[LOG]: Min overall GPU Util: " << gpu_util;
      std::cout << "; Min overall CPU Util: " << cpu_util;
      std::cout << "; VM Scale Flag: " << (int16_t)vm_scale_flag << std::endl;

      // Check if machines need to be started
      if ((num_exec < max_workers) &&
          ((gpu_util > util_thresh && gpu_util < 100.0) ||
           (cpu_util > util_thresh) || (vm_scale_flag == 1))) {
        std::cout << "[LOG]: Scaling triggered" << std::endl;

        std::string next_worker;
        std::string next_machine_type;
        bool is_cpuinstance = false;
        if (cpu_util > util_thresh) {
          next_machine_type = machine_type_cpu;
          next_worker =
              exec_prefix + "-cpu-" + std::to_string(inst_type_map[CPU]);
          inst_type_map[CPU]++;
          is_cpuinstance = true;
          std::cout << "[LOG]: Adding a new CPU instance: " << next_worker
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

        // IP addresses are 10 numbers + 3 periods + 1 for 0-indexing
        std::string exec_ip;
        if (result.size() == 13) {
          exec_ip = result;
        } else {
          exec_ip = result.substr(result.size() - 14);
        }

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
          usleep(sleep_seconds);
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
            std::cerr << "Failure to set " << next_worker << " as CPU only!"
                      << std::endl;
            throw std::runtime_error("Failure to set worker as CPU only");
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

      if (((gpu_util <= gpu_shutdown_min) && (cpu_util <= cpu_shutdown_min)) ||
          ((avg_gpu_util <= avg_gpu_shutdown_min) &&
           (avg_cpu_util <= avg_gpu_shutdown_min))) {
        vm_shutdown_counter++;  // Give a machine vm_shutdown_thresh chances
                                // before killing it
        if (vm_shutdown_counter == vm_shutdown_thresh) {
          // Check that the min is the same for both CPU and GPU
          std::vector<std::string> min_cpu_name = rm_.min_cpu_util_name(1);
          std::vector<std::string> min_gpu_name = rm_.min_gpu_util_name(3);

          bool gpu_check = false;
          auto it =
              find(min_gpu_name.begin(), min_gpu_name.end(), min_cpu_name[0]);
          if (it != min_gpu_name.end()) { gpu_check = true; }

          // Make sure the first workers are not deleted
          if (gpu_check && (num_exec > min_workers)) {
            std::cout << "[LOG]: Shutdown triggered, victim worker: ";
            std::cout << min_cpu_name[0] << std::endl;

            // First get the worker's instance-id
            std::string inst_id = rm_.get_executor_instid(min_cpu_name[0]);
            if (inst_id == "FAIL") {
              std::cerr << "Failed to get instance-id of " << min_cpu_name[0]
                        << std::endl;
              throw std::runtime_error("Failed to get instance-id");
            }

            // Get the worker's IP: useful if the machine will be persisted.
            std::string del_exec_ip =
                (rm_.get_executor_addr(min_cpu_name[0])).ip;

            // Decrement the appropriate machine's counter
            if (rm_.is_exec_onlycpu(min_cpu_name[0])) {
              inst_type_map[CPU]--;
              std::cout << "[LOG]: Now there's " << inst_type_map[CPU]
                        << " CPU-only workers";
              std::cout << std::endl;
            } else {
              inst_type_map[GPU]--;
              std::cout << "[LOG]: Now there's " << inst_type_map[GPU]
                        << " CPU/GPU workers";
              std::cout << std::endl;
            }

            // Remove the worker from the metadata store
            int8_t rc = rm_.delete_executor(min_cpu_name[0]);
            if (rc == -1) {
              std::cerr << "Failure to delete " << min_cpu_name[0]
                        << " from metadata store!" << std::endl;
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
                std::cerr << "Failure to delete " << min_cpu_name[0]
                          << " instance!" << std::endl;
                throw std::runtime_error("Failure to delete worker instance");
              }
            } else {
              Aws::String inst_id_aws(inst_id.c_str(), inst_id.size());
              Aws::EC2::Model::StopInstancesRequest request;
              request.AddInstanceIds(inst_id_aws);
              request.SetDryRun(false);
              auto term_outcome = ec2.StopInstances(request);
              if (!term_outcome.IsSuccess()) {
                std::cerr << "Failure to stop " << min_cpu_name[0]
                          << " instance!" << std::endl;
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
          std::cout << vm_shutdown_thresh << " shutdown chances counted"
                    << std::endl;
        }
      } else {
        vm_shutdown_counter = 0;  // Reset to zero
      }
    }

    if (vm_backoff_counter != num_iter) {
      vm_backoff_counter++;
      std::cout << "[LOG]: backoff counter: " << vm_backoff_counter
                << " out of ";
      std::cout << num_iter << std::endl;
      if (vm_backoff_counter == num_iter) {
        // Unset scale flag
        if (rm_.unset_vm_scale() < 0) {
          std::cerr << "Error resetting VM scale flag!" << std::endl;
          throw std::runtime_error("Error resetting VM scale flag");
        }
      }
    }

    std::cout << "===============================================" << std::endl;
    usleep(sleep_seconds);
  }

  Aws::ShutdownAPI(options);
  return 0;
}
