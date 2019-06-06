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

#include <time.h>
#include <algorithm>  // sort, set_intersection, min, shuffle
#include <chrono>
#include <cstdint>
#include <deque>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/ListObjectsV2Request.h>

#include <grpcpp/grpcpp.h>
#include "include/constants.h"
#include "metadata-store/redis_metadata.h"
#include "queryfe.grpc.pb.h"

#include "worker/query_client.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

//// Constants and global variables ////
static const int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;

// Decision-making constants
static const int16_t gmod_max_results = 7;
static const int16_t gmod_max_lru = 5;
static const int16_t qps_vm_scale_query_limit = 500;
static const double qps_vm_scale_time_interval = 1000.0;

enum MasterDecisions {
  INFAAS_ALL = 0,
  INFAAS_NOQPSLAT = 1,
  ROUNDROBIN = 2,
  ROUNDROBIN_STATIC = 3,
  GPUSHARETRIGGER = 4,
  CPUBLISTCHECK = 5
};

namespace infaaspublic {
namespace infaasqueryfe {
namespace {
// Timestamp to millisecond duration.
double ts_to_ms(const struct timeval& start, const struct timeval& end) {
  return (end.tv_sec - start.tv_sec) * 1000.0 +
         (end.tv_usec - start.tv_usec) / 1000.0;
}

void parse_s3_url(const std::string& src_url, std::string* src_bucket,
                  std::string* obj_name) {
  // *obj_name = src_url.substr(bucket_prefix.size());
  *obj_name = src_url;
  auto pre_ind = obj_name->find("/");  // exclude bucket name
  *src_bucket = obj_name->substr(0, pre_ind);
  *obj_name = obj_name->substr(pre_ind + 1);
}

}  // namespace

// Logic and data behind the server's behavior.
class QueryServiceImpl final : public Query::Service {
public:
  QueryServiceImpl(const struct Address redis_addr,
                   const int8_t decision_policy)
      : redis_addr_(redis_addr),
        all_exec_counter_(0),
        all_exec_gpu_counter_(0),
        last_worker_picked_("") {
    rm_ = std::unique_ptr<RedisMetadata>(new RedisMetadata(redis_addr_));

    if (decision_policy == 0) {
      master_decision_ = INFAAS_ALL;
      std::cout << "Using mode: INFAAS_ALL" << std::endl;
    } else if (decision_policy == 1) {
      master_decision_ = INFAAS_NOQPSLAT;
      std::cout << "Using mode: INFAAS_NOQPSLAT" << std::endl;
    } else if (decision_policy == 2) {
      master_decision_ = ROUNDROBIN;
      std::cout << "Using mode: ROUNDROBIN" << std::endl;
    } else if (decision_policy == 3) {
      master_decision_ = ROUNDROBIN_STATIC;
      std::cout << "Using mode: ROUNDROBIN_STATIC" << std::endl;
    } else if (decision_policy == 4) {
      master_decision_ = GPUSHARETRIGGER;
      std::cout << "Using mode: GPUSHARETRIGGER" << std::endl;
    } else if (decision_policy == 5) {
      master_decision_ = CPUBLISTCHECK;
      std::cout << "Using mode: CPUBLISTCHECK" << std::endl;
    } else {
      std::cerr << (int16_t)decision_policy << " is not a valid decision policy"
                << std::endl;
      throw std::runtime_error("Invalid decision policy");
    }

    // Set up workers for Round-Robin
    if ((master_decision_ == ROUNDROBIN) ||
        (master_decision_ == ROUNDROBIN_STATIC)) {
      // Get all executors
      all_exec_ = rm_->get_all_executors();
      for (std::string ae : all_exec_) {
        if (!rm_->is_exec_onlycpu(ae)) { all_gpu_exec_.push_back(ae); }
      }
    }
  }

private:
  // The assumption for this function is that BOTH an accuracy and a latency
  //// constraint are provided. The search will first check for models that
  ///satisfy the / accuracy constraint before moving on to finding one that
  ///satisfies latency.
  // This algorithm is similar to the parent latency-accuracy search with the
  // exception
  //// that the initial model pruning cannot be an exhaustive search
  std::vector<std::string> gpar_lat_acc_search(
      const std::string& gparent_model, const double& accuracy_constraint,
      const int64_t& latency_constraint, const int16_t& batch_size,
      int8_t* is_running, MasterDecisions dec_policy) {
    // Do a tailored fast check search using the last couple of queries.
    // If there is a valid running model, use it. Otherwise, do an exhaustive
    // search.
    if (!gmod_cache_.empty()) {
      std::string candidate_model;
      for (std::string avl : gmod_cache_) {
        std::cout << "[LOG]: GPAR fast check -- currently considering model "
                     "variant: ";
        std::cout << avl << std::endl;

        // Check that batch size is valid
        std::string mv_batch = rm_->get_model_info(avl, "max_batch");
        int16_t mv_batch_int = std::stoi(mv_batch);
        if (mv_batch != "FAIL") {
          if (mv_batch_int < batch_size) {  // Not supported
            continue;
          }
        } else {
          return {};
        }

        std::cout << "[LOG]: Passes batch" << std::endl;

        // Check that the accuracy is valid
        double mv_acc = rm_->get_accuracy(avl);
        if (mv_acc > 0.0) {
          if (mv_acc < accuracy_constraint) { continue; }
        } else {
          return {};
        }

        std::cout << "[LOG]: Passes accuracy" << std::endl;

        // Get its inference latency based on slope and intercept computed
        // during registration
        double mv_inf_lat, slope, intercept, batch_for_compute;
        if (latency_constraint > 0) {
          slope = std::stod(rm_->get_model_info(avl, "slope"));
          intercept = std::stod(rm_->get_model_info(avl, "intercept"));

          if (mv_batch_int > 64) {  // CPU model
            batch_for_compute = (double)batch_size;
          } else {
            batch_for_compute = ((double)std::min((int16_t)32, mv_batch_int));
          }

          mv_inf_lat = slope * batch_for_compute + intercept;
          if (mv_inf_lat > 0.0) {
            if (mv_inf_lat > latency_constraint) { continue; }
          } else {
            return {};
          }
        }

        std::cout << "[LOG]: Passes latency" << std::endl;

        // Check if it's running
        if (rm_->is_model_running(avl)) {
          std::cout << "[LOG]: " << avl << " is running" << std::endl;

          if (dec_policy == INFAAS_NOQPSLAT) {  // Just pick the model
            *is_running = 1;
            candidate_model = avl;
            break;
          }

          // If model is being unloaded, skip it
          if (rm_->get_model_load_unload(avl)) {
            std::cout << "[LOG] GPAR fast-check: " << avl
                      << " is being unloaded";
            std::cout << std::endl;
            continue;
          }

          // Check if the minimum QPS replica has been blacklisted
          std::vector<std::string> min_worker_name = rm_->min_qps_name(avl, 1);
          if (min_worker_name.empty()) {
            std::cout << "[LOG]: Model set to running, but min_qps_name query "
                         "failed.";
            std::cout << "Continuing search" << std::endl;
            continue;
          }

          int8_t is_blisted =
              rm_->get_model_avglat_blacklist(min_worker_name[0], avl);

          if (is_blisted < 0) {
            throw std::runtime_error(
                "Failed to check if model was blacklisted");
          } else if (!is_blisted) {
            std::cout << "[LOG]: Meets blacklist check, picking ";
            std::cout << avl << std::endl;
            *is_running = 1;
            candidate_model = avl;
            break;
          } else {  // It's blacklisted
            std::cout << "[LOG]: Failed blacklist check" << std::endl;
            // If it's a GPU fast check, return it, but set is_running to 0
            if (mv_batch_int < 64) {
              std::cout << "[LOG]: Picking blacklisted GPU model in fast-check"
                        << std::endl;
              if ((dec_policy == GPUSHARETRIGGER) ||
                  (dec_policy ==
                   CPUBLISTCHECK)) {  // Ask for new VM from autoscaler
                std::cout << "[LOG]: GPU model blacklisted in fast-check,";
                std::cout << " triggering new VM" << std::endl;
                rm_->set_vm_scale();
              }
              *is_running = 0;
              candidate_model = avl;
              break;
            }
          }
        }
      }
      if (candidate_model.empty()) {
        std::cout
            << "[LOG]: No models running for GPAR fast-check, switching to";
        std::cout << " general search (SLO will be violated)" << std::endl;
      } else {
        // Update the cache order
        std::cout << "[LOG]: Found in cache, updating and returning"
                  << std::endl;
        gmod_cache_.erase(
            std::find(gmod_cache_.begin(), gmod_cache_.end(), candidate_model));
        gmod_cache_.push_front(candidate_model);
        return {candidate_model};
      }
    }

    std::vector<std::string> acc_opts;
    double min_latency = (latency_constraint > 100.0) ? 100.0 : 0.0;
    std::pair<std::string, double> candidate_variant("dummy", 100000.0);
    std::pair<std::string, double> best_cpu("dummy", 100000.0);
    int16_t min_valid_batch = 512;
    bool general_valid_model = false;
    bool cpu_blacklisted = false;

    // Get feasible models based on accuracy
    acc_opts =
        rm_->gpar_acc_bin(gparent_model, accuracy_constraint, gmod_max_results);

    if (acc_opts.size() == 0) {
      std::cout << "[LOG]: Accuracy bin search returned no models!"
                << std::endl;
      return {};
    }

    std::cout << "[LOG]: Reviewing " << acc_opts.size() << " variants for ";
    std::cout << gparent_model << std::endl;

    // Go through all model variants and find first one that satisfies
    // 1) batch, 2) latency, 3) is loaded.
    // If no model is loaded, find the one with the lowest total latency
    for (std::string av : acc_opts) {
      std::cout << "[LOG]: Currently considering model variant: " << av
                << std::endl;

      bool better_batch = false;

      // Get its batch size
      std::string mv_batch = rm_->get_model_info(av, "max_batch");
      int16_t mv_batch_int = std::stoi(mv_batch);
      if (mv_batch != "FAIL") {
        if (mv_batch_int < batch_size) {  // Not supported
          continue;
        } else {  // Batch size supported
          if (mv_batch_int < min_valid_batch) {
            std::cout << "[LOG]: " << av << " (batch-" << mv_batch_int;
            std::cout << ") is the current best batch fit" << std::endl;
            min_valid_batch = mv_batch_int;
            better_batch = true;
          }
        }
      } else {
        return {};
      }

      std::cout << "[LOG]: Passes batch" << std::endl;

      // Get its inference latency based on slope and intercept computed during
      // registration
      double mv_inf_lat, slope, intercept, batch_for_compute;
      if (latency_constraint > 0) {
        slope = std::stod(rm_->get_model_info(av, "slope"));
        intercept = std::stod(rm_->get_model_info(av, "intercept"));

        if (mv_batch_int > 64) {  // CPU model
          batch_for_compute = (double)batch_size;
        } else {
          batch_for_compute = ((double)std::min((int16_t)32, mv_batch_int));
        }

        mv_inf_lat = slope * batch_for_compute + intercept;
        if (mv_inf_lat > 0.0) {
          if (mv_inf_lat > latency_constraint) { continue; }
        } else {
          return {};
        }
      }

      std::cout << "[LOG]: Passes latency" << std::endl;

      // Check if it's running
      bool valid_model_running = false;
      if (rm_->is_model_running(av)) {
        std::cout << "[LOG]: " << av << " is running" << std::endl;

        if (dec_policy == INFAAS_NOQPSLAT) {  // Just pick the model
          *is_running = 1;
          auto it = std::find(gmod_cache_.begin(), gmod_cache_.end(), av);
          if (it == gmod_cache_.end()) {
            // Update cache
            std::cout << "[LOG]: Was not in cache, adding..." << std::endl;
            if (gmod_cache_.size() == gmod_max_lru) { gmod_cache_.pop_back(); }
          } else {
            // If found, it means the model wasn't running
            std::cout << "[LOG]: Was in cache, but not running. Moving to front"
                      << std::endl;
            gmod_cache_.erase(it);
          }
          gmod_cache_.push_front(av);
          return {av};
        }

        // If model is being unloaded, skip it
        if (rm_->get_model_load_unload(av)) {
          std::cout << "[LOG]: " << av << " is being unloaded" << std::endl;
          continue;
        }

        // Check if the minimum QPS replica has been blacklisted
        std::vector<std::string> min_worker_name = rm_->min_qps_name(av, 1);
        if (min_worker_name.empty()) {
          std::cout
              << "[LOG]: Model set to running, but min_qps_name query failed.";
          std::cout << "Taking not running path" << std::endl;
        } else {
          if (mv_batch_int > 64) {
            std::string parent_model = rm_->get_parent_model(av);
            int8_t pmod_scaledown =
                rm_->get_parent_scaledown(min_worker_name[0], parent_model);

            if (pmod_scaledown < 0) {
              throw std::runtime_error(
                  "Failed to check if model was in scaledown mode");
            } else if (!pmod_scaledown) {
              std::cout << "[LOG]: Passes scaledown check for CPU" << std::endl;
              valid_model_running = true;
            } else {
              std::cout << "[LOG]: Failed scaledown check for CPU" << std::endl;
            }
          } else {
            std::cout << "[LOG]: Passes scaledown check" << std::endl;
            valid_model_running = true;
          }
        }

        if (valid_model_running) {
          int8_t is_blisted =
              rm_->get_model_avglat_blacklist(min_worker_name[0], av);

          if (is_blisted < 0) {
            throw std::runtime_error(
                "Failed to check if model was blacklisted");
          } else if (!is_blisted) {
            std::cout << "[LOG]: Meets blacklist check" << std::endl;
            double lat_diff = latency_constraint - mv_inf_lat;
            if ((lat_diff >= 0.0) && (lat_diff < candidate_variant.second)) {
              std::cout << "[LOG]: Current best: " << av << std::endl;
              *is_running = 1;
              general_valid_model = true;
              candidate_variant.first = av;
              candidate_variant.second = lat_diff;
            }
          } else {  // It's blacklisted
            std::cout << "[LOG]: Failed blacklist check" << std::endl;
            if (mv_batch_int > 64) {
              cpu_blacklisted = true;
            } else {
              // If a GPU model is blacklisted, make it a candidate, but
              //// set a flag to indicate that it should not be seen as running.
              double lat_diff = latency_constraint - mv_inf_lat;
              if ((lat_diff >= 0.0) && (lat_diff < candidate_variant.second)) {
                std::cout << "[LOG]: Making blacklisted GPU model a candidate"
                          << std::endl;
                *is_running = 0;
                candidate_variant.first = av;
                candidate_variant.second = lat_diff;
              }
              if ((dec_policy == GPUSHARETRIGGER) ||
                  (dec_policy ==
                   CPUBLISTCHECK)) {  // Ask for new VM from autoscaler
                std::cout << "[LOG]: GPU model blacklisted, triggering new VM"
                          << std::endl;
                rm_->set_vm_scale();
              }
            }
          }
        }
      }

      if (!valid_model_running) {
        // Model is not running, make it a candidate based on the load+inf
        // latency
        //// if it is a better batch size than previously seen variants.
        // Thus, these variants will not get picked unless no variant is running

        // "Punish" these models by adding an extra 1000ms; otherwise, for very
        // small
        //// models it is possible for these models to look "better" than loaded
        ///models
        std::cout << "[LOG]: Model not running, recording total latency"
                  << std::endl;
        double mv_load_lat = rm_->get_load_lat(av);
        double mv_total_lat = 1000.0 + mv_load_lat + mv_inf_lat;
        if (better_batch && (mv_total_lat < candidate_variant.second)) {
          *is_running = 0;
          candidate_variant.first = av;
          candidate_variant.second = mv_total_lat;
        }
        if ((min_latency == 100.0) && (mv_batch_int == 128)) {  // CPU models
          std::cout << "[LOG]: Checking CPU for total latency" << std::endl;
          if (mv_total_lat < best_cpu.second) {
            best_cpu.first = av;
            best_cpu.second = mv_total_lat;
          }
        }
      }
    }

    // If best cpu model is valid, it means the latency constraint
    //// was loose and no model was running.
    // If no other running model was found and if the CPU wasn't blacklisted,
    // start a CPU
    if ((best_cpu.first != "dummy") && !general_valid_model &&
        !cpu_blacklisted) {
      std::cout
          << "[LOG]: No model running, constraint dictates picking a CPU model";
      std::cout << std::endl;
      *is_running = 0;
      auto it =
          std::find(gmod_cache_.begin(), gmod_cache_.end(), best_cpu.first);
      if (it == gmod_cache_.end()) {
        // Update cache
        std::cout << "[LOG]: Was not in cache, adding..." << std::endl;
        if (gmod_cache_.size() == gmod_max_lru) { gmod_cache_.pop_back(); }
      } else {
        // If found, it means the model wasn't running
        std::cout << "[LOG]: Was in cache, but not running. Moving to front"
                  << std::endl;
        gmod_cache_.erase(it);
      }
      gmod_cache_.push_front(best_cpu.first);
      return {best_cpu.first};
    }

    // Otherwise, if candidate model is valid, return it
    if (candidate_variant.first != "dummy") {
      std::cout << "[LOG]: Valid running model was found" << std::endl;
      auto it = std::find(gmod_cache_.begin(), gmod_cache_.end(),
                          candidate_variant.first);
      if (it == gmod_cache_.end()) {
        // Update cache
        std::cout << "[LOG]: Was not in cache, adding..." << std::endl;
        if (gmod_cache_.size() == gmod_max_lru) { gmod_cache_.pop_back(); }
      } else {
        // If found, it means the model wasn't running
        std::cout << "[LOG]: Was in cache, but not running. Moving to front"
                  << std::endl;
        gmod_cache_.erase(it);
      }
      gmod_cache_.push_front(candidate_variant.first);
      return {candidate_variant.first};
    }

    std::cout << "[LOG]: GPAR: No model found! This will fail..." << std::endl;
    return {};
  }

  std::vector<std::string> par_lat_search(const std::string& parent_model,
                                          const int64_t& latency_constraint,
                                          const int16_t& batch_size,
                                          int8_t* is_running,
                                          MasterDecisions dec_policy) {
    std::pair<std::string, double> candidate_variant("dummy", 100000.0);
    int16_t max_batch_blisted = 0;
    std::pair<std::string, int16_t> min_valid_batch = {"dummy", 512};

    // Skip scaledown check if CPU model is blacklisted
    bool skip_scaledown = false;
    // Do a tailored latency search given the latency constraint. If models are
    // found
    //// and running, return it. Otherwise, do an exhaustive search.
    double min_latency = (latency_constraint > 100.0) ? 100.0 : 0.0;
    std::vector<std::string> lowest_tot =
        rm_->inf_lat_bin(parent_model, min_latency, latency_constraint, 5);
    if (!lowest_tot.empty()) {
      for (std::string avl : lowest_tot) {
        std::cout
            << "[LOG]: Fast check -- currently considering model variant: ";
        std::cout << avl << std::endl;

        // Check that batch size is valid
        std::string mv_batch = rm_->get_model_info(avl, "max_batch");
        int16_t mv_batch_int = std::stoi(mv_batch);
        if (mv_batch != "FAIL") {
          if (mv_batch_int < batch_size) {  // Not supported
            continue;
          }
        } else {
          return {};
        }

        std::cout << "[LOG]: Passes batch" << std::endl;

        // Check if it's running
        if (rm_->is_model_running(avl)) {
          std::cout << "[LOG]: " << avl << " is running" << std::endl;

          if (dec_policy == INFAAS_NOQPSLAT) {  // Just pick the model
            *is_running = 1;
            return {avl};
          }

          // If model is being unloaded, skip it
          if (rm_->get_model_load_unload(avl)) {
            std::cout << "[LOG] PAR fast-check: " << avl << " is being unloaded"
                      << std::endl;
            continue;
          }

          // Check if the minimum QPS replica has been blacklisted
          std::vector<std::string> min_worker_name = rm_->min_qps_name(avl, 1);
          if (min_worker_name.empty()) {
            std::cout << "[LOG]: Model set to running, but min_qps_name query "
                         "failed.";
            std::cout << "Continuing search" << std::endl;
            continue;
          }

          int8_t is_blisted =
              rm_->get_model_avglat_blacklist(min_worker_name[0], avl);

          if (is_blisted < 0) {
            throw std::runtime_error(
                "Failed to check if model was blacklisted");
          } else if (!is_blisted) {
            if (dec_policy == CPUBLISTCHECK) {
              // If it's a CPU model and it's in the set, skip it
              if (mv_batch_int > 64) {
                if (cpu_blist_.find(avl) != cpu_blist_.end()) {
                  std::cout << "[LOG]: " << avl << " in cpu_blist_, skipping..."
                            << std::endl;
                  continue;
                }
              }
            }
            std::cout << "[LOG]: Meets blacklist check, picking ";
            std::cout << avl << std::endl;
            *is_running = 1;
            return {avl};
          } else {  // It's blacklisted
            std::cout << "[LOG]: Failed blacklist check" << std::endl;
            // If it's a GPU fast check, return it, but set is_running to 0
            if (mv_batch_int < 64) {
              std::cout << "[LOG]: Picking blacklisted GPU model in fast-check"
                        << std::endl;
              if ((dec_policy == GPUSHARETRIGGER) ||
                  (dec_policy ==
                   CPUBLISTCHECK)) {  // Ask for new VM from autoscaler
                std::cout << "[LOG]: GPU model blacklisted in fast-check,";
                std::cout << " triggering new VM" << std::endl;
                rm_->set_vm_scale();
              }
              *is_running = 0;
              return {avl};
            } else {
              // Skip scaledown if it's a CPU
              std::cout << "[LOG] Skipping future scaledown checks"
                        << std::endl;
              skip_scaledown = true;
            }
          }
        }
      }
      std::cout << "[LOG]: No models running for fast-check, switching to";
      std::cout << " general search (SLO will be violated)" << std::endl;
    }

    // Get all model variants
    std::vector<std::string> all_var =
        rm_->get_all_model_variants(parent_model);

    if (all_var.size() == 0) {
      std::cout << "[LOG]: All variant query returned no models!" << std::endl;
      return {};
    }

    std::cout << "[LOG]: Parent has " << all_var.size() << " variants"
              << std::endl;

    // Go through all model variants and find first one that satisfies
    // 1) batch, 2) latency, 3) is loaded.
    // If no model is loaded, find the one with the lowest total latency

    // This boolean checks to see if there is even a model that can satisfy the
    // request based on accuracy and inference latency. If such a model does not
    // exist, we should not attempt to even search for one if no model is
    // running.
    bool model_exists = false;
    bool gpu_blisted = false;
    bool cpuonly_blisted = false;
    std::string val_cpu_running = "dummy";
    std::string val_gpu_running = "dummy";
    for (std::string av : all_var) {
      std::cout << "[LOG]: Currently considering model variant: " << av
                << std::endl;

      // Get its batch size
      std::string mv_batch = rm_->get_model_info(av, "max_batch");
      int16_t mv_batch_int = std::stoi(mv_batch);
      if (mv_batch != "FAIL") {
        if (mv_batch_int < batch_size) {  // Not supported
          continue;
        } else {  // Batch size supported
          // Skip CPU models if the latency constraint is low
          if (mv_batch_int == 128 && (latency_constraint <= 50.0)) {
            std::cout << "[LOG]: Skipping CPU model for latency constraint of ";
            std::cout << latency_constraint << std::endl;
            continue;
          }

          if (mv_batch_int < min_valid_batch.second) {
            std::cout << "[LOG]: Saving " << av << " (batch-" << mv_batch_int;
            std::cout << ") for later" << std::endl;
            min_valid_batch.first = av;
            min_valid_batch.second = mv_batch_int;
          }
        }
      } else {
        return {};
      }

      std::cout << "[LOG]: Passes batch" << std::endl;

      // Get its inference latency based on slope and intercept computed during
      // registration
      double mv_inf_lat, slope, intercept, batch_for_compute;
      if (latency_constraint > 0) {
        slope = std::stod(rm_->get_model_info(av, "slope"));
        intercept = std::stod(rm_->get_model_info(av, "intercept"));

        if (mv_batch_int > 64) {  // CPU model
          batch_for_compute = (double)batch_size;
        } else {
          batch_for_compute = ((double)std::min((int16_t)32, mv_batch_int));
        }

        mv_inf_lat = slope * batch_for_compute + intercept;
        if (mv_inf_lat > 0.0) {
          if (mv_inf_lat > latency_constraint) { continue; }
        } else {
          return {};
        }
      }

      std::cout << "[LOG]: Passes latency" << std::endl;

      // At this point, we know the model exists
      model_exists = true;

      // Check if it's running
      if (rm_->is_model_running(av)) {
        std::cout << "[LOG]: " << av << " is running" << std::endl;

        if (dec_policy == INFAAS_NOQPSLAT) {  // Just pick the model
          *is_running = 1;
          return {av};
        }

        // If model is being unloaded, skip it
        if (rm_->get_model_load_unload(av)) {
          std::cout << "[LOG]: " << av << " is being unloaded" << std::endl;
          continue;
        }

        // Check if the minimum QPS replica has been blacklisted
        std::vector<std::string> min_worker_name = rm_->min_qps_name(av, 1);
        if (min_worker_name.empty()) {
          std::cout
              << "[LOG]: Model set to running, but min_qps_name query failed.";
          std::cout << "Continuing search" << std::endl;
          continue;
        } else {
          // Check if parent model scaledown flag has been set.
          // If so, skip over GPU models
          if (!skip_scaledown) {
            int8_t pmod_scaledown =
                rm_->get_parent_scaledown(min_worker_name[0], parent_model);

            if (pmod_scaledown < 0) {
              throw std::runtime_error(
                  "Failed to check if model was in scaledown mode");
            } else if (pmod_scaledown && (min_latency == 100.0)) {
              std::cout << "[LOG]: Scaledown set and user asked for a CPU"
                        << std::endl;
              break;
            }

            std::cout << "[LOG]: Passes scaledown check" << std::endl;
          } else {
            std::cout << "[LOG]: Skipped scaledown check" << std::endl;
          }

          int8_t is_blisted =
              rm_->get_model_avglat_blacklist(min_worker_name[0], av);

          if (is_blisted < 0) {
            throw std::runtime_error(
                "Failed to check if model was blacklisted");
          } else if (!is_blisted) {
            if (dec_policy == CPUBLISTCHECK) {
              // If this is a GPU and a CPU was seen to be running, overwrite
              // it. Otherwise, if it's a CPU, first check if a GPU had
              // previously
              //// been running. If so, skip it. Otherwise, proceed as normal
              if (mv_batch_int < 64) {
                val_gpu_running = av;
                if (val_cpu_running != "dummy") {
                  std::cout << "[LOG]: Valid CPU running, but " << av;
                  std::cout << " will overwrite it" << std::endl;
                  double lat_diff = latency_constraint - mv_inf_lat;
                  candidate_variant.first = av;
                  candidate_variant.second = lat_diff;
                  gpu_blisted =
                      false;  // Reset flag; it should be seen as running
                  continue;
                }
              } else {
                val_cpu_running = av;
                if (val_gpu_running != "dummy") {
                  std::cout << "[LOG]: Skipping " << av << " because";
                  std::cout << " a valid GPU is running" << std::endl;
                  continue;
                }
              }
            }

            std::cout << "[LOG]: Meets blacklist check" << std::endl;
            double lat_diff = latency_constraint - mv_inf_lat;
            if ((lat_diff >= 0.0) && (lat_diff < candidate_variant.second)) {
              candidate_variant.first = av;
              candidate_variant.second = lat_diff;
              gpu_blisted = false;  // Reset flag; it should be seen as running
              cpuonly_blisted =
                  false;  // Reset flag; it should be seen as running
            }
          } else {  // It's blacklisted
            std::cout << "[LOG]: Failed blacklist check" << std::endl;
            if (mv_batch_int > max_batch_blisted) {
              max_batch_blisted = mv_batch_int;
            }
            // If a GPU model is blacklisted, make it a candidate, but
            //// set a flag to indicate that it should not be seen as running.
            if (mv_batch_int < 64) {
              double lat_diff = latency_constraint - mv_inf_lat;
              if ((lat_diff >= 0.0) && (lat_diff < candidate_variant.second)) {
                std::cout << "[LOG]: Making blacklisted GPU model a candidate"
                          << std::endl;
                candidate_variant.first = av;
                candidate_variant.second = lat_diff;
                gpu_blisted = true;
              }
              if ((dec_policy == GPUSHARETRIGGER) ||
                  (dec_policy ==
                   CPUBLISTCHECK)) {  // Ask for new VM from autoscaler
                std::cout << "[LOG]: GPU model blacklisted, triggering new VM"
                          << std::endl;
                rm_->set_vm_scale();
              }
            } else {  // CPU is blacklisted
              if (dec_policy == CPUBLISTCHECK) {
                std::cout << "[LOG] Adding " << av << " to cpu_blist_"
                          << std::endl;
                if (cpu_blist_.find(val_cpu_running) == cpu_blist_.end()) {
                  cpu_blist_.insert(val_cpu_running);
                }

                // If model is a PyTorch one, blacklist, but still possibly pick
                // it
                if (rm_->is_pytorch_only(parent_model) == 1) {
                  std::cout << "[LOG]: " << av
                            << " is a blacklisted PyTorch variant";
                  std::cout << std::endl;
                  double lat_diff = latency_constraint - mv_inf_lat;
                  if ((lat_diff >= 0.0) &&
                      (lat_diff < candidate_variant.second)) {
                    std::cout << "[LOG]: Making blacklisted Pytorch model";
                    std::cout << " a candidate" << std::endl;
                    candidate_variant.first = av;
                    candidate_variant.second = lat_diff;
                    cpuonly_blisted = true;
                  }
                }
              }
            }
          }
        }
      } else {  // Is not running
        if (dec_policy == CPUBLISTCHECK) {
          // If it's a CPU model and it's in the set,
          //// it is no longer running. Remove it.
          if (mv_batch_int > 64) {
            if (cpu_blist_.find(av) != cpu_blist_.end()) {
              std::cout << "[LOG]: Removing " << av << " from cpu_blist_"
                        << std::endl;
              cpu_blist_.erase(av);
            }
          }
        }
      }
    }

    if (model_exists == false) { return {}; }

    // If candidate model is valid, return it
    if (candidate_variant.first != "dummy") {
      if (gpu_blisted || cpuonly_blisted) {
        std::cout << "[LOG]: Picking a blacklisted GPU/PyTorch model"
                  << std::endl;
        *is_running = 0;
      } else {
        *is_running = 1;
      }
      return {candidate_variant.first};
    } else {
      // If max_batch_blisted is 0, it means no model was running.
      if (lowest_tot.empty()) {
        // If min_latency == 100, it means the parent model has no CPU variants.
        //// Run search again, but expand to all variants.
        if (min_latency == 100.0) {
          std::cout
              << "[LOG]: PAR: No models running, and no CPU variants available";
          std::cout << std::endl;
          std::vector<std::string> lowest_tot_cpu =
              rm_->inf_lat_bin(parent_model, 0, latency_constraint, 1);
          if (!lowest_tot_cpu.empty()) { return lowest_tot_cpu; }
        }
        std::cout << "[LOG]: PAR: No model found! This will fail..."
                  << std::endl;
        return {};
      }

      *is_running = 0;
      if (max_batch_blisted == 0) {
        std::cout << "[LOG]: Valid models found, but none were running..."
                  << std::endl;
        if (min_latency == 100.0) {
          std::cout << "[LOG]: Latency contraint permits returning a CPU model"
                    << std::endl;
          return {lowest_tot[0]};
        } else {
          // Submit minimum valid batch model
          if (min_valid_batch.first != "dummy") {
            std::cout << "[LOG]: " << min_valid_batch.first
                      << " meets batch constraint";
            std::cout << std::endl;
            return {min_valid_batch.first};
          } else {
            std::cout
                << "[LOG]: No valid batch size model for max_batch_blist=0";
            std::cout << " (SHOULD NEVER BE REACHED!)" << std::endl;
            return {};
          }
        }
      } else {  // Model(s) running, but blacklisted
        std::cout << "[LOG]: Valid models found, but were blacklisted..."
                  << std::endl;
        // If max_batch_blisted is a CPU model, move to the first valid batch
        // model on GPU
        if (min_valid_batch.first != "dummy") {
          std::cout << "[LOG]: " << min_valid_batch.first
                    << " meets batch constraint";
          std::cout << " from blacklist" << std::endl;
          return {min_valid_batch.first};
        } else {
          std::cout
              << "[LOG]: No valid batch size model for max_batch_blist!=0";
          std::cout << " (SHOULD NEVER BE REACHED!)" << std::endl;
          return {};
        }
      }
    }
  }

  Status QueryOnline(ServerContext* context, const QueryOnlineRequest* request,
                     QueryOnlineResponse* reply) override {
    struct timeval time1, time2, time3;
    gettimeofday(&time1, NULL);

    infaaspublic::RequestReply* rs = reply->mutable_status();

    // User should submit empty string if empty
    std::string grandparent_model = request->grandparent_model();
    std::string parent_model = request->parent_model();
    std::string model = request->model_variant();

    std::string submitter = request->submitter();
    auto slo = request->slo();

    std::string next_worker;
    struct Address dest_addr = {"0", "0"};
    int8_t is_running = 1;  // Will be changed below if applicable

    if (!model.empty()) {  // Model variant provided
      if (!rm_->model_registered(model)) {
        rs->set_status(infaaspublic::RequestReplyEnum::UNAVAILABLE);
        rs->set_msg("Model has not been registered");
        return Status::OK;
      }

      // Check if model is running
      // If decision mode is not INFAAS_ALL, set to 0 since we are using a naive
      // policy
      if (master_decision_ != ROUNDROBIN) {
        is_running = rm_->is_model_running(model);
      } else {
        is_running = 0;
      }
    } else if (!parent_model.empty()) {  // Parent model provided
      if (!rm_->parent_model_registered(parent_model)) {
        rs->set_status(infaaspublic::RequestReplyEnum::UNAVAILABLE);
        rs->set_msg("Parent model has not been registered");
        return Status::OK;
      }

      // Right now, assumption is that latency and accuracy are set to 0 if they
      // are unset
      int64_t latency = slo.latencyinusec();
      double accuracy = slo.minaccuracy();
      if ((latency == 0) && (accuracy == 0.0)) {
        std::cout << "No hints given, send to some available running model"
                  << std::endl;
      }

      std::vector<std::string> meets_slo =
          par_lat_search(parent_model, latency, request->raw_input().size(),
                         &is_running, master_decision_);

      // Empty means there is no model that can satisfy the request
      if (meets_slo.empty()) {
        rs->set_status(infaaspublic::RequestReplyEnum::UNAVAILABLE);
        rs->set_msg(
            "PAR: No model can satisfy this request. Try a different "
            "accuracy/latency");
        return Status::OK;
      }

      model = meets_slo[0];
    } else {  // Grandparent model provided
      if (!rm_->gparent_model_registered(grandparent_model)) {
        rs->set_status(infaaspublic::RequestReplyEnum::UNAVAILABLE);
        rs->set_msg("Grandparent model has not been registered");
        return Status::OK;
      }

      // For grandparent model, assumption is that BOTH latency and accuracy
      // must be set
      int64_t latency = slo.latencyinusec();
      double accuracy = slo.minaccuracy();
      if ((latency == 0) || (accuracy == 0.0)) {
        std::cout << "For a grandparent-only query, ";
        std::cout << "latency and accuracy are required" << std::endl;
      }

      std::vector<std::string> meets_slo = gpar_lat_acc_search(
          grandparent_model, accuracy, latency, request->raw_input().size(),
          &is_running, master_decision_);

      // Empty means there is no model that can satisfy the request
      if (meets_slo.empty()) {
        rs->set_status(infaaspublic::RequestReplyEnum::UNAVAILABLE);
        rs->set_msg(
            "GPAR: No model can satisfy this request. Try a different "
            "accuracy/latency");
        return Status::OK;
      }

      model = meets_slo[0];
    }

    // By this point, we have a model.
    bool valid_is_running = false;
    std::cout << "[LOG]: Model variant selected is: " << model << std::endl;
    if (is_running < 0) {
      // Since we have already checked if the model is registered,
      // if is_running < 0, the Redis query failed
      std::cout << "[LOG]: is_running is negative!" << std::endl;
      throw std::runtime_error("Query to Redis failed");
    } else if (is_running) {
      // By this point, we know this return value will be valid
      std::vector<std::string> dest_name = rm_->min_qps_name(model, 3);
      if (dest_name.empty()) {
        // If this happens, it means the model was shut down in the time that a
        //// decision was made. Just leave valid_is_running as false.
        std::cout
            << "[LOG]: Model was said to be running, but min_qps_name query";
        std::cout << " is empty. Leaving valid_is_running as false."
                  << std::endl;
      } else {
        std::cout << "[LOG]: " << dest_name.size()
                  << " options for running workers";
        std::cout << std::endl;

        std::cout << "[LOG]: BEFORE..." << std::endl;
        for (std::string d : dest_name) { std::cout << "\t" << d << std::endl; }

        if ((master_decision_ == GPUSHARETRIGGER) ||
            (master_decision_ == CPUBLISTCHECK)) {
          // Seed for shuffle
          uint64_t seed =
              std::chrono::system_clock::now().time_since_epoch().count();
          std::shuffle(dest_name.begin(), dest_name.end(), std::mt19937(seed));
          std::cout << "[LOG]: AFTER..." << std::endl;
          for (std::string d : dest_name) {
            std::cout << "\t" << d << std::endl;
          }
        }
        // Now walk through the candidates and select the first one that passes
        //// both blacklist checks
        for (std::string d : dest_name) {
          std::cout << "[LOG]: Checking " << d << std::endl;
          if (rm_->is_blacklisted(d)) {
            std::cout << "[LOG]: " << d << " is blacklisted.";
            continue;
          } else {
            // If a model-variant was provided but is blacklisted on the
            // requested worker,
            //// leave valid_is_running as false
            int8_t is_blisted = rm_->get_model_avglat_blacklist(d, model);
            if (is_blisted) {
              std::cout << "[LOG]: " << d << " has blacklisted " << model
                        << std::endl;
            } else {
              next_worker = d;
              valid_is_running = true;
              break;
            }
          }
        }
      }

      // If valid_is_running was false, none of the explored workers were valid.
      if (!valid_is_running) {
        std::cout
            << "[LOG]: Warning: all explored workers failed blacklist checks";
        std::cout << " or model is not running anymore." << std::endl;
      }
    }
    if (!valid_is_running) {  // Not running, make decision based on executor +
                              // decision mode
      if ((master_decision_ != ROUNDROBIN) &&
          (master_decision_ != ROUNDROBIN_STATIC)) {
        // Use this to figure out if a model needs a GPU
        std::string mv_batch = rm_->get_model_info(model, "max_batch");
        bool needs_gpu = std::stoi(mv_batch) < 64;
        std::cout << "[LOG]: Needs GPU: " << (int16_t)needs_gpu << std::endl;

        // Get workers with minimum GPU utilization
        std::vector<std::string> min_gpu = rm_->min_gpu_util_name(10);

        // Get workers with minimum CPU utilization
        std::vector<std::string> min_cpu = rm_->min_cpu_util_name(10);

        // Find intersection between both vectors. If no intersection exists,
        // use the worker with the minimum CPU utilization
        std::vector<std::string> intersection;
        if (needs_gpu) {
          std::sort(min_gpu.begin(), min_gpu.end());
          std::sort(min_cpu.begin(), min_cpu.end());

          std::set_intersection(min_gpu.begin(), min_gpu.end(), min_cpu.begin(),
                                min_cpu.end(), back_inserter(intersection));

          // Shuffle intersection vector, since it was sorted for intersecting
          //// and will always return the same values

          // Seed for shuffle
          uint64_t seed =
              std::chrono::system_clock::now().time_since_epoch().count();
          std::shuffle(intersection.begin(), intersection.end(),
                       std::mt19937(seed));
        }

        // If no GPU is needed, only consider CPU utilization
        if (!needs_gpu || intersection.empty()) {
          if (min_cpu.empty()) {
            // This is a fatal error: there is nowhere to send the request!
            throw std::runtime_error(
                "No workers available to service request!");
          }
          // Walk through CPU options
          if (min_cpu.size() == 1) {
            if (needs_gpu && (rm_->is_exec_onlycpu(min_cpu[0]))) {
              // This is a fatal error: there is nowhere to send the request!
              throw std::runtime_error(
                  "Needs GPU, but none available to service request!");
            }

            next_worker = min_cpu[0];
          } else {
            for (const std::string mc : min_cpu) {
              std::cout << "[LOG]: Min CPU, considering: " << mc << std::endl;
              if (needs_gpu && (rm_->is_exec_onlycpu(mc))) {
                std::cout << "[LOG]: Skipping " << mc
                          << " because it is CPU only";
                std::cout << std::endl;
                continue;
              }
              if (last_worker_picked_ != mc) {
                next_worker = mc;
                last_worker_picked_ = mc;
                break;
              } else {
                std::cout << "[LOG]: Skipping " << mc
                          << " because it was last picked";
                std::cout << std::endl;
              }
            }
          }
        } else {
          if (intersection.size() == 1) {
            next_worker = intersection[0];
          } else {
            // Walk through "intersected" options
            for (const std::string inter : intersection) {
              std::cout << "[LOG]: Intersection, considering: " << inter
                        << std::endl;
              if (needs_gpu && (rm_->is_exec_onlycpu(inter))) {
                std::cout << "[LOG]: Skipping " << inter
                          << " because it is CPU only";
                std::cout << std::endl;
                continue;
              }
              if (last_worker_picked_ != inter) {
                next_worker = inter;
                last_worker_picked_ = inter;
                break;
              } else {
                std::cout << "[LOG]: Skipping " << inter
                          << " because it was last picked";
                std::cout << std::endl;
              }
            }
          }
        }

        // No available worker found.
        // This likely occurred because there was more than one worker, but only
        // one GPU
        //// instance, and it was last picked.
        // Simply use last_worker_picked_.
        if (next_worker.empty()) {
          std::cout << "[LOG]: Warning: next_worker is empty. Using last worker"
                    << std::endl;
          next_worker = last_worker_picked_;
        }
      } else {
        if (master_decision_ == ROUNDROBIN_STATIC) {
          if (static_model_worker_map_.find(model) !=
              static_model_worker_map_.end()) {
            std::cout << "[LOG]: " << model << " previously queried"
                      << std::endl;
            next_worker = static_model_worker_map_[model];
          } else {
            std::cout << "[LOG]: " << model << " not seen before, using RR"
                      << std::endl;

            // Pick the next worker and increment the round robin counter
            // If a GPU is needed, walk through all workers until a GPU is found
            std::string mv_batch = rm_->get_model_info(model, "max_batch");
            bool needs_gpu = std::stoi(mv_batch) < 64;
            std::cout << "[LOG]: Needs GPU: " << (int16_t)needs_gpu
                      << std::endl;

            if (needs_gpu) {
              next_worker = all_gpu_exec_[all_exec_gpu_counter_];
              all_exec_gpu_counter_ =
                  (all_exec_gpu_counter_ + 1) % all_gpu_exec_.size();
            } else {
              next_worker = all_exec_[all_exec_counter_];
              all_exec_counter_ = (all_exec_counter_ + 1) % all_exec_.size();
            }

            static_model_worker_map_.insert(
                std::pair<std::string, std::string>(model, next_worker));
          }
        } else {
          // Get all executors
          std::vector<std::string> all_exec_ = rm_->get_all_executors();

          // Pick the next worker and increment the round robin counter
          next_worker = all_exec_[all_exec_counter_];
          all_exec_counter_ = (all_exec_counter_ + 1) % all_exec_.size();
        }
      }
    }

    dest_addr = rm_->get_executor_addr(next_worker);
    if (RedisMetadata::is_empty_address(dest_addr)) {
      rs->set_status(infaaspublic::RequestReplyEnum::INVALID);
      rs->set_msg("Destination address is empty");
      std::cout << "[LOG]: Destination address is empty" << std::endl;
      return Status::OK;
    } else {
      std::cout << "[LOG]: Model will be serviced by: ";
      std::cout << next_worker << " (";
      std::cout << RedisMetadata::Address_to_str(dest_addr) << ")" << std::endl;
    }

    // Update qps worker map
    std::chrono::time_point<std::chrono::system_clock> curr_time =
        std::chrono::system_clock::now();
    if (qps_worker_scaler_.find(next_worker) == qps_worker_scaler_.end()) {
      qps_worker_scaler_.insert(
          std::pair<
              std::string,
              std::pair<std::chrono::time_point<std::chrono::system_clock>,
                        int16_t>>(
              next_worker,
              std::pair<std::chrono::time_point<std::chrono::system_clock>,
                        int16_t>(curr_time, 1)));
    } else {
      auto time_difference =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              curr_time - qps_worker_scaler_[next_worker].first)
              .count();

      std::cout << "[LOG]: Interval: " << time_difference << std::endl;
      if (time_difference < qps_vm_scale_time_interval) {
        qps_worker_scaler_[next_worker].second++;
        std::cout << "[LOG]: Less than one second between requests"
                  << std::endl;
        if (qps_worker_scaler_[next_worker].second ==
            qps_vm_scale_query_limit) {
          std::cout << "[LOG]: Setting VM Scale, " << next_worker;
          std::cout << " is overloaded" << std::endl;
          // rm_->set_vm_scale();
          qps_worker_scaler_[next_worker].first = curr_time;
          qps_worker_scaler_[next_worker].second = 0;
          // If there is more than one worker, blacklist this worker
          if (rm_->get_num_executors() > 1) {
            if (rm_->blacklist_executor(next_worker, 2) < 0) {
              throw std::runtime_error("Failed to blacklist a worker");
            }
          }
        }
      } else {
        std::cout
            << "[LOG]: Over one second between requests; resetting counter"
            << std::endl;
        qps_worker_scaler_[next_worker].first = curr_time;
        qps_worker_scaler_[next_worker].second = 0;
      }
    }

    gettimeofday(&time2, NULL);
    printf("[queryfe_server.cc] Master decision-making total time: %.4lf ms.\n",
           ts_to_ms(time1, time2));
    fflush(stdout);

    grpc::ChannelArguments arguments;
    arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
    arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);

    infaas::internal::QueryClient query_client(grpc::CreateCustomChannel(
        RedisMetadata::Address_to_str(dest_addr),
        grpc::InsecureChannelCredentials(), arguments));
    auto worker_reply = query_client.QueryOnline(
        request->raw_input(), {model}, submitter, reply->mutable_raw_output(),
        slo.latencyinusec(), slo.minaccuracy(), slo.maxcost());
    gettimeofday(&time3, NULL);
    printf("[queryfe_server.cc] Master QueryOnline total time: %.4lf ms.\n",
           ts_to_ms(time1, time3));
    fflush(stdout);

    // For logging purposes
    std::cout << "===================================================="
              << std::endl;

    if (worker_reply.status() !=
        infaas::internal::InfaasRequestStatusEnum::SUCCESS) {
      std::cout << "[FAIL]: error msg: " << worker_reply.msg() << std::endl;
      rs->set_status(infaaspublic::RequestReplyEnum::INVALID);
      rs->set_msg("Query failed: " + worker_reply.msg());
      return Status::OK;
    }

    rs->set_status(infaaspublic::RequestReplyEnum::SUCCESS);
    rs->set_msg("Successfully executed query");
    return Status::OK;
  }

  Status QueryOffline(ServerContext* context,
                      const QueryOfflineRequest* request,
                      QueryOfflineResponse* reply) override {
    std::string input_url = request->input_url();
    std::string parent_model = request->model();
    std::string output_url = request->output_url();
    std::string submitter = request->submitter();
    double maxcost = request->maxcost();

    infaaspublic::RequestReply* rs = reply->mutable_status();

    Aws::SDKOptions options;
    Aws::InitAPI(options);
    Aws::Client::ClientConfiguration clientConfig;
    clientConfig.region = Aws::String(region.c_str());
    Aws::S3::S3Client s3_client(clientConfig);

    // Check that input bucket is valid
    std::string input_bucket, input_prefix;
    parse_s3_url(input_url, &input_bucket, &input_prefix);
    Aws::S3::Model::ListObjectsV2Request list_inp_request;
    Aws::String input_bucket_aws(input_bucket.c_str(), input_bucket.size());
    list_inp_request.SetBucket(input_bucket_aws);
    list_inp_request.SetPrefix(Aws::String(input_prefix.c_str()));

    auto list_inp_outcome = s3_client.ListObjectsV2(list_inp_request);
    if (!list_inp_outcome.IsSuccess()) {
      rs->set_status(infaaspublic::RequestReplyEnum::INVALID);
      rs->set_msg(input_bucket +
                  " is not a valid bucket or cannot be accessed by INFaaS");
      return Status::OK;
    }

    // Check that output bucket is valid
    std::string output_bucket, output_prefix;
    parse_s3_url(output_url, &output_bucket, &output_prefix);
    Aws::S3::Model::ListObjectsV2Request list_out_request;
    Aws::String output_bucket_aws(output_bucket.c_str(), output_bucket.size());
    list_out_request.SetBucket(output_bucket_aws);
    list_out_request.SetPrefix(Aws::String(output_prefix.c_str()));

    auto list_out_outcome = s3_client.ListObjectsV2(list_out_request);
    if (!list_out_outcome.IsSuccess()) {
      rs->set_status(infaaspublic::RequestReplyEnum::INVALID);
      rs->set_msg(output_bucket +
                  " is not a valid bucket or cannot be accessed by INFaaS");
      return Status::OK;
    }

    // Check that parent model is valid
    if (!rm_->parent_model_registered(parent_model)) {
      rs->set_status(infaaspublic::RequestReplyEnum::UNAVAILABLE);
      rs->set_msg("Parent model has not been registered");
      return Status::OK;
    }

    // Get all model variants and find the first one that supports preset
    // framework (framework can be configured later)
    std::vector<std::string> all_var =
        rm_->get_all_model_variants(parent_model);

    // Go through all model variants and find the first one whose framework
    // matches the preset framework
    std::string model_var = "";
    for (std::string av : all_var) {
      std::string framework = rm_->get_model_info(av, "framework");
      if (framework == offline_framework) {
        model_var = av;
        break;
      }
    }

    // Fail if no model variant was found
    if (model_var.empty()) {
      rs->set_status(infaaspublic::RequestReplyEnum::UNAVAILABLE);
      rs->set_msg("No model variant with " + offline_framework + " framework");
      return Status::OK;
    }

    // If mode is not ROUNDROBIN, find the worker with the minimum CPU
    // utilization. Otherwise, select using round-robin.
    std::string next_worker;
    if ((master_decision_ != ROUNDROBIN) &&
        (master_decision_ != ROUNDROBIN_STATIC)) {
      std::vector<std::string> min_cpu = rm_->min_cpu_util_name(10);

      // Shuffle to load-balance
      // Seed for shuffle
      uint64_t seed =
          std::chrono::system_clock::now().time_since_epoch().count();
      std::shuffle(min_cpu.begin(), min_cpu.end(), std::mt19937(seed));

      // This is a fatal error: there is nowhere to send the request!
      if (min_cpu.empty()) {
        throw std::runtime_error("No workers available to service request!");
      }

      // Walk through CPU options
      if (min_cpu.size() == 1) {
        next_worker = min_cpu[0];
      } else {
        for (const std::string mc : min_cpu) {
          if (last_worker_picked_ != mc) {
            next_worker = mc;
            last_worker_picked_ = mc;
            break;
          } else {
            std::cout << "[LOG]: Skipping " << mc
                      << " because it was last picked";
            std::cout << std::endl;
          }
        }
      }
    } else {
      if (master_decision_ == ROUNDROBIN_STATIC) {
        if (static_model_worker_map_.find(model_var) !=
            static_model_worker_map_.end()) {
          std::cout << "[LOG]: " << model_var << " previously queried"
                    << std::endl;
          next_worker = static_model_worker_map_[model_var];
        } else {
          std::cout << "[LOG]: " << model_var << " not seen before, using RR"
                    << std::endl;

          // Pick the next worker and increment the round robin counter
          // If a GPU is needed, walk through all workers until a GPU is found
          std::string mv_batch = rm_->get_model_info(model_var, "max_batch");
          bool needs_gpu = std::stoi(mv_batch) < 64;
          std::cout << "[LOG]: Needs GPU: " << (int16_t)needs_gpu << std::endl;

          if (needs_gpu) {
            next_worker = all_gpu_exec_[all_exec_gpu_counter_];
            all_exec_gpu_counter_ =
                (all_exec_gpu_counter_ + 1) % all_gpu_exec_.size();
          } else {
            next_worker = all_exec_[all_exec_counter_];
            all_exec_counter_ = (all_exec_counter_ + 1) % all_exec_.size();
          }

          static_model_worker_map_.insert(
              std::pair<std::string, std::string>(model_var, next_worker));
        }
      } else {
        // Get all executors
        std::vector<std::string> all_exec_ = rm_->get_all_executors();

        // Pick the next worker and increment the round robin counter
        next_worker = all_exec_[all_exec_counter_];
        all_exec_counter_ = (all_exec_counter_ + 1) % all_exec_.size();
      }
    }

    // Update last_worker_picked_
    last_worker_picked_ = next_worker;

    struct Address dest_addr = rm_->get_executor_addr(next_worker);

    std::cout << "[LOG]: Offline query will be serviced by: ";
    std::cout << next_worker << " (";
    std::cout << RedisMetadata::Address_to_str(dest_addr) << ")" << std::endl;

    // Forward request to worker
    grpc::ChannelArguments arguments;
    arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
    arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);

    infaas::internal::QueryClient query_client(grpc::CreateCustomChannel(
        RedisMetadata::Address_to_str(dest_addr),
        grpc::InsecureChannelCredentials(), arguments));
    auto worker_reply = query_client.QueryOffline(
        input_url, {model_var}, submitter, output_url, maxcost);

    // For logging purposes
    std::cout << "===================================================="
              << std::endl;

    if (worker_reply.status() !=
        infaas::internal::InfaasRequestStatusEnum::SUCCESS) {
      rs->set_status(infaaspublic::RequestReplyEnum::INVALID);
      rs->set_msg("Failure to start Offline job");
      return Status::OK;
    }

    rs->set_status(infaaspublic::RequestReplyEnum::SUCCESS);
    rs->set_msg("Successfully executed query");
    return Status::OK;
  }

  Status AllParentInfo(ServerContext* context, const AllParRequest* request,
                       AllParResponse* reply) override {
    infaaspublic::AllParReply* apr = reply->mutable_reply();
    infaaspublic::RequestReply rs;

    std::string task = request->task();
    std::string dataset = request->dataset();

    // Get all parents
    std::vector<std::string> all_par =
        rm_->get_all_parent_models(task, dataset);
    *(apr->mutable_all_models()) = {all_par.begin(), all_par.end()};
    rs.set_status(infaaspublic::RequestReplyEnum::SUCCESS);
    rs.set_msg("Successful all parents");
    apr->mutable_status()->CopyFrom(rs);

    return Status::OK;
  }

  Status QueryModelInfo(ServerContext* context,
                        const QueryModelInfoRequest* request,
                        QueryModelInfoResponse* reply) override {
    infaaspublic::QueryModelReply* qmir = reply->mutable_reply();
    infaaspublic::RequestReply rs;

    std::string parent_model = request->model();

    // Check that parent model is valid
    if (!rm_->parent_model_registered(parent_model)) {
      rs.set_status(infaaspublic::RequestReplyEnum::UNAVAILABLE);
      rs.set_msg("Parent model has not been registered");
      qmir->mutable_status()->CopyFrom(rs);
      return Status::OK;
    }

    // Get a parent model's children and use one of them to get the model info
    std::vector<std::string> mod_var =
        rm_->get_all_model_variants(parent_model);

    rs.set_status(infaaspublic::RequestReplyEnum::SUCCESS);
    rs.set_msg("Successful info query");
    qmir->set_img_dim(std::stoi(rm_->get_model_info(mod_var[0], "img_dim")));
    qmir->set_accuracy(rm_->get_accuracy(mod_var[0]));
    qmir->mutable_status()->CopyFrom(rs);

    // Copy all mod_var
    *(qmir->mutable_all_models()) = {mod_var.begin(), mod_var.end()};

    return Status::OK;
  }

  Status Heartbeat(ServerContext* context, const HeartbeatRequest* request,
                   HeartbeatResponse* reply) override {
    infaaspublic::RequestReply* rs = reply->mutable_status();

    if (request->status().status() != infaaspublic::RequestReplyEnum::SUCCESS) {
      rs->set_status(infaaspublic::RequestReplyEnum::INVALID);
      rs->set_msg("Invalid heartbeat request");
      return Status::OK;
    }

    rs->set_status(infaaspublic::RequestReplyEnum::SUCCESS);
    rs->set_msg("Successful heartbeat");
    return Status::OK;
  }

  // Private variables
  const struct Address redis_addr_;
  std::unique_ptr<RedisMetadata> rm_;

  std::deque<std::string> gmod_cache_;

  std::string last_worker_picked_;
  std::map<std::string, std::string> static_model_worker_map_;

  MasterDecisions master_decision_;
  int16_t all_exec_counter_;
  int16_t all_exec_gpu_counter_;
  std::vector<std::string> all_exec_;
  std::vector<std::string> all_gpu_exec_;

  std::set<std::string> cpu_blist_;

  std::map<
      std::string,
      std::pair<std::chrono::time_point<std::chrono::system_clock>, int16_t>>
      qps_worker_scaler_;
};

}  // namespace infaasqueryfe
}  // namespace infaaspublic

void RunQueryFEServer(const struct Address& redis_addr,
                      const int8_t decision_policy) {
  std::string server_address("0.0.0.0:50052");
  infaaspublic::infaasqueryfe::QueryServiceImpl service(redis_addr,
                                                        decision_policy);

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
    std::cout
        << "Usage: ./queryfe_server <redis_ip> <redis_port> <decision_policy>";
    std::cout << std::endl;
    std::cout
        << "decision_policy: 0=INFAAS_ALL, 1=INFAAS_NOQPSLAT, 2=ROUNDROBIN, ";
    std::cout << "3=ROUNDROBIN_STATIC, 4=GPUSHARETRIGGER, 5=CPUBLISTCHECK"
              << std::endl;
    std::cout << "INFAAS_ALL: Use all features of INFAAS" << std::endl;
    std::cout
        << "INFAAS_NOQPSLAT: Only ocnsider if model is running, not QPS/latency"
        << std::endl;
    std::cout << "ROUNDROBIN: Pick machines in a round-robin fashion; assumes "
                 "model-variant";
    std::cout << " is always provided." << std::endl;
    std::cout << "ROUNDROBIN_STATIC: Same as ROUNDROBIN, but if model-variant "
                 "has been";
    std::cout << " queried before, send to that worker." << std::endl;
    std::cout
        << "GPUSHARETRIGGER: Same as INFAAS_ALL but asks for a new VM when a";
    std::cout << " GPU model is blacklisted" << std::endl;
    std::cout
        << "CPUBLISTCHECK: Same as GPUSHARETRIGGER, but will not choose CPU";
    std::cout << " if a GPU model is running under the same parent"
              << std::endl;
    return 1;
  }

  const struct Address redis_addr = {argv[1], argv[2]};
  if (RedisMetadata::is_empty_address(redis_addr)) {
    std::cout << "Invalid redis server address: " << argv[1] << ":" << argv[2]
              << std::endl;
    return 1;
  }
  const int8_t decision_policy = std::stoi(argv[3]);

  RunQueryFEServer(redis_addr, decision_policy);

  return 0;
}
