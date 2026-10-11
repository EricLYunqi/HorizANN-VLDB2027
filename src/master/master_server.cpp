#include "master_server.h"
#include "query_router.h"
#include "aggregator.h"
#include "include/kmeans.h"
#include "include/vec_bytes.h"
#include "proto/ann.pb.h"
#include "proto/ann.grpc.pb.h"
#include <iostream>
#include <chrono>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <future>
#include <atomic>
#include <mutex>
#include <random>
#include <omp.h>
#include <filesystem>
#include <limits>
#include <map>
#include <cmath>
#include <numeric>

#include <yaml-cpp/yaml.h>
#include "src/config/config_loader.h"

namespace ann {

MasterServiceImpl::MasterServiceImpl(std::unique_ptr<IQueryRouter> router, std::unique_ptr<IAggregator> aggregator, std::deque<std::string> free_workers, int split_threshold, bool manual_rebalance,
                                      RebalanceMethod rebalance_method, double rebalance_eta0, double rebalance_epsilon)
    : query_router_(std::move(router)), aggregator_(std::move(aggregator)), free_workers_(std::move(free_workers)), split_threshold_(split_threshold), manual_rebalance_(manual_rebalance),
      rebalance_method_(rebalance_method), rebalance_eta0_(rebalance_eta0), rebalance_epsilon_(rebalance_epsilon) {
    std::remove("/tmp/master_batched_insert.csv");
}

std::shared_ptr<WorkerClient> MasterServiceImpl::getOrCreateWorkerClient(const std::string& worker_address) {
    // lock required for multiple concurrent requests to master from different clients
    std::lock_guard<std::mutex> lock(workers_mutex_);
    auto it = workers_.find(worker_address);
    if (it != workers_.end()) return it->second;
    auto client = std::make_shared<WorkerClient>(worker_address);
    workers_[worker_address] = client;
    return client;
}


grpc::Status MasterServiceImpl::Search(grpc::ServerContext* context, const Query* request, AggregatedResult* response) {
    
    try {

        auto route_t0 = std::chrono::steady_clock::now();
        const float* query_data = as_floats(request->vector_data());
        const int dim = static_cast<int>(request->vector_data().size() / sizeof(float));
        auto candidate_workers = query_router_->routeQuery(query_data, dim, request->k());
        auto route_t1 = std::chrono::steady_clock::now();
        const double route_ms = std::chrono::duration<double, std::milli>(route_t1 - route_t0).count();
        
        // Log routing decisions for accuracy testing
        // std::filesystem::create_directories("indexes");
        // std::ofstream routing_log("indexes/routing_decisions.csv", std::ios::app);
        // if (routing_log.tellp() == 0) {
        //     routing_log << "query_id,routed_workers\n";
        // }
        // std::string workers_str;
        // for (size_t i = 0; i < candidate_workers.size(); ++i) {
        //     if (i > 0) workers_str += ";";
        //     workers_str += candidate_workers[i].worker_address;
        // }
        // routing_log << request->query_id() << ',' << workers_str << '\n';
        
        // Write centroids once (check file existence instead of static variable)
        // std::string centroids_path = "indexes/centroids.csv";
        // if (!std::filesystem::exists(centroids_path)) {
        //     std::ofstream centroids_file(centroids_path);
        //     centroids_file << "worker_address,centroid\n";
        //     auto all_clusters = query_router_->getAllClusters();
        //     for (const auto& cluster : all_clusters) {
        //         centroids_file << cluster.worker_address << ",";
        //         // Write centroid values separated by pipe (|) to avoid CSV comma issues
        //         for (size_t j = 0; j < cluster.centroid.size(); ++j) {
        //             if (j > 0) centroids_file << "|";
        //             centroids_file << cluster.centroid[j];
        //         }
        //         centroids_file << "\n";
        //     }
        // }

        auto dispatch_t0 = std::chrono::steady_clock::now();
        std::vector<WorkerResult> worker_results = forwardSearchToWorkers(*request, candidate_workers);
        auto dispatch_t1 = std::chrono::steady_clock::now();
        const double rpc_ms = std::chrono::duration<double, std::milli>(dispatch_t1 - dispatch_t0).count();
        
        auto agg_t0 = std::chrono::steady_clock::now();
        std::vector<const WorkerResult*> wr_ptrs;
        wr_ptrs.reserve(worker_results.size());
        for (const auto& wr : worker_results) wr_ptrs.push_back(&wr);
        *response = aggregator_->aggregateResults(wr_ptrs, request->k());
        auto agg_t1 = std::chrono::steady_clock::now();
        const double agg_ms = std::chrono::duration<double, std::milli>(agg_t1 - agg_t0).count();

        const double master_compute_ms = route_ms + rpc_ms + agg_ms;  // includes RPC time too which will be later subtracted during analysis
        
        // Write per-query row to master search CSV
        // std::ofstream ms_csv("/tmp/master_search.csv", std::ios::app);
        // if (ms_csv.tellp() == 0) {
        //     ms_csv << "query_id,master_search_compute_ms,worker_rpc_ms" << '\n';
        // }
        // ms_csv << request->query_id() << ',' << master_compute_ms << ',' << rpc_ms << '\n';
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        std::cerr << "Master: Exception in Search: " << e.what() << '\n';
        return grpc::Status(grpc::StatusCode::INTERNAL, "Internal server error: " + std::string(e.what()));
    } catch (...) {
        std::cerr << "Master: Unknown exception in Search" << '\n';
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown internal server error");
    }
}

grpc::Status MasterServiceImpl::BatchedSearch(grpc::ServerContext* context, const BatchedQuery* request, BatchedAggregatedResult* response) {
    try {
        auto total_t0 = std::chrono::steady_clock::now();
        const int num_queries = request->batch_size();

        const float* all_q = as_floats(request->vectors_data());
        const int q_dim = num_queries > 0
            ? static_cast<int>(request->vectors_data().size()) / (num_queries * static_cast<int>(sizeof(float)))
            : 0;

        double routing_ms = 0.0;

        auto routing_start = std::chrono::steady_clock::now();
        std::unordered_map<std::string, std::vector<int>> worker_to_queries;
        const int router_k = request->router_k() > 0 ? request->router_k() : request->k() * 3;
        size_t total_routed_workers = 0;
            std::cout << "[Master] BatchedSearch received batch_size=" << num_queries
                      << " router_k=" << router_k
                      << (request->router_k() > 0 ? " (explicit)" : " (default)")
                      << " efsearch=";
            if (request->has_efsearch()) {
                std::cout << request->efsearch() << " (explicit)";
            } else {
                std::cout << "default";
            }
            std::cout << '\n';

        // find workers for each query (routing) — parallelized across queries
        std::vector<std::vector<ClusterInfo>> per_query_workers(num_queries);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_queries; ++i) {
            per_query_workers[i] = query_router_->routeQuery(all_q + i * q_dim, q_dim, router_k);
        }
        // merge into shared map (serial)
        for (int i = 0; i < num_queries; ++i) {
            total_routed_workers += per_query_workers[i].size();
            for (const auto& worker : per_query_workers[i]) {
                worker_to_queries[worker.worker_address].push_back(i);
            }
        }
        if (request->has_efsearch() && request->efsearch() == 999) {
            // debugging
            const double avg_workers_per_query =
                num_queries > 0 ? static_cast<double>(total_routed_workers) / num_queries : 0.0;
            std::cout << "[Master] BatchedSearch routing stats for efsearch 999: avg_workers_per_query="
                      << avg_workers_per_query << ", router_k=" << router_k << '\n';
        }
        auto routing_end = std::chrono::steady_clock::now();
        routing_ms += std::chrono::duration<double, std::milli>(routing_end - routing_start).count();

        std::vector<BatchedWorkerResult> worker_results;
        double rpc_ms = 0.0;
        auto rpc_start = std::chrono::steady_clock::now();
        std::vector<std::future<BatchedWorkerResult>> worker_futures;
        // launch async search for each worker — capture client before async to avoid lock inside lambda
        for (const auto& [worker_address, query_indices] : worker_to_queries) {
            auto client = getOrCreateWorkerClient(worker_address);
            worker_futures.emplace_back(std::async(std::launch::async, [&, client, worker_address, query_indices]() -> BatchedWorkerResult {
                BatchedQuery worker_batch;
                if (request->has_efsearch()) {
                    worker_batch.set_efsearch(request->efsearch());
                }
                worker_batch.set_k(request->k());
                worker_batch.set_batch_size(static_cast<int>(query_indices.size()));
                for (int qi : query_indices) {
                    append_float_bytes(worker_batch.mutable_vectors_data(), all_q + qi * q_dim, q_dim);
                    worker_batch.add_query_ids(qi);
                }

                BatchedWorkerResult worker_result;
                grpc::ClientContext worker_context;

                grpc::Status status = client->stub->BatchedSearch(&worker_context, worker_batch, &worker_result);

                if (status.ok()) {
                    return worker_result;
                } else {
                    return BatchedWorkerResult{};
                }
            }));
        }

        for (auto& future : worker_futures) {
            auto result = future.get();
            if (result.results_size() > 0) {
                worker_results.push_back(result);
            }
        }
        auto rpc_end = std::chrono::steady_clock::now();
        rpc_ms = std::chrono::duration<double, std::milli>(rpc_end - rpc_start).count();
               
        auto aggregation_ms = 0.0;
        auto agg_start = std::chrono::steady_clock::now();

        std::unordered_map<int32_t, std::vector<const WorkerResult*>> results_by_query_id;
        for (const auto& batched_worker_result : worker_results) {
            for (const auto& worker_result : batched_worker_result.results()) {
                results_by_query_id[worker_result.query_id()].push_back(&worker_result);
            }
        }

        for (int i = 0; i < num_queries; ++i) {
            std::vector<const WorkerResult*> ptrs;
            auto it = results_by_query_id.find(i);
            if (it != results_by_query_id.end()) ptrs = it->second;
            *response->add_results() = aggregator_->aggregateResults(ptrs, request->k());
        }

        auto agg_end = std::chrono::steady_clock::now();
        aggregation_ms = std::chrono::duration<double, std::milli>(agg_end - agg_start).count();

        auto total_t1 = std::chrono::steady_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();
        
        std::cout << "[Master] BatchedSearch timing: total_ms=" << total_ms
                  << " routing_ms=" << routing_ms
                  << " rpc_ms=" << rpc_ms
                  << " aggregation_ms=" << aggregation_ms
                  << " batch_size=" << num_queries << '\n';
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        std::cerr << "Master: Exception in BatchedSearch: " << e.what() << '\n';
        return grpc::Status(grpc::StatusCode::INTERNAL, "Internal server error: " + std::string(e.what()));
    } catch (...) {
        std::cerr << "Master: Unknown exception in BatchedSearch" << '\n';
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown internal server error");
    }
}

grpc::Status MasterServiceImpl::Insert(grpc::ServerContext* context, const InsertData* request, InsertAck* response) {
    
    try {
        auto route_t0 = std::chrono::steady_clock::now();
        const float* query_data = as_floats(request->vector_data());
        const int dim = static_cast<int>(request->vector_data().size() / sizeof(float));

        if (dim == 0) {
            std::cout << "Master: Empty vector provided, insert failed" << '\n';
            response->set_success(false);
            return grpc::Status::OK;
        }

        auto target_worker = query_router_->insertQuery(query_data, dim);

        maybeAddToReservoir(query_data, dim);

        // Bootstrap index when there's only one worker
        const size_t bootstrap_threshold = static_cast<size_t>(query_router_->getMaxNodes());
        if (bootstrap_count_ < bootstrap_threshold) {
            if (bootstrap_dim_ == 0) bootstrap_dim_ = dim;
            bootstrap_flat_.insert(bootstrap_flat_.end(), query_data, query_data + dim);
            ++bootstrap_count_;
            if (bootstrap_count_ == bootstrap_threshold) {
                (void)std::async(std::launch::async, [this]() {
                    query_router_->addVectorsToRoutingIndex(bootstrap_flat_.data(),
                                                            static_cast<int>(bootstrap_count_),
                                                            bootstrap_dim_);
                    std::cout << "Master: Built routing index with " << bootstrap_count_ << " vectors" << '\n';
                });
            }
        }
        
        auto route_t1 = std::chrono::steady_clock::now();
        const double route_ms = std::chrono::duration<double, std::milli>(route_t1 - route_t0).count();

        auto rpc_t0 = std::chrono::steady_clock::now();
        *response = forwardInsertToWorker(*request, target_worker);
        auto rpc_t1 = std::chrono::steady_clock::now();
        const double rpc_ms = std::chrono::duration<double, std::milli>(rpc_t1 - rpc_t0).count();
        
        const double master_insert_compute_ms = route_ms + rpc_ms; 

        // std::ofstream mi_csv("/tmp/master_insert.csv", std::ios::app);
        // if (mi_csv.tellp() == 0) {
        //     mi_csv << "query_id,master_insert_compute_ms,worker_rpc_ms" << '\n';
        // }
        // mi_csv << request->id() << ',' << (route_ms + rpc_ms) << ',' << rpc_ms << '\n';
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        std::cerr << "Master: Exception in Insert: " << e.what() << '\n';
        response->set_success(false);
        return grpc::Status::OK;  // RPC succeeds but insert fails
    } catch (...) {
        std::cerr << "Master: Unknown exception in Insert" << '\n';
        response->set_success(false);
        return grpc::Status::OK;  // RPC succeeds but insert fails
    }
}

grpc::Status MasterServiceImpl::BatchedInsert(grpc::ServerContext* context, const BatchedInsertData* request, BatchedInsertAck* response) {
    try {
        auto total_t0 = std::chrono::steady_clock::now();
        const int num_inserts = request->batch_size();

        static constexpr int router_k_ghost = 10;
        static constexpr int M_ghost = 16;

        // =========================================================
        // PHASE 1: Route and insert vectors into resident workers
        // =========================================================
        auto phase1_t0 = std::chrono::steady_clock::now();

        // Flat byte buffer access — derive dim once for the whole batch
        const float* all_vecs = as_floats(request->vectors_data());
        const int ins_dim = num_inserts > 0
            ? static_cast<int>(request->vectors_data().size()) / (num_inserts * static_cast<int>(sizeof(float)))
            : 0;

        if (ins_dim == 0 && num_inserts > 0) {
            std::cout << "Master: Empty vectors_data in BatchedInsert, aborting" << '\n';
            for (int i = 0; i < num_inserts; ++i) response->add_acks()->set_success(false);
            return grpc::Status::OK;
        }

        // 1a. Routing — parallelised (insertQuery is read-only on routing HNSW)
        double routing_ms = 0.0;
        std::unordered_map<std::string, std::vector<int>> worker_to_inserts;
        std::vector<int> insert_to_worker_batch_position(num_inserts, -1);
        std::vector<std::string> insert_to_worker_address(num_inserts);

        auto routing_start = std::chrono::steady_clock::now();
        {
            std::vector<ClusterInfo> routing_results(num_inserts);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_inserts; ++i) {
                routing_results[i] = query_router_->insertQuery(all_vecs + i * ins_dim, ins_dim);
            }
            for (int i = 0; i < num_inserts; ++i) {
                insert_to_worker_address[i] = routing_results[i].worker_address;
                insert_to_worker_batch_position[i] = static_cast<int>(worker_to_inserts[routing_results[i].worker_address].size());
                worker_to_inserts[routing_results[i].worker_address].push_back(i);
            }
        }
        routing_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - routing_start).count();

        // 1b+c. Reservoir + bootstrap — single serial loop (both have internal locks / thresholds)
        double reservoir_ms = 0.0, bootstrap_ms = 0.0;
        {
            auto reservoir_start = std::chrono::steady_clock::now();
            const size_t bootstrap_threshold = static_cast<size_t>(query_router_->getMaxNodes());
            for (int i = 0; i < num_inserts; ++i) {
                const float* vec = all_vecs + i * ins_dim;
                maybeAddToReservoir(vec, ins_dim);
                if (bootstrap_count_ < bootstrap_threshold) {
                    if (bootstrap_dim_ == 0) bootstrap_dim_ = ins_dim;
                    bootstrap_flat_.insert(bootstrap_flat_.end(), vec, vec + ins_dim);
                    ++bootstrap_count_;
                    if (bootstrap_count_ == bootstrap_threshold) {
                        (void)std::async(std::launch::async, [this]() {
                            query_router_->addVectorsToRoutingIndex(bootstrap_flat_.data(),
                                                                    static_cast<int>(bootstrap_count_),
                                                                    bootstrap_dim_);
                            std::cout << "Master: Built routing index with " << bootstrap_count_ << " vectors" << '\n';
                        });
                    }
                }
            }
            reservoir_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - reservoir_start).count();
        }

        // 1d. BatchedInsert RPCs to resident workers — flat byte sub-batches, client captured before async
        struct InsertFutureResult { std::string worker_address; BatchedInsertAck ack; double rpc_ms; double batch_build_ms; };
        std::vector<std::future<InsertFutureResult>> worker_futures;
        double worker_batch_build_ms = 0.0;

        auto insert_rpc_wall_start = std::chrono::steady_clock::now();
        for (const auto& [worker_address, insert_indices] : worker_to_inserts) {
            auto client = getOrCreateWorkerClient(worker_address);
            worker_futures.emplace_back(std::async(std::launch::async, [&, client, worker_address, insert_indices]() -> InsertFutureResult {
                auto batch_build_t0 = std::chrono::steady_clock::now();
                BatchedInsertData worker_batch;
                worker_batch.set_batch_size(static_cast<int>(insert_indices.size()));
                for (int insert_idx : insert_indices) {
                    append_float_bytes(worker_batch.mutable_vectors_data(), all_vecs + insert_idx * ins_dim, ins_dim);
                    worker_batch.add_ids(request->ids(insert_idx));
                }
                double batch_build_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - batch_build_t0).count();

                BatchedInsertAck worker_result;
                grpc::ClientContext worker_context;
                auto worker_rpc_t0 = std::chrono::steady_clock::now();
                grpc::Status status = client->stub->BatchedInsert(&worker_context, worker_batch, &worker_result);
                double worker_rpc_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - worker_rpc_t0).count();
                if (status.ok()) {
                    return {worker_address, worker_result, worker_rpc_ms, batch_build_ms};
                }
                std::cout << "Master failed to get batched insert results from worker " << worker_address
                          << ": " << status.error_code() << " - " << status.error_message() << '\n';
                return {worker_address, BatchedInsertAck{}, worker_rpc_ms, batch_build_ms};
            }));
        }

        struct GhostRouteResult {
            std::vector<std::vector<std::pair<std::string, int>>> per_insert_neighbors;
            double route_ms;
        };

        std::future<GhostRouteResult> ghost_route_future;
        const bool should_compute_ghost_routes = query_router_->getAllClusters().size() >= 2 && num_inserts > 0;
        if (should_compute_ghost_routes) {
            ghost_route_future = std::async(std::launch::async,
                [this, all_vecs, ins_dim, num_inserts, &insert_to_worker_address]() -> GhostRouteResult {
                    GhostRouteResult result{{}, 0.0};
                    result.per_insert_neighbors.resize(num_inserts);
                    auto ghost_route_start = std::chrono::steady_clock::now();
                    #pragma omp parallel for schedule(static)
                    for (int i = 0; i < num_inserts; ++i) {
                        const std::string& target_addr = insert_to_worker_address[i];
                        const float* vec = all_vecs + i * ins_dim;
                        auto routed = query_router_->routeQuery(vec, ins_dim, router_k_ghost);
                        for (const auto& cluster : routed) {
                            if (cluster.worker_address == target_addr) continue;
                            result.per_insert_neighbors[i].push_back({cluster.worker_address, i});
                        }
                    }
                    result.route_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - ghost_route_start).count();
                    return result;
                });
        }

        std::unordered_map<std::string, BatchedInsertAck> worker_results;
        double min_insert_rpc_ms = std::numeric_limits<double>::max();
        double max_insert_rpc_ms = 0.0, sum_insert_rpc_ms = 0.0;
        int num_insert_workers = 0;
        for (auto& future : worker_futures) {
            auto res = future.get();
            min_insert_rpc_ms = std::min(min_insert_rpc_ms, res.rpc_ms);
            max_insert_rpc_ms = std::max(max_insert_rpc_ms, res.rpc_ms);
            sum_insert_rpc_ms += res.rpc_ms;
            worker_batch_build_ms += res.batch_build_ms;
            ++num_insert_workers;
            if (res.ack.acks_size() > 0) {
                worker_results[res.worker_address] = res.ack;
            }
        }

        const double avg_insert_rpc_ms = num_insert_workers > 0 ? sum_insert_rpc_ms / num_insert_workers : 0.0;
        const double insert_rpc_wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - insert_rpc_wall_start).count();

        const double phase1_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - phase1_t0).count();

        // =========================================================
        // PHASE 2: Ghost edge search for successfully inserted vectors
        // =========================================================
        auto phase2_t0 = std::chrono::steady_clock::now();

        // Determine which inserts succeeded
        auto success_filter_start = std::chrono::steady_clock::now();
        std::vector<int> successful_insert_indices;
        successful_insert_indices.reserve(num_inserts);
        for (int i = 0; i < num_inserts; ++i) {
            if (insert_to_worker_batch_position[i] < 0) continue; // empty vector
            const auto& addr = insert_to_worker_address[i];
            int pos = insert_to_worker_batch_position[i];
            auto it = worker_results.find(addr);
            if (it != worker_results.end() && pos < it->second.acks_size() && it->second.acks(pos).success()) {
                successful_insert_indices.push_back(i);
            }
        }
        const double success_filter_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - success_filter_start).count();

        std::vector<std::vector<Neighbor>> insert_ghost_neighbors(num_inserts);
        double ghost_route_ms = 0.0, ghost_merge_ms = 0.0, ghost_rpc_ms = 0.0, ghost_aggregate_ms = 0.0;
        double ghost_route_wait_ms = 0.0;
        double ghost_rpc_build_ms = 0.0, ghost_rpc_call_ms = 0.0;
        int ghost_neighbor_workers = 0, ghost_neighbor_queries = 0;

        GhostRouteResult ghost_routes{{}, 0.0};
        if (should_compute_ghost_routes) {
            auto ghost_route_wait_start = std::chrono::steady_clock::now();
            ghost_routes = ghost_route_future.get();
            ghost_route_wait_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - ghost_route_wait_start).count();
            ghost_route_ms = ghost_routes.route_ms;
        }

        if (should_compute_ghost_routes && !successful_insert_indices.empty()) {
            const int num_successful = static_cast<int>(successful_insert_indices.size());

            // Merge into neighbor_worker -> idx map
            // e.g., W2 -> inserts [0, 3, 9]
            auto ghost_merge_start = std::chrono::steady_clock::now();
            std::unordered_map<std::string, std::vector<int>> neighbor_worker_to_inserts;
            for (int i : successful_insert_indices) {
                for (const auto& [addr, idx] : ghost_routes.per_insert_neighbors[i]) {
                    neighbor_worker_to_inserts[addr].push_back(idx);
                    ++ghost_neighbor_queries;
                }
            }
            ghost_neighbor_workers = static_cast<int>(neighbor_worker_to_inserts.size());
            ghost_merge_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - ghost_merge_start).count();

            // BatchedSearch RPCs to neighboring workers
            // M_ghost is 16, so for the neighboring workers, given a vector i, 
            // search the M_ghost nearest neighbors
            struct GhostFutResult { std::vector<std::pair<int, WorkerResult>> results; double build_ms; double rpc_ms; };
            std::vector<std::future<GhostFutResult>> ghost_futures;
            ghost_futures.reserve(neighbor_worker_to_inserts.size());

            auto ghost_rpc_start = std::chrono::steady_clock::now();
            for (const auto& [nbr_addr, indices] : neighbor_worker_to_inserts) {
                auto ghost_client = getOrCreateWorkerClient(nbr_addr);
                ghost_futures.push_back(std::async(std::launch::async,
                    [this, ghost_client, nbr_addr, indices, all_vecs, ins_dim]() -> GhostFutResult {
                        auto build_t0 = std::chrono::steady_clock::now();
                        BatchedQuery bq;
                        bq.set_k(M_ghost);
                        bq.set_batch_size(static_cast<int>(indices.size()));
                        for (int idx : indices) {
                            append_float_bytes(bq.mutable_vectors_data(), all_vecs + idx * ins_dim, ins_dim);
                            bq.add_query_ids(idx);
                        }
                        double build_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - build_t0).count();

                        BatchedWorkerResult bwr;
                        grpc::ClientContext ctx;
                        auto rpc_t0 = std::chrono::steady_clock::now();
                        grpc::Status s = ghost_client->stub->BatchedSearch(&ctx, bq, &bwr);
                        double rpc_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - rpc_t0).count();
                        GhostFutResult out{{}, build_ms, rpc_ms};
                        if (!s.ok()) return out;
                        for (auto& wr : *bwr.mutable_results()) {
                            int insert_idx = wr.query_id();
                            out.results.push_back({insert_idx, std::move(wr)});
                        }
                        return out;
                    }));
            }

            // Collect per-insert WorkerResults from all neighboring workers
            std::vector<std::vector<WorkerResult>> ghost_results_by_insert(num_inserts);
            for (auto& f : ghost_futures) {
                auto res = f.get();
                ghost_rpc_build_ms = std::max(ghost_rpc_build_ms, res.build_ms);
                ghost_rpc_call_ms  = std::max(ghost_rpc_call_ms,  res.rpc_ms);
                for (auto& [idx, wr] : res.results) {
                    ghost_results_by_insert[idx].push_back(std::move(wr));
                }
            }
            ghost_rpc_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - ghost_rpc_start).count();

            // Aggregate ghost neighbors per successful insert
            // each neighboring worker returns M_ghost vectors
            // the aggregator makes a global M_ghost vectors finally
            // in the insert_ghost_neighbors
            auto ghost_aggregate_start = std::chrono::steady_clock::now();
            #pragma omp parallel for schedule(static)
            for (int si = 0; si < num_successful; ++si) {
                int i = successful_insert_indices[si];
                if (ghost_results_by_insert[i].empty()) continue;
                std::vector<const WorkerResult*> ptrs;
                ptrs.reserve(ghost_results_by_insert[i].size());
                for (const auto& wr : ghost_results_by_insert[i]) ptrs.push_back(&wr);
                auto agg = aggregator_->aggregateResults(ptrs, M_ghost);
                insert_ghost_neighbors[i].assign(agg.topk().begin(), agg.topk().end());
            }
            ghost_aggregate_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - ghost_aggregate_start).count();
        }

        const double phase2_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - phase2_t0).count();

        // =========================================================
        // PHASE 3: Send ghost edges back to resident workers
        // =========================================================
        auto phase3_t0 = std::chrono::steady_clock::now();

        // Group ghost edges by the resident worker that owns each inserted vector
        // W1 -> [
        //     inserted_id=id0, ghost_neighbors=[...],
        //     inserted_id=id3, ghost_neighbors=[...]
        // ]
        auto ghost_send_group_start = std::chrono::steady_clock::now();
        std::unordered_map<std::string, BatchedAddGhostEdgesRequest> worker_to_ghost_req;
        int ghost_send_workers = 0, ghost_send_entries = 0;
        for (int i : successful_insert_indices) {
            if (insert_ghost_neighbors[i].empty()) continue;
            const std::string& addr = insert_to_worker_address[i];
            auto* entry = worker_to_ghost_req[addr].add_entries();
            entry->set_inserted_id(request->ids(i));
            for (const auto& nb : insert_ghost_neighbors[i]) {
                *entry->add_ghost_neighbors() = nb;
            }
            ++ghost_send_entries;
        }
        ghost_send_workers = static_cast<int>(worker_to_ghost_req.size());
        const double ghost_send_group_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ghost_send_group_start).count();

        // AddGhostEdges RPCs to resident workers (parallel)
        auto ghost_send_rpc_start = std::chrono::steady_clock::now();
        std::vector<std::future<bool>> ghost_send_futures;
        ghost_send_futures.reserve(worker_to_ghost_req.size());
        for (auto& [addr, ghost_req] : worker_to_ghost_req) {
            auto ghost_send_client = getOrCreateWorkerClient(addr);
            ghost_send_futures.push_back(std::async(std::launch::async,
                [ghost_send_client, ghost_req = std::move(ghost_req)]() mutable -> bool {
                    BatchedAddGhostEdgesAck ack;
                    grpc::ClientContext ctx;
                    grpc::Status st = ghost_send_client->stub->AddGhostEdges(&ctx, ghost_req, &ack);
                    return st.ok() && ack.success();
                }));
        }
        for (auto& f : ghost_send_futures) f.get();
        const double ghost_send_rpc_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ghost_send_rpc_start).count();

        const double phase3_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - phase3_t0).count();

        // =========================================================
        // POST: Worker count update, rebalance, response construction
        // =========================================================
        auto post_t0 = std::chrono::steady_clock::now();

        double worker_count_update_ms = 0.0, split_ms = 0.0, pb_construction_ms = 0.0;
        bool should_rebalance = false;

        auto worker_count_update_start = std::chrono::steady_clock::now();
        for (const auto& [worker_address, ack] : worker_results) {
            int successes = 0;
            for (int i = 0; i < ack.acks_size(); ++i) {
                if (ack.acks(i).success()) successes++;
            }
            if (successes > 0) {
                auto client = getOrCreateWorkerClient(worker_address);
                client->node_count += successes;
                if (!manual_rebalance_ && client->node_count > split_threshold_ && !free_workers_.empty()) {
                    should_rebalance = true;
                }
            }
        }
        worker_count_update_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - worker_count_update_start).count();
        if (should_rebalance) {
            auto split_start = std::chrono::steady_clock::now();
            triggerRebalance();
            split_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - split_start).count();
            std::cout << "Master: Rebalanced workers in " << split_ms << " ms" << '\n';
        }

        auto pb_add_start = std::chrono::steady_clock::now();
        for (int i = 0; i < num_inserts; ++i) {
            const std::string& worker_address = insert_to_worker_address[i];
            int position_in_batch = insert_to_worker_batch_position[i];
            auto it = worker_results.find(worker_address);
            if (it != worker_results.end() && position_in_batch >= 0 && position_in_batch < it->second.acks_size()) {
                *response->add_acks() = it->second.acks(position_in_batch);
            } else {
                InsertAck* failed_ack = response->add_acks();
                failed_ack->set_success(false);
            }
        }
        pb_construction_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - pb_add_start).count();

        const double post_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - post_t0).count();

        const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_t0).count();

        static std::atomic<uint64_t> batched_insert_log_counter{0};
        const uint64_t batch_idx = batched_insert_log_counter.fetch_add(1) + 1;

        if (batch_idx == 1) {
            std::ofstream csv("/tmp/master_batched_insert.csv");
            csv << "batch_idx,batch_size,total_ms,phase1_ms,phase2_ms,phase3_ms,post_ms,"
                << "routing_ms,reservoir_ms,bootstrap_ms,batch_build_ms,rpc_wall_ms,"
                << "rpc_min_ms,rpc_avg_ms,rpc_max_ms,insert_workers,"
                << "filter_ms,ghost_route_ms,ghost_route_wait_ms,ghost_merge_ms,ghost_rpc_ms,ghost_rpc_build_ms,ghost_rpc_call_ms,ghost_aggregate_ms,"
                << "neighbor_workers,neighbor_queries,"
                << "ghost_send_group_ms,ghost_send_rpc_ms,ghost_send_workers,ghost_send_entries,"
                << "count_update_ms,split_ms,pb_construction_ms\n";
        }
        {
            std::ofstream csv("/tmp/master_batched_insert.csv", std::ios::app);
            csv << batch_idx << ',' << num_inserts << ',' << total_ms << ','
                << phase1_ms << ',' << phase2_ms << ',' << phase3_ms << ',' << post_ms << ','
                << routing_ms << ',' << reservoir_ms << ',' << bootstrap_ms << ','
                << worker_batch_build_ms << ',' << insert_rpc_wall_ms << ','
                << min_insert_rpc_ms << ',' << avg_insert_rpc_ms << ',' << max_insert_rpc_ms << ',' << num_insert_workers << ','
                << success_filter_ms << ',' << ghost_route_ms << ',' << ghost_route_wait_ms << ',' << ghost_merge_ms << ',' << ghost_rpc_ms << ',' << ghost_rpc_build_ms << ',' << ghost_rpc_call_ms << ',' << ghost_aggregate_ms << ','
                << ghost_neighbor_workers << ',' << ghost_neighbor_queries << ','
                << ghost_send_group_ms << ',' << ghost_send_rpc_ms << ',' << ghost_send_workers << ',' << ghost_send_entries << ','
                << worker_count_update_ms << ',' << split_ms << ',' << pb_construction_ms << '\n';
        }

        if ((batch_idx % 10000) == 0) {
            std::cout << "[Master] BatchedInsert[" << batch_idx << "]"
                      << " total_ms=" << total_ms
                      << " phase1_ms=" << phase1_ms
                      << " phase2_ms=" << phase2_ms
                      << " phase3_ms=" << phase3_ms
                      << " post_ms=" << post_ms
                      << " batch_size=" << num_inserts
                      << '\n'
                      << "  Phase1(insert): routing_ms=" << routing_ms
                      << " reservoir_ms=" << reservoir_ms
                      << " bootstrap_ms=" << bootstrap_ms
                      << " batch_build_ms=" << worker_batch_build_ms
                      << " rpc_wall_ms=" << insert_rpc_wall_ms
                      << " rpc_min_ms=" << min_insert_rpc_ms
                      << " rpc_avg_ms=" << avg_insert_rpc_ms
                      << " rpc_max_ms=" << max_insert_rpc_ms
                      << " workers=" << num_insert_workers
                      << '\n'
                      << "  Phase2(ghost search): filter_ms=" << success_filter_ms
                      << " ghost_route_ms=" << ghost_route_ms
                      << " ghost_route_wait_ms=" << ghost_route_wait_ms
                      << " ghost_merge_ms=" << ghost_merge_ms
                      << " ghost_rpc_ms=" << ghost_rpc_ms
                      << " ghost_rpc_request_build_ms (max)=" << ghost_rpc_build_ms
                      << " ghost_rpc_call_ms (max)=" << ghost_rpc_call_ms
                      << " ghost_aggregate_ms=" << ghost_aggregate_ms
                      << " neighbor_workers=" << ghost_neighbor_workers
                      << " neighbor_queries=" << ghost_neighbor_queries
                      << '\n'
                      << "  Phase3(ghost send): group_ms=" << ghost_send_group_ms
                      << " rpc_ms=" << ghost_send_rpc_ms
                      << " workers=" << ghost_send_workers
                      << " entries=" << ghost_send_entries
                      << '\n'
                      << "  Post: count_update_ms=" << worker_count_update_ms
                      << " split_ms=" << split_ms
                      << " pb_construction_ms=" << pb_construction_ms
                      << '\n';
        }

        return grpc::Status::OK;

    } catch (const std::exception& e) {
        std::cerr << "Master: Exception in BatchedInsert: " << e.what() << '\n';
        return grpc::Status(grpc::StatusCode::INTERNAL, "Internal server error: " + std::string(e.what()));
    } catch (...) {
        std::cerr << "Master: Unknown exception in BatchedInsert" << '\n';
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown internal server error");
    }
}

std::vector<WorkerResult> MasterServiceImpl::forwardSearchToWorkers(const Query& query, const std::vector<ClusterInfo>& workers) {
    std::vector<WorkerResult> worker_results;
    worker_results.reserve(workers.size());

    std::vector<std::future<std::optional<WorkerResult>>> futures;
    futures.reserve(workers.size());

    for (const auto& cluster : workers) {
        auto client = getOrCreateWorkerClient(cluster.worker_address);
        futures.emplace_back(std::async(std::launch::async, [client, &query, cluster]() -> std::optional<WorkerResult> {
            try {

                WorkerResult worker_result;
                grpc::ClientContext worker_context;
                worker_context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(5000));
                grpc::Status status = client->stub->Search(&worker_context, query, &worker_result);
                
                if (status.ok()) {
                    return worker_result;
                }
                std::cout << "Master failed to get results from worker " << cluster.cluster_id
                          << ": " << status.error_code() << " - " << status.error_message() << '\n';
                return std::nullopt;
            } catch (const std::exception& e) {
                std::cerr << "Master: Exception calling worker " << cluster.cluster_id << ": " << e.what() << '\n';
                return std::nullopt;
            } catch (...) {
                std::cerr << "Master: Unknown exception calling worker " << cluster.cluster_id << '\n';
                return std::nullopt;
            }
        }));
    }

    for (auto& f : futures) {
        try {
            auto opt = f.get();
            if (opt.has_value()) worker_results.push_back(std::move(opt.value()));
        } catch (const std::exception& e) {
            std::cerr << "Master: Exception joining worker future: " << e.what() << '\n';
        } catch (...) {
            std::cerr << "Master: Unknown exception joining worker future" << '\n';
        }
    }

    return worker_results;
}


InsertAck MasterServiceImpl::forwardInsertToWorker(const InsertData& data, const ClusterInfo& cluster) {
    
    auto client = getOrCreateWorkerClient(cluster.worker_address);
    
    InsertAck insert_ack;
    grpc::ClientContext worker_context;
    worker_context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(5000));
    grpc::Status status = client->stub->Insert(&worker_context, data, &insert_ack);
    if (status.ok()) {
        // std::cout << "Master got insert response from worker " << cluster.cluster_id 
        //           << ": " << (insert_ack.success() ? "success" : "failed") << '\n';
    } else {
        std::cout << "Master failed to insert to worker " << cluster.cluster_id 
                  << ": " << status.error_code() << " - " << status.error_message() << '\n';
        insert_ack.set_success(false);
    }
    
    return insert_ack;
}

grpc::Status MasterServiceImpl::AddWorker(grpc::ServerContext* context, const AddWorkerRequest* request, AddWorkerResponse* response) {
    std::cout << "Master: AddWorker RPC received" << '\n';

    if (free_workers_.empty()) {
        response->set_success(false);
        response->set_message("No free workers available");
        std::cout << "Master: AddWorker failed - no free workers available" << '\n';
        return grpc::Status::OK;
    }

    try {
        triggerRebalance();
        response->set_success(true);
        response->set_message("Worker added and rebalance completed");
        std::cout << "Master: AddWorker succeeded - rebalance completed" << '\n';
    } catch (const std::exception& e) {
        response->set_success(false);
        response->set_message(std::string("Rebalance failed: ") + e.what());
        std::cerr << "Master: AddWorker failed - " << e.what() << '\n';
    }

    return grpc::Status::OK;
}

grpc::Status MasterServiceImpl::GetClusterEdgeStats(grpc::ServerContext* context,
                                                     const GetClusterEdgeStatsRequest* request,
                                                     GetClusterEdgeStatsResponse* response) {
    (void)context;
    (void)request;

    std::vector<ClusterInfo> clusters = query_router_->getAllClusters();

    struct StatsResult {
        bool ok;
        std::string worker_address;
        EdgeStatsResponse stats;
    };

    std::vector<std::future<StatsResult>> futures;
    futures.reserve(clusters.size());
    for (const auto& cluster : clusters) {
        auto wc = getOrCreateWorkerClient(cluster.worker_address);
        futures.push_back(std::async(std::launch::async,
            [wc, worker_address = cluster.worker_address]() -> StatsResult {
                GetEdgeStatsRequest req;
                EdgeStatsResponse resp;
                grpc::ClientContext ctx;
                grpc::Status st = wc->stub->GetEdgeStats(&ctx, req, &resp);
                if (!st.ok()) {
                    std::cerr << "Master: GetEdgeStats failed for " << worker_address
                              << ": " << st.error_message() << '\n';
                    return {false, worker_address, resp};
                }
                return {true, worker_address, resp};
            }));
    }

    int64_t total_element_count = 0, total_normal_edges = 0;
    int64_t total_ghost_edges = 0, total_ghost_linked_vectors = 0;
    for (auto& f : futures) {
        StatsResult result = f.get();
        if (!result.ok) continue;

        WorkerEdgeStats* per_worker = response->add_per_worker();
        per_worker->set_worker_address(result.worker_address);
        *per_worker->mutable_stats() = result.stats;

        total_element_count       += result.stats.element_count();
        total_normal_edges        += result.stats.normal_edge_count();
        total_ghost_edges         += result.stats.ghost_edge_count();
        total_ghost_linked_vectors += result.stats.ghost_linked_vector_count();
    }

    EdgeStatsResponse* totals = response->mutable_totals();
    totals->set_element_count(total_element_count);
    totals->set_normal_edge_count(total_normal_edges);
    totals->set_ghost_edge_count(total_ghost_edges);
    totals->set_ghost_linked_vector_count(total_ghost_linked_vectors);

    std::cout << "Master: GetClusterEdgeStats - workers=" << response->per_worker_size()
              << " total_element_count=" << total_element_count
              << " total_normal_edges=" << total_normal_edges
              << " total_ghost_edges=" << total_ghost_edges
              << " total_ghost_linked_vectors=" << total_ghost_linked_vectors
              << '\n';

    return grpc::Status::OK;
}

void MasterServiceImpl::rebalanceWorkers() {
    std::cout << "\n===== REBALANCE TRIGGERED =====" << '\n';

    std::string new_worker_addr = getFreeWorkerAddress();
    if (new_worker_addr.empty()) {
        std::cout << "Master: No free workers available for rebalance" << '\n';
        return;
    }
    std::cout << "Master: New worker for rebalance: " << new_worker_addr << '\n';

    {
        std::lock_guard<std::mutex> lock(workers_mutex_);
        if (workers_.find(new_worker_addr) == workers_.end()) {
            workers_[new_worker_addr] = std::make_shared<WorkerClient>(new_worker_addr);
        }
    }

    auto existing_clusters = query_router_->getAllClusters();
    const int num_existing = static_cast<int>(existing_clusters.size());
    const int k = num_existing + 1;

    std::cout << "Master: " << num_existing << " existing clusters, will create " << k
              << " balanced partitions" << '\n';

    // Collect samples for K-means++ from reservoir if available; otherwise fall back to on-demand sampling.
    // Reservoir path: use flat buffer directly. Fallback: build vector<vector<float>> from worker samples.
    std::vector<float>              reservoir_snapshot;
    int                             reservoir_snapshot_dim = 0;
    size_t                          reservoir_snapshot_n   = 0;
    std::vector<std::vector<float>> fallback_points; // used only when reservoir is empty
    {
        std::lock_guard<std::mutex> lock(reservoir_mutex_);
        if (!reservoir_flat_.empty() && reservoir_dim_ > 0) {
            reservoir_snapshot     = reservoir_flat_;  // single flat copy
            reservoir_snapshot_dim = reservoir_dim_;
            reservoir_snapshot_n   = reservoir_flat_.size() / static_cast<size_t>(reservoir_dim_);
            std::cout << "Master: Using " << reservoir_snapshot_n
                      << " reservoir samples for rebalance" << '\n';
        }
    }

    if (reservoir_snapshot.empty()) {
        // Fallback: sample points from workers
        const int sample_size = 5000;
        auto sample_t0 = std::chrono::steady_clock::now();
        std::unordered_map<int, std::vector<std::vector<float>>> cluster_samples;
        for (const auto& cluster : existing_clusters) {
            auto worker_client = getOrCreateWorkerClient(cluster.worker_address);
            SamplePointsRequest sample_req;
            sample_req.set_num_points(sample_size);
            grpc::ClientContext ctx;
            auto reader = worker_client->stub->SamplePoints(&ctx, sample_req);

            std::vector<std::vector<float>> sampled_points;
            SamplePointsChunk chunk;
            while (reader->Read(&chunk)) {
                for (const auto& point : chunk.points()) {
                    const float* pv = as_floats(point.vector_data());
                    const int    pd = static_cast<int>(point.vector_data().size() / sizeof(float));
                    sampled_points.emplace_back(pv, pv + pd);
                }
                if (chunk.is_last_chunk()) break;
            }

            grpc::Status status = reader->Finish();
            if (status.ok() && !sampled_points.empty()) {
                cluster_samples[cluster.cluster_id] = std::move(sampled_points);
                std::cout << "Master: Sampled " << cluster_samples[cluster.cluster_id].size()
                          << " points from cluster " << cluster.cluster_id
                          << " (" << cluster.worker_address << ")" << '\n';
            } else {
                std::cout << "Master: Failed to sample from cluster " << cluster.cluster_id
                          << ": " << (status.ok() ? "no points" : status.error_message()) << '\n';
            }
        }

        auto sample_t1 = std::chrono::steady_clock::now();
        double sample_ms = std::chrono::duration<double, std::milli>(sample_t1 - sample_t0).count();
        std::cout << "Master: Sampling took " << sample_ms << " ms" << '\n';

        if (cluster_samples.empty()) {
            std::cout << "Master: No samples collected, aborting rebalance" << '\n';
            free_workers_.push_back(new_worker_addr);
            return;
        }

        for (const auto& [cid, points] : cluster_samples) {
            for (const auto& p : points) fallback_points.push_back(p);
        }
    }

    const size_t total_sample_n = reservoir_snapshot.empty()
        ? fallback_points.size() : reservoir_snapshot_n;
    std::cout << "Master: Running soft balanced K-means on "
              << total_sample_n << " sampled points into " << k
              << " clusters" << '\n';

    // Run balanced K-means on sampled points to get k new balanced clusters
    kmeans::BalancedKMeans<float> bal_km(k, /*max_iterations=*/50);
    auto kmeans_t0 = std::chrono::steady_clock::now();
    if (!reservoir_snapshot.empty()) {
        bal_km.fit(reservoir_snapshot.data(), static_cast<int>(reservoir_snapshot_n), reservoir_snapshot_dim);
    } else {
        bal_km.fit(fallback_points);
    }
    auto kmeans_t1 = std::chrono::steady_clock::now();
    double kmeans_ms = std::chrono::duration<double, std::milli>(kmeans_t1 - kmeans_t0).count();
    std::cout << "Master: Balanced K-means took " << kmeans_ms << " ms" << '\n';

    const auto& new_centroids = bal_km.get_centroids();      // size k
    const auto& assignments   = bal_km.get_assignments();    // size all_sampled_points
    if (new_centroids.empty()) {
        std::cout << "Master: Balanced K-means produced no centroids, aborting rebalance"
                  << '\n';
        free_workers_.push_back(new_worker_addr);
        return;
    }

    const int dim = static_cast<int>(new_centroids[0].size());

    // Report cluster sizes
    auto cluster_sizes = bal_km.get_cluster_sizes();
    std::cout << "Balanced K-means cluster sizes: ";
    for (int i = 0; i < static_cast<int>(cluster_sizes.size()); ++i) {
        std::cout << "c" << i << "=" << cluster_sizes[i] << " ";
    }
    std::cout << '\n';

    // --- Assign new centers to machines ---
    // Match each existing machine to one new centroid; the remaining centroid goes to the new worker.

    // Collect old centroids of existing clusters
    std::vector<std::vector<float>> old_centroids;
    old_centroids.reserve(num_existing);
    for (const auto& cluster : existing_clusters) {
        old_centroids.push_back(cluster.centroid);
    }

    auto centroid_dist_sq = [dim](const std::vector<float>& a,
                                  const std::vector<float>& b) -> float {
        float s = 0.f;
        for (int d = 0; d < dim; ++d) {
            float diff = a[d] - b[d];
            s += diff * diff;
        }
        return s;
    };

    // Greedy one-to-one matching between old_centroids (size num_existing)
    // and new_centroids (size k).
    std::vector<int> centroid_for_worker(num_existing, -1); // index in [0, k)
    std::vector<bool> centroid_used(k, false);
    for (int match = 0; match < num_existing; ++match) {
        float best = std::numeric_limits<float>::max();
        int best_w = -1;
        int best_c = -1;
        for (int wi = 0; wi < num_existing; ++wi) {
            if (centroid_for_worker[wi] != -1) continue; // already matched
            for (int cj = 0; cj < k; ++cj) {
                if (centroid_used[cj]) continue;
                float d = centroid_dist_sq(old_centroids[wi], new_centroids[cj]);
                if (d < best) {
                    best = d;
                    best_w = wi;
                    best_c = cj;
                }
            }
        }
        if (best_w == -1 || best_c == -1) {
            std::cout << "Master: Failed to assign centroid in greedy matching, aborting rebalance"
                      << '\n';
            free_workers_.push_back(new_worker_addr);
            return;
        }
        centroid_for_worker[best_w] = best_c;
        centroid_used[best_c] = true;
    }

    int new_worker_centroid_idx = -1;
    for (int cj = 0; cj < k; ++cj) {
        if (!centroid_used[cj]) {
            new_worker_centroid_idx = cj;
            break;
        }
    }
    if (new_worker_centroid_idx == -1) {
        std::cout << "Master: Could not find remaining centroid for new worker, aborting rebalance"
                  << '\n';
        free_workers_.push_back(new_worker_addr);
        return;
    }

    std::cout << "Centroid-to-worker mapping after balanced K-means:" << '\n';
    for (int wi = 0; wi < num_existing; ++wi) {
        std::cout << "  Existing cluster " << existing_clusters[wi].cluster_id
                  << " (" << existing_clusters[wi].worker_address
                  << ") <- centroid " << centroid_for_worker[wi] << '\n';
    }
    std::cout << "  New worker (" << new_worker_addr
              << ") <- centroid " << new_worker_centroid_idx << '\n';

    // Build centroid -> (cluster_id, worker_address) mapping
    std::vector<int> centroid_to_cluster_id(k, -1);
    std::vector<std::string> centroid_to_worker_addr(k);

    for (int wi = 0; wi < num_existing; ++wi) {
        int cidx = centroid_for_worker[wi];
        centroid_to_cluster_id[cidx] = existing_clusters[wi].cluster_id;
        centroid_to_worker_addr[cidx] = existing_clusters[wi].worker_address;
    }

    int new_cluster_id = 0;
    int max_cluster_id = -1;
    for (const auto& c : existing_clusters) {
        if (c.cluster_id > max_cluster_id) max_cluster_id = c.cluster_id;
    }
    new_cluster_id = max_cluster_id + 1;
    centroid_to_cluster_id[new_worker_centroid_idx] = new_cluster_id;
    centroid_to_worker_addr[new_worker_centroid_idx] = new_worker_addr;

    // Flatten centroids once for all workers
    std::vector<float> centroids_flat;
    centroids_flat.reserve(static_cast<size_t>(k) * dim);
    for (int cj = 0; cj < k; ++cj) {
        centroids_flat.insert(centroids_flat.end(),
                              new_centroids[cj].begin(),
                              new_centroids[cj].end());
    }

    // Send Rebalance to all EXISTING workers in parallel
    //    Each worker splits its index and streams sub-indexes according to
    //    the global centroid-to-worker mapping above.
    struct RebalanceResult {
        bool ok;
        std::string worker_address;
    };
    auto rebalance_t0 = std::chrono::steady_clock::now();
    auto rpc_t0 = std::chrono::steady_clock::now();
    std::cout << "\n--- Rebalance: Split & Transfer ---" << '\n';
    std::vector<RebalanceResult> rebalance_results(num_existing);
    {
        std::vector<std::future<RebalanceResult>> futures;
        for (int wi = 0; wi < num_existing; ++wi) {
            const auto& cluster = existing_clusters[wi];
            int my_centroid_idx = centroid_for_worker[wi];
            auto wc = getOrCreateWorkerClient(cluster.worker_address);

            futures.emplace_back(std::async(std::launch::async,
                [wc, &centroids_flat, &centroid_to_worker_addr, dim, k, my_centroid_idx, cluster]() -> RebalanceResult {

                    RebalanceRequest req;
                    set_float_bytes(req.mutable_centroids_data(), centroids_flat.data(),
                                    static_cast<int>(centroids_flat.size()));
                    req.set_num_centroids(k);
                    req.set_dimension(dim);
                    req.set_assigned_centroid_idx(my_centroid_idx);

                    for (int cidx = 0; cidx < k; ++cidx) {
                        auto* m = req.add_centroid_to_worker();
                        m->set_centroid_idx(cidx);
                        m->set_worker_address(centroid_to_worker_addr[cidx]);
                    }

                    RebalanceResponse resp;
                    grpc::ClientContext ctx;
                    grpc::Status st = wc->stub->Rebalance(&ctx, req, &resp);
                    if (st.ok() && resp.success()) {
                        std::cout << "Master: Rebalance OK for " << cluster.worker_address
                                  << '\n';
                        return {true, cluster.worker_address};
                    }
                    std::cout << "Master: Rebalance FAILED for " << cluster.worker_address
                              << ": " << (st.ok() ? "worker error" : st.error_message()) << '\n';
                    return {false, cluster.worker_address};
                }));
        }

        bool all_ok = true;
        for (int wi = 0; wi < num_existing; ++wi) {
            rebalance_results[wi] = futures[wi].get();
            if (!rebalance_results[wi].ok) all_ok = false;
        }
        if (!all_ok) {
            std::cout << "Master: Some workers failed rebalance" << '\n';
        }
    }
    auto rpc_t1 = std::chrono::steady_clock::now();
    double rpc_ms = std::chrono::duration<double, std::milli>(rpc_t1 - rpc_t0).count();
    std::cout << "Master: Split & Transfer RPCs took " << rpc_ms << " ms" << '\n';

    if (std::all_of(rebalance_results.begin(), rebalance_results.end(),
                    [](const RebalanceResult& result) { return result.ok; })) {
        std::vector<std::string> workers_to_query;
        workers_to_query.reserve(num_existing + 1);
        for (const auto& cluster : existing_clusters) {
            workers_to_query.push_back(cluster.worker_address);
        }
        workers_to_query.push_back(new_worker_addr);

        struct CountResult {
            bool ok;
            std::string worker_address;
            int64_t element_count;
        };

        std::vector<std::future<CountResult>> count_futures;
        count_futures.reserve(workers_to_query.size());
        for (const auto& worker_address : workers_to_query) {
            auto wc_count = getOrCreateWorkerClient(worker_address);
            count_futures.emplace_back(std::async(std::launch::async,
                [wc_count, worker_address]() -> CountResult {
                    GetCurrentElementCountRequest req;
                    GetCurrentElementCountResponse resp;
                    grpc::ClientContext ctx;
                    grpc::Status st = wc_count->stub->GetCurrentElementCount(&ctx, req, &resp);
                    if (st.ok()) {
                        std::cout << "Master: Stable count for " << worker_address
                                  << " = " << resp.element_count() << '\n';
                        return {true, worker_address, resp.element_count()};
                    }
                    std::cout << "Master: Count query failed for " << worker_address
                              << ": " << st.error_message() << '\n';
                    return {false, worker_address, 0};
                }));
        }

        std::vector<CountResult> count_results;
        count_results.reserve(count_futures.size());
        for (auto& future : count_futures) {
            count_results.push_back(future.get());
        }

        std::lock_guard<std::mutex> lock(workers_mutex_);
        for (const auto& result : count_results) {
            if (!result.ok) continue;
            auto it = workers_.find(result.worker_address);
            if (it != workers_.end()) {
                it->second->node_count = result.element_count;
            }
        }
    } else {
        std::cout << "Master: Skipping stable count refresh because rebalance did not succeed on every worker"
                  << '\n';
    }
    auto rebalance_t1 = std::chrono::steady_clock::now();
    double rebalance_ms = std::chrono::duration<double, std::milli>(rebalance_t1 - rebalance_t0).count();
    std::cout << "Master: Rebalance (including final merge) took " << rebalance_ms << " ms" << '\n';

    auto update_query_router_t0 = std::chrono::steady_clock::now();

    // Update query router: overwrite centroids for existing clusters with their
    // newly assigned balanced centroids, and add the new cluster for the new worker.
    for (int wi = 0; wi < num_existing; ++wi) {
        int cidx = centroid_for_worker[wi];
        query_router_->addCluster(
            existing_clusters[wi].cluster_id,
            new_centroids[cidx],
            existing_clusters[wi].worker_address);
    }
    query_router_->addCluster(new_cluster_id, new_centroids[new_worker_centroid_idx], new_worker_addr);

    // Rebuild routing index from balanced K-means assignments on sampled points
    std::unordered_map<int, std::vector<std::vector<float>>> new_cluster_samples;
    {
        if (!reservoir_snapshot.empty()) {
            for (size_t i = 0; i < reservoir_snapshot_n; ++i) {
                int a = assignments[i];
                int cid = centroid_to_cluster_id[a];
                const float* p = reservoir_snapshot.data() + i * reservoir_snapshot_dim;
                new_cluster_samples[cid].emplace_back(p, p + reservoir_snapshot_dim);
            }
        } else {
            for (size_t i = 0; i < fallback_points.size(); ++i) {
                int a = assignments[i];
                int cid = centroid_to_cluster_id[a];
                new_cluster_samples[cid].push_back(fallback_points[i]);
            }
        }
    }
    query_router_->rebuildRoutingIndex(new_cluster_samples);
    auto update_query_router_t1 = std::chrono::steady_clock::now();
    double update_query_router_ms = std::chrono::duration<double, std::milli>(update_query_router_t1 - update_query_router_t0).count();
    std::cout << "Master: Updating query router took " << update_query_router_ms << " ms" << '\n';

    auto final_clusters = query_router_->getAllClusters();
    std::cout << "\nMaster: Rebalance complete. " << final_clusters.size() << " clusters:" << '\n';
    for (const auto& cluster : final_clusters) {
        std::cout << "  Cluster " << cluster.cluster_id << " (" << cluster.worker_address << ")" << '\n';
    }
    std::cout << "===== END REBALANCE =====" << '\n';

}

void MasterServiceImpl::triggerRebalance() {
    if (rebalance_method_ == RebalanceMethod::kPriceAuction) rebalanceWorkersPriceAuction();
    else rebalanceWorkers();
}

void MasterServiceImpl::rebalanceWorkersPriceAuction() {
    std::cout << "\n===== PRICE-AUCTION REBALANCE TRIGGERED =====" << '\n';

    std::string new_worker_addr = getFreeWorkerAddress();
    if (new_worker_addr.empty()) {
        std::cout << "Master: No free workers available for rebalance" << '\n';
        return;
    }
    std::cout << "Master: New worker for rebalance: " << new_worker_addr << '\n';

    {
        std::lock_guard<std::mutex> lock(workers_mutex_);
        if (workers_.find(new_worker_addr) == workers_.end()) {
            workers_[new_worker_addr] = std::make_shared<WorkerClient>(new_worker_addr);
        }
    }

    auto existing_clusters = query_router_->getAllClusters();
    const int num_existing = static_cast<int>(existing_clusters.size());
    if (num_existing == 0) {
        std::cout << "Master: No existing clusters, aborting rebalance" << '\n';
        free_workers_.push_back(new_worker_addr);
        return;
    }
    const int k = num_existing + 1;

    std::cout << "Master: " << num_existing << " existing clusters, will create " << k
              << " balanced partitions" << '\n';

    // Sample collection for K-means++ init, identical to rebalanceWorkers()
    std::vector<float>              reservoir_snapshot;
    int                             reservoir_snapshot_dim = 0;
    size_t                          reservoir_snapshot_n   = 0;
    std::vector<std::vector<float>> fallback_points;
    {
        std::lock_guard<std::mutex> lock(reservoir_mutex_);
        if (!reservoir_flat_.empty() && reservoir_dim_ > 0) {
            reservoir_snapshot     = reservoir_flat_;
            reservoir_snapshot_dim = reservoir_dim_;
            reservoir_snapshot_n   = reservoir_flat_.size() / static_cast<size_t>(reservoir_dim_);
            std::cout << "Master: Using " << reservoir_snapshot_n
                      << " reservoir samples for rebalance" << '\n';
        }
    }

    if (reservoir_snapshot.empty()) {
        const int sample_size = 5000;
        std::unordered_map<int, std::vector<std::vector<float>>> cluster_samples;
        for (const auto& cluster : existing_clusters) {
            auto worker_client = getOrCreateWorkerClient(cluster.worker_address);
            SamplePointsRequest sample_req;
            sample_req.set_num_points(sample_size);
            grpc::ClientContext ctx;
            auto reader = worker_client->stub->SamplePoints(&ctx, sample_req);

            std::vector<std::vector<float>> sampled_points;
            SamplePointsChunk chunk;
            while (reader->Read(&chunk)) {
                for (const auto& point : chunk.points()) {
                    const float* pv = as_floats(point.vector_data());
                    const int    pd = static_cast<int>(point.vector_data().size() / sizeof(float));
                    sampled_points.emplace_back(pv, pv + pd);
                }
                if (chunk.is_last_chunk()) break;
            }

            grpc::Status status = reader->Finish();
            if (status.ok() && !sampled_points.empty()) {
                cluster_samples[cluster.cluster_id] = std::move(sampled_points);
            }
        }

        if (cluster_samples.empty()) {
            std::cout << "Master: No samples collected, aborting rebalance" << '\n';
            free_workers_.push_back(new_worker_addr);
            return;
        }

        for (const auto& [cid, points] : cluster_samples) {
            for (const auto& p : points) fallback_points.push_back(p);
        }
    }

    const size_t total_sample_n = reservoir_snapshot.empty()
        ? fallback_points.size() : reservoir_snapshot_n;
    std::cout << "Master: Running K-means++ init on "
              << total_sample_n << " sampled points into " << k
              << " clusters" << '\n';

    kmeans::BalancedKMeans<float> bal_km(k, /*max_iterations=*/50);
    if (!reservoir_snapshot.empty()) {
        bal_km.fit(reservoir_snapshot.data(), static_cast<int>(reservoir_snapshot_n), reservoir_snapshot_dim);
    } else {
        bal_km.fit(fallback_points);
    }

    const auto& new_centroids = bal_km.get_centroids();
    if (new_centroids.empty()) {
        std::cout << "Master: K-means produced no centroids, aborting rebalance" << '\n';
        free_workers_.push_back(new_worker_addr);
        return;
    }
    const int dim = static_cast<int>(new_centroids[0].size());

    // Greedy centroid-to-worker matching, identical to rebalanceWorkers()
    std::vector<std::vector<float>> old_centroids;
    old_centroids.reserve(num_existing);
    for (const auto& cluster : existing_clusters) old_centroids.push_back(cluster.centroid);

    auto centroid_dist_sq = [dim](const std::vector<float>& a, const std::vector<float>& b) -> float {
        float s = 0.f;
        for (int d = 0; d < dim; ++d) { float diff = a[d] - b[d]; s += diff * diff; }
        return s;
    };

    std::vector<int> centroid_for_worker(num_existing, -1);
    std::vector<bool> centroid_used(k, false);
    for (int match = 0; match < num_existing; ++match) {
        float best = std::numeric_limits<float>::max();
        int best_w = -1, best_c = -1;
        for (int wi = 0; wi < num_existing; ++wi) {
            if (centroid_for_worker[wi] != -1) continue;
            for (int cj = 0; cj < k; ++cj) {
                if (centroid_used[cj]) continue;
                float d = centroid_dist_sq(old_centroids[wi], new_centroids[cj]);
                if (d < best) { best = d; best_w = wi; best_c = cj; }
            }
        }
        if (best_w == -1 || best_c == -1) {
            std::cout << "Master: Failed to assign centroid in greedy matching, aborting rebalance" << '\n';
            free_workers_.push_back(new_worker_addr);
            return;
        }
        centroid_for_worker[best_w] = best_c;
        centroid_used[best_c] = true;
    }

    int new_worker_centroid_idx = -1;
    for (int cj = 0; cj < k; ++cj) {
        if (!centroid_used[cj]) { new_worker_centroid_idx = cj; break; }
    }
    if (new_worker_centroid_idx == -1) {
        std::cout << "Master: Could not find remaining centroid for new worker, aborting rebalance" << '\n';
        free_workers_.push_back(new_worker_addr);
        return;
    }

    std::vector<int> centroid_to_cluster_id(k, -1);
    std::vector<std::string> centroid_to_worker_addr(k);
    for (int wi = 0; wi < num_existing; ++wi) {
        int cidx = centroid_for_worker[wi];
        centroid_to_cluster_id[cidx] = existing_clusters[wi].cluster_id;
        centroid_to_worker_addr[cidx] = existing_clusters[wi].worker_address;
    }
    int max_cluster_id = -1;
    for (const auto& c : existing_clusters) if (c.cluster_id > max_cluster_id) max_cluster_id = c.cluster_id;
    int new_cluster_id = max_cluster_id + 1;
    centroid_to_cluster_id[new_worker_centroid_idx] = new_cluster_id;
    centroid_to_worker_addr[new_worker_centroid_idx] = new_worker_addr;

    std::vector<float> centroids_flat;
    centroids_flat.reserve(static_cast<size_t>(k) * dim);
    for (int cj = 0; cj < k; ++cj) {
        centroids_flat.insert(centroids_flat.end(), new_centroids[cj].begin(), new_centroids[cj].end());
    }

    // Price-adjustment round loop: repeatedly count assignments under the current price vector
    // and nudge prices toward equal per-centroid counts before touching any data.
    constexpr int kMaxRounds = 20;
    constexpr double kEtaDecay = 0.75;
    std::vector<double> price(k, 0.0);
    bool eta_fixed_by_user = (rebalance_eta0_ > 0.0);
    double eta = eta_fixed_by_user ? rebalance_eta0_ : 0.0;

    struct PlanResult {
        bool ok;
        int64_t staying;
        int64_t migrating;
        double distortion;
        std::vector<std::pair<int, int64_t>> breakdown; // (target_centroid_idx, count)
    };

    for (int round = 1; round <= kMaxRounds; ++round) {
        std::vector<std::future<PlanResult>> futures;
        futures.reserve(num_existing);
        for (int wi = 0; wi < num_existing; ++wi) {
            int my_centroid_idx = centroid_for_worker[wi];
            auto wc = getOrCreateWorkerClient(existing_clusters[wi].worker_address);
            futures.emplace_back(std::async(std::launch::async,
                [wc, &centroids_flat, &price, dim, k, my_centroid_idx]() -> PlanResult {
                    AnalyzeMigrationRequest req;
                    set_float_bytes(req.mutable_centroids_data(), centroids_flat.data(),
                                    static_cast<int>(centroids_flat.size()));
                    req.set_num_centroids(k);
                    req.set_dimension(dim);
                    req.set_assigned_centroid_idx(my_centroid_idx);
                    for (double p : price) req.add_cluster_price(static_cast<float>(p));

                    AnalyzeMigrationResponse resp;
                    grpc::ClientContext ctx;
                    grpc::Status st = wc->stub->AnalyzeMigration(&ctx, req, &resp);
                    if (!st.ok()) return {false, 0, 0, 0.0, {}};

                    std::vector<std::pair<int, int64_t>> breakdown;
                    for (const auto& mc : resp.migration_breakdown()) {
                        breakdown.push_back({mc.target_centroid_idx(), mc.count()});
                    }
                    return {true, resp.staying_points(), resp.migrating_points(), resp.distortion(), breakdown};
                }));
        }

        std::vector<int64_t> n_j(k, 0);
        double total_distortion = 0.0;
        int64_t total_N = 0;
        bool all_ok = true;
        for (int wi = 0; wi < num_existing; ++wi) {
            PlanResult r = futures[wi].get();
            if (!r.ok) { all_ok = false; continue; }
            n_j[centroid_for_worker[wi]] += r.staying;
            for (const auto& [c, count] : r.breakdown) n_j[c] += count;
            total_distortion += r.distortion;
            total_N += r.staying + r.migrating;
        }
        if (!all_ok) {
            std::cout << "Master: A worker failed AnalyzeMigration, aborting rebalance" << '\n';
            free_workers_.push_back(new_worker_addr);
            return;
        }
        if (total_N == 0) {
            std::cout << "Master: No points to rebalance, aborting" << '\n';
            free_workers_.push_back(new_worker_addr);
            return;
        }

        const double q_j = static_cast<double>(total_N) / k;
        if (round == 1) {
            if (!eta_fixed_by_user) {
                // eta_0 = 0.5 * mean(d^2), where mean(d^2) is round 1's aggregated distortion / total_N
                // (round 1 runs at zero price, so this is the true full-dataset mean squared distance).
                eta = 0.5 * (total_distortion / static_cast<double>(total_N));
            }
            std::cout << "Master: eta_0=" << eta
                       << (eta_fixed_by_user ? " (user-specified)" : " (auto-derived from round-1 distortion)")
                       << '\n';
        }

        double max_rel_err = 0.0;
        for (int c = 0; c < k; ++c) max_rel_err = std::max(max_rel_err, std::abs(n_j[c] - q_j) / q_j);

        std::cout << "Master: Round " << round << " n_j=[";
        for (int c = 0; c < k; ++c) std::cout << n_j[c] << (c + 1 < k ? "," : "");
        std::cout << "] price=[";
        for (int c = 0; c < k; ++c) std::cout << price[c] << (c + 1 < k ? "," : "");
        std::cout << "] eta=" << eta << " max_rel_err=" << max_rel_err << '\n';

        if (max_rel_err <= rebalance_epsilon_) {
            std::cout << "Master: Price auction CONVERGED after " << round << " round(s): max_rel_err="
                       << max_rel_err << " <= epsilon=" << rebalance_epsilon_ << '\n';
            break;
        }
        if (round == kMaxRounds) {
            std::cout << "Master: WARNING price auction did NOT converge after " << round << " round(s): max_rel_err="
                       << max_rel_err << " > epsilon=" << rebalance_epsilon_ << '\n';
            break;
        }

        for (int c = 0; c < k; ++c) price[c] += eta * (n_j[c] - q_j) / q_j;
        double mean_price = std::accumulate(price.begin(), price.end(), 0.0) / k;
        for (auto& p : price) p -= mean_price;
        eta *= kEtaDecay;
    }

    // Round to float precision now, since this is the value that will actually go over the wire
    // to workers below -- keeps the router rebuild's nearest_by_score in exact agreement with it.
    for (auto& p : price) p = static_cast<double>(static_cast<float>(p));

    // Execute the split once, with the converged price vector, mirroring rebalanceWorkers()'s
    // Rebalance RPC fan-out and count-refresh but adding cluster_price to each request.
    struct RebalanceResult {
        bool ok;
        std::string worker_address;
    };
    auto rebalance_t0 = std::chrono::steady_clock::now();
    std::cout << "\n--- Price-auction Rebalance: Split & Transfer ---" << '\n';
    std::vector<RebalanceResult> rebalance_results(num_existing);
    {
        std::vector<std::future<RebalanceResult>> futures;
        for (int wi = 0; wi < num_existing; ++wi) {
            const auto& cluster = existing_clusters[wi];
            int my_centroid_idx = centroid_for_worker[wi];
            auto wc = getOrCreateWorkerClient(cluster.worker_address);

            futures.emplace_back(std::async(std::launch::async,
                [wc, &centroids_flat, &centroid_to_worker_addr, &price, dim, k, my_centroid_idx, cluster]() -> RebalanceResult {

                    RebalanceRequest req;
                    set_float_bytes(req.mutable_centroids_data(), centroids_flat.data(),
                                    static_cast<int>(centroids_flat.size()));
                    req.set_num_centroids(k);
                    req.set_dimension(dim);
                    req.set_assigned_centroid_idx(my_centroid_idx);
                    for (double p : price) req.add_cluster_price(static_cast<float>(p));

                    for (int cidx = 0; cidx < k; ++cidx) {
                        auto* m = req.add_centroid_to_worker();
                        m->set_centroid_idx(cidx);
                        m->set_worker_address(centroid_to_worker_addr[cidx]);
                    }

                    RebalanceResponse resp;
                    grpc::ClientContext ctx;
                    grpc::Status st = wc->stub->Rebalance(&ctx, req, &resp);
                    if (st.ok() && resp.success()) {
                        std::cout << "Master: Rebalance OK for " << cluster.worker_address << '\n';
                        return {true, cluster.worker_address};
                    }
                    std::cout << "Master: Rebalance FAILED for " << cluster.worker_address
                              << ": " << (st.ok() ? "worker error" : st.error_message()) << '\n';
                    return {false, cluster.worker_address};
                }));
        }

        bool all_ok = true;
        for (int wi = 0; wi < num_existing; ++wi) {
            rebalance_results[wi] = futures[wi].get();
            if (!rebalance_results[wi].ok) all_ok = false;
        }
        if (!all_ok) {
            std::cout << "Master: Some workers failed rebalance" << '\n';
        }
    }

    if (std::all_of(rebalance_results.begin(), rebalance_results.end(),
                    [](const RebalanceResult& result) { return result.ok; })) {
        std::vector<std::string> workers_to_query;
        workers_to_query.reserve(num_existing + 1);
        for (const auto& cluster : existing_clusters) workers_to_query.push_back(cluster.worker_address);
        workers_to_query.push_back(new_worker_addr);

        struct CountResult {
            bool ok;
            std::string worker_address;
            int64_t element_count;
        };

        std::vector<std::future<CountResult>> count_futures;
        count_futures.reserve(workers_to_query.size());
        for (const auto& worker_address : workers_to_query) {
            auto wc_count = getOrCreateWorkerClient(worker_address);
            count_futures.emplace_back(std::async(std::launch::async,
                [wc_count, worker_address]() -> CountResult {
                    GetCurrentElementCountRequest req;
                    GetCurrentElementCountResponse resp;
                    grpc::ClientContext ctx;
                    grpc::Status st = wc_count->stub->GetCurrentElementCount(&ctx, req, &resp);
                    if (st.ok()) return {true, worker_address, resp.element_count()};
                    std::cout << "Master: Count query failed for " << worker_address
                              << ": " << st.error_message() << '\n';
                    return {false, worker_address, 0};
                }));
        }

        std::vector<CountResult> count_results;
        count_results.reserve(count_futures.size());
        for (auto& future : count_futures) count_results.push_back(future.get());

        std::lock_guard<std::mutex> lock(workers_mutex_);
        for (const auto& result : count_results) {
            if (!result.ok) continue;
            auto it = workers_.find(result.worker_address);
            if (it != workers_.end()) it->second->node_count = result.element_count;
        }
    } else {
        std::cout << "Master: Skipping stable count refresh because rebalance did not succeed on every worker" << '\n';
    }
    auto rebalance_t1 = std::chrono::steady_clock::now();
    double rebalance_ms = std::chrono::duration<double, std::milli>(rebalance_t1 - rebalance_t0).count();
    std::cout << "Master: Rebalance (including final merge) took " << rebalance_ms << " ms" << '\n';

    // Update query router with the (unchanged) final centroids
    for (int wi = 0; wi < num_existing; ++wi) {
        int cidx = centroid_for_worker[wi];
        query_router_->addCluster(existing_clusters[wi].cluster_id, new_centroids[cidx], existing_clusters[wi].worker_address);
    }
    query_router_->addCluster(new_cluster_id, new_centroids[new_worker_centroid_idx], new_worker_addr);

    // Rebuild routing buckets using the same score function actually executed on the workers
    // (centroid distance + converged price), not the pre-price sample assignment.
    auto nearest_by_score = [&](const float* point) -> int {
        int best_c = 0;
        double best_score = std::numeric_limits<double>::max();
        for (int c = 0; c < k; ++c) {
            double d = 0.0;
            for (int dd = 0; dd < dim; ++dd) {
                double diff = point[dd] - new_centroids[c][dd];
                d += diff * diff;
            }
            double score = d + price[c];
            if (score < best_score) { best_score = score; best_c = c; }
        }
        return best_c;
    };

    std::unordered_map<int, std::vector<std::vector<float>>> new_cluster_samples;
    if (!reservoir_snapshot.empty()) {
        for (size_t i = 0; i < reservoir_snapshot_n; ++i) {
            const float* p = reservoir_snapshot.data() + i * reservoir_snapshot_dim;
            int cid = centroid_to_cluster_id[nearest_by_score(p)];
            new_cluster_samples[cid].emplace_back(p, p + reservoir_snapshot_dim);
        }
    } else {
        for (const auto& p : fallback_points) {
            int cid = centroid_to_cluster_id[nearest_by_score(p.data())];
            new_cluster_samples[cid].push_back(p);
        }
    }
    query_router_->rebuildRoutingIndex(new_cluster_samples);

    auto final_clusters = query_router_->getAllClusters();
    std::cout << "\nMaster: Price-auction rebalance complete. " << final_clusters.size() << " clusters:" << '\n';
    for (const auto& cluster : final_clusters) {
        std::cout << "  Cluster " << cluster.cluster_id << " (" << cluster.worker_address << ")" << '\n';
    }
    std::cout << "===== END PRICE-AUCTION REBALANCE =====" << '\n';
}

std::string MasterServiceImpl::getFreeWorkerAddress() {
    if (free_workers_.empty()) {
        return "";
    }
    std::string worker = free_workers_.front();
    free_workers_.pop_front();
    return worker;
}

void MasterServiceImpl::maybeAddToReservoir(const float* vec, int dim) {
    if (vec == nullptr || dim == 0) return;

    std::lock_guard<std::mutex> lock(reservoir_mutex_);
    reservoir_seen_count_++;
    if (reservoir_dim_ == 0) reservoir_dim_ = dim;

    const size_t current_count = reservoir_flat_.size() / static_cast<size_t>(reservoir_dim_);
    if (current_count < kReservoirMaxSize_) {
        reservoir_flat_.insert(reservoir_flat_.end(), vec, vec + reservoir_dim_);
        return;
    }

    std::uniform_int_distribution<uint64_t> dist(0, reservoir_seen_count_ - 1);
    const uint64_t j = dist(reservoir_rng_);
    if (j < kReservoirMaxSize_) {
        std::copy(vec, vec + reservoir_dim_,
                  reservoir_flat_.begin() + static_cast<ptrdiff_t>(j) * reservoir_dim_);
    }
}

MasterServer::~MasterServer() {
    if (grpc_server_) {
        grpc_server_->Shutdown();
    }
}

void MasterServer::Run() {
    std::cout << "Master server listening on " << server_address_ << '\n';
    grpc_server_->Wait();
}

void MasterServer::Shutdown() {
    if (grpc_server_) {
        grpc_server_->Shutdown();
    }
}

MasterServer::MasterServer(const std::string& config_path, int split_threshold, bool manual_rebalance, int dim,
                            RebalanceMethod rebalance_method, double rebalance_eta0, double rebalance_epsilon) {
    ClusterConfig cfg = loadClusterConfig(config_path);
    std::string address = cfg.master.address();

    std::filesystem::create_directories("indexes");
    std::string centroids_path = "indexes/centroids.csv";
    if (std::filesystem::exists(centroids_path)) {
        std::filesystem::remove(centroids_path);
    }

    // Create components - start with only the first worker, rest go to free list
    std::vector<std::string> initial_worker_addresses;
    std::deque<std::string> free_worker_addresses;
    
    if (!cfg.workers.empty()) {
        initial_worker_addresses.push_back(cfg.workers[0].address());
        for (size_t i = 1; i < cfg.workers.size(); ++i) {
            free_worker_addresses.push_back(cfg.workers[i].address());
        }
    }
    
    int max_nodes = 50000;
    int M = 16;
    int efConstruction = 100;

    query_router_ = std::make_unique<QueryRouter>(dim, max_nodes, M, efConstruction, "", initial_worker_addresses);
    // query_router_ = std::make_unique<QueryRouter>(dim, max_nodes, M, efConstruction, "/path/to/sift_base.fvecs", initial_worker_addresses);
    aggregator_ = std::make_unique<Aggregator>();

    // Create service with free workers list and split threshold
    service_ = std::make_unique<MasterServiceImpl>(std::move(query_router_), std::move(aggregator_), std::move(free_worker_addresses), split_threshold, manual_rebalance,
                                                    rebalance_method, rebalance_eta0, rebalance_epsilon);

    // Build server
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    grpc_server_ = builder.BuildAndStart();
    if (!grpc_server_) {
        throw std::runtime_error("Failed to start master server");
    }
    server_address_ = address;
}

} // namespace ann

#ifndef ANN_BUILDING_TESTS
int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::string config_path = "config/workers.yaml";
    int split_threshold = SPLIT_THRESHOLD;
    bool manual_rebalance = false;
    int dim = -1;
    ann::RebalanceMethod rebalance_method = ann::RebalanceMethod::kNaive;
    double rebalance_eta0 = -1.0;
    double rebalance_epsilon = 0.10;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--split-threshold" && i + 1 < argc) {
            split_threshold = std::stoi(argv[++i]);
        } else if (arg == "--manual-rebalance") {
            manual_rebalance = true;
        } else if (arg == "--dim" && i + 1 < argc) {
            dim = std::stoi(argv[++i]);
        } else if (arg == "--rebalance-method" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "naive") rebalance_method = ann::RebalanceMethod::kNaive;
            else if (v == "price") rebalance_method = ann::RebalanceMethod::kPriceAuction;
            else { std::cerr << "Error: --rebalance-method must be 'naive' or 'price'" << std::endl; return 1; }
        } else if (arg == "--rebalance-eta" && i + 1 < argc) {
            rebalance_eta0 = std::stod(argv[++i]);
        } else if (arg == "--rebalance-epsilon" && i + 1 < argc) {
            rebalance_epsilon = std::stod(argv[++i]);
        }
    }
    if (dim <= 0) {
        std::cerr << "Error: --dim <positive_integer> is required" << std::endl;
        return 1;
    }
    try {
        ann::MasterServer server(config_path, split_threshold, manual_rebalance, dim,
                                  rebalance_method, rebalance_eta0, rebalance_epsilon);
        server.Run();
    } catch (const std::exception& e) {
        std::cerr << "Master server failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
#endif // ANN_BUILDING_TESTS
