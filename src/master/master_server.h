#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <future>
#include <random>
#include <deque>
#include <grpcpp/grpcpp.h>
#include "proto/ann.pb.h"
#include "proto/ann.grpc.pb.h"
#include "i_query_router.h"
#include "i_aggregator.h"

#define SPLIT_THRESHOLD 100000000

namespace ann {

enum class RebalanceMethod { kNaive, kPriceAuction };

struct WorkerClient {
    explicit WorkerClient(const std::string& addr)
        : worker_address(addr),
          channel(grpc::CreateChannel(addr, grpc::InsecureChannelCredentials())),
          stub(WorkerService::NewStub(channel)),
          node_count(0) {}
    std::string worker_address;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<WorkerService::Stub> stub;
    long long node_count;
};

class MasterServiceImpl final : public MasterService::Service {
public:
    MasterServiceImpl(std::unique_ptr<IQueryRouter> router, std::unique_ptr<IAggregator> aggregator, std::deque<std::string> free_workers, int split_threshold = SPLIT_THRESHOLD, bool manual_rebalance = false,
                       RebalanceMethod rebalance_method = RebalanceMethod::kNaive, double rebalance_eta0 = -1.0, double rebalance_epsilon = 0.10);
    
    // gRPC service implementations
    grpc::Status Search(grpc::ServerContext* context, const Query* request, AggregatedResult* response) override;
    grpc::Status BatchedSearch(grpc::ServerContext* context, const BatchedQuery* request, BatchedAggregatedResult* response) override;
    grpc::Status Insert(grpc::ServerContext* context, const InsertData* request, InsertAck* response) override;
    grpc::Status BatchedInsert(grpc::ServerContext* context, const BatchedInsertData* request, BatchedInsertAck* response) override;
    grpc::Status AddWorker(grpc::ServerContext* context, const AddWorkerRequest* request, AddWorkerResponse* response) override;
    grpc::Status GetClusterEdgeStats(grpc::ServerContext* context, const GetClusterEdgeStatsRequest* request, GetClusterEdgeStatsResponse* response) override;

private:
    std::unique_ptr<IQueryRouter> query_router_;
    std::unique_ptr<IAggregator> aggregator_;
    std::mutex workers_mutex_;
    std::unordered_map<std::string, std::shared_ptr<WorkerClient>> workers_;
    std::deque<std::string> free_workers_; // FIFO queue of available worker addresses
    std::shared_ptr<WorkerClient> getOrCreateWorkerClient(const std::string& worker_address);
    int split_threshold_; // trigger split when node_count exceeds this
    bool manual_rebalance_; // if true, rebalance only when client calls AddWorker
    RebalanceMethod rebalance_method_; // which rebalance workflow triggerRebalance() dispatches to
    double rebalance_eta0_; // price step size for round 1; <=0 means auto-derive from round-1 distortion
    double rebalance_epsilon_; // convergence tolerance on max relative count error

    // seed vectors for routing index (flat buffer)
    std::vector<float> bootstrap_flat_;
    int    bootstrap_dim_   = 0;
    size_t bootstrap_count_ = 0;
    
    // Helper method to forward search to workers
    std::vector<WorkerResult> forwardSearchToWorkers(const Query& query, const std::vector<ClusterInfo>& clusters);
    
    // Helper method to forward insert to worker
    InsertAck forwardInsertToWorker(const InsertData& data, const ClusterInfo& cluster);

    // OLD: 2-way split (commented out, replaced by rebalanceWorkers)
    // void splitWorker(const std::string& worker_address);
    
    // Rebalance: run balanced k-means, multi-way split, transfer sub-indexes
    void rebalanceWorkers();

    // Rebalance via price-adjusted (auction-style) full-data balancing; see rebalance_method_
    void rebalanceWorkersPriceAuction();

    // Dispatches to rebalanceWorkers() or rebalanceWorkersPriceAuction() based on rebalance_method_
    void triggerRebalance();

    // Helper to get a free worker address (pops from free_workers_)
    std::string getFreeWorkerAddress();

    // Global reservoir sampling over inserted vectors (flat buffer)
    static constexpr size_t kReservoirMaxSize_ = 50000;
    std::mutex reservoir_mutex_;
    std::vector<float> reservoir_flat_;
    int      reservoir_dim_          = 0;
    uint64_t reservoir_seen_count_   = 0;
    std::mt19937 reservoir_rng_{std::random_device{}()};

    void maybeAddToReservoir(const float* vec, int dim);
};

class MasterServer {
public:
    explicit MasterServer(const std::string& config_path, int split_threshold, bool manual_rebalance, int dim,
                           RebalanceMethod rebalance_method = RebalanceMethod::kNaive,
                           double rebalance_eta0 = -1.0, double rebalance_epsilon = 0.10);
    ~MasterServer();
    
    void Run();
    void Shutdown();

private:
    std::string server_address_;
    std::unique_ptr<grpc::Server> grpc_server_;
    std::unique_ptr<MasterServiceImpl> service_;
    std::unique_ptr<IQueryRouter> query_router_;
    std::unique_ptr<IAggregator> aggregator_;
};

} // namespace ann
