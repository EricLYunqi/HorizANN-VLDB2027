#pragma once

#include <memory>
#include <string>
#include <mutex>
#include <utility>
#include <vector>
#include <grpcpp/grpcpp.h>
#include <rocksdb/db.h>
#include "proto/ann.pb.h"
#include "proto/ann.grpc.pb.h"
#include "i_cluster_index.h"

namespace ann {

template<typename dist_t>
class WorkerServiceImpl final : public WorkerService::Service {
public:
    WorkerServiceImpl(std::unique_ptr<IClusterIndex<dist_t>> cluster_index, int worker_id, int dim, int max_elements, std::string scratch_dir = "");

    grpc::Status Search(grpc::ServerContext* context, const Query* request, WorkerResult* response) override;
    grpc::Status BatchedSearch(grpc::ServerContext* context, const BatchedQuery* request, BatchedWorkerResult* response) override;
    grpc::Status Insert(grpc::ServerContext* context, const InsertData* request, InsertAck* response) override;
    grpc::Status BatchedInsert(grpc::ServerContext* context, const BatchedInsertData* request, BatchedInsertAck* response) override;
    grpc::Status Split(grpc::ServerContext* context, const SplitRequest* request, SplitResponse* response) override;
    grpc::Status TransferIndex(grpc::ServerContext* context, grpc::ServerReader<IndexChunk>* reader, LoadIndexResponse* response) override;
    grpc::Status SamplePoints(grpc::ServerContext* context, const SamplePointsRequest* request, grpc::ServerWriter<SamplePointsChunk>* writer) override;
    grpc::Status AnalyzeMigration(grpc::ServerContext* context, const AnalyzeMigrationRequest* request, AnalyzeMigrationResponse* response) override;
    grpc::Status Rebalance(grpc::ServerContext* context, const RebalanceRequest* request, RebalanceResponse* response) override;
    grpc::Status MergeIndex(grpc::ServerContext* context, grpc::ServerReader<IndexChunk>* reader, LoadIndexResponse* response) override;
    grpc::Status GetCurrentElementCount(grpc::ServerContext* context, const GetCurrentElementCountRequest* request, GetCurrentElementCountResponse* response) override;
    grpc::Status AddGhostEdges(grpc::ServerContext* context, const BatchedAddGhostEdgesRequest* request, BatchedAddGhostEdgesAck* response) override;
    grpc::Status GetEdgeStats(grpc::ServerContext* context, const GetEdgeStatsRequest* request, EdgeStatsResponse* response) override;

    IClusterIndex<dist_t>* getClusterIndex() { return cluster_index_.get(); }

private:
    bool ensureResidentGhostLinksLoaded(const std::string& reason);
    bool persistResidentGhostLinks(const std::string& reason);

    // Flush a list of (inserted_label, ghost_edges) to the resident ghost file and
    // maintain the bounded in-memory insertion ghost buffer.
    struct GhostEdge { int64_t neighbor_id; float distance; };
    void flushInsertionGhostBuf(
        const std::vector<std::pair<int64_t, std::vector<GhostEdge>>>& new_entries);

    std::unique_ptr<IClusterIndex<dist_t>> cluster_index_;
    int worker_id_;
    int dim_;
    int max_elements_;
    std::string scratch_dir_;
    std::mutex index_mutex_;
    std::string resident_ghost_links_path_;
    bool resident_ghost_links_on_disk_ = false;

    // Persistent RocksDB handle for insertion ghost links.
    // Kept open to avoid per-batch open/close and the LOCK contention that causes.
    rocksdb::DB* ghost_insertion_db_ = nullptr;
    std::mutex ghost_db_mutex_;    // guards ghost_insertion_db_ open/close only
    rocksdb::DB* ghostDb();        // opens ghost_insertion_db_ on demand, shared by append/save/load
    void closeGhostInsertionDb();  // closes and nulls ghost_insertion_db_

};

// Wrapper class around the gRPC server
class WorkerServer {
public:
    WorkerServer(const std::string& server_address, int worker_id, int dim, int max_elements, const std::string& vector_type, std::string scratch_dir = "");
    ~WorkerServer();

    void Run();       // blocking call
    void Shutdown();  // clean shutdown (used by signal handler)

private:
    std::string server_address_;
    int worker_id_;

    std::unique_ptr<grpc::Server> grpc_server_;
    std::unique_ptr<WorkerService::Service> service_;
};

} // namespace ann
