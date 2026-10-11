#include "worker_server.h"
#include "cluster_index.h"
#include "include/hnsw_splitter.h"
#include "include/space_l2.h"
#include "include/space_traits.h"
#include "include/kmeans.h"
#include "include/vec_bytes.h"
#include "include/ghost_db_stats.h"
#include <type_traits>
#include "proto/ann.pb.h"
#include <fstream>
#include <iostream>
#include <grpcpp/grpcpp.h>
#include "proto/ann.grpc.pb.h" 
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <csignal>
#include <omp.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>
#include <set>
#include <limits>
#include <atomic>
#include <future>
#include <chrono>

namespace ann {

static std::atomic<uint64_t> g_merge_file_counter{0};

// Helper function to get file size, returns 0 if file does not exist or error occurs
static uint64_t getFileSize(const std::string& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<uint64_t>(size);
}

// Helper function to get total size of a directory (for RocksDB ghost link DBs)
static uint64_t getDirSize(const std::string& path) {
    uint64_t total = 0;
    std::error_code ec;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path, ec)) {
        if (entry.is_regular_file(ec)) {
            total += entry.file_size(ec);
        }
    }
    return total;
}

// Helper function to get memory usage from /proc/self/status
struct MemoryStats {
    size_t rss_mb;      // Current Resident Set Size
    size_t peak_rss_mb; // Peak Resident Set Size (HWM - High Water Mark)
    size_t vm_size_mb;  // Virtual Memory Size
};

static MemoryStats getMemoryStats() {
    MemoryStats stats = {0, 0, 0};
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string label;
            size_t value;
            std::string unit;
            iss >> label >> value >> unit;
            stats.rss_mb = value / 1024;  // Convert from KB to MB
        } else if (line.substr(0, 6) == "VmHWM:") {
            std::istringstream iss(line);
            std::string label;
            size_t value;
            std::string unit;
            iss >> label >> value >> unit;
            stats.peak_rss_mb = value / 1024;  // Convert from KB to MB
        } else if (line.substr(0, 7) == "VmSize:") {
            std::istringstream iss(line);
            std::string label;
            size_t value;
            std::string unit;
            iss >> label >> value >> unit;
            stats.vm_size_mb = value / 1024;  // Convert from KB to MB
        }
    }
    return stats;
}

// Helper function to log memory usage
static void logMemoryUsage(const std::string& context, int worker_id) {
    MemoryStats stats = getMemoryStats();
    std::cout << "[Worker " << worker_id << "] Memory [" << context << "]: " 
              << "RSS=" << stats.rss_mb << " MB, "
              << "Peak=" << stats.peak_rss_mb << " MB, "
              << "VmSize=" << stats.vm_size_mb << " MB" << '\n';
}

void crashHandler(int sig) {
    std::cout << "[Worker] CRASH DETECTED: Signal " << sig << " received!" << '\n';
    exit(1);
}

// ---------------- Worker Service Implementation ----------------

template<typename dist_t>
WorkerServiceImpl<dist_t>::WorkerServiceImpl(std::unique_ptr<IClusterIndex<dist_t>> cluster_index, int worker_id, int dim, int max_elements, std::string scratch_dir)
    : cluster_index_(std::move(cluster_index)),
      worker_id_(worker_id),
      dim_(dim),
      max_elements_(max_elements),
      scratch_dir_(std::move(scratch_dir)) {
    const std::filesystem::path base = std::filesystem::path(scratch_dir_);
    resident_ghost_links_path_ = (base / ("worker_" + std::to_string(worker_id_) + "_resident.ghost.db")).string();
    // Clear any ghost links left on disk from a previous experiment
    if (std::filesystem::exists(resident_ghost_links_path_)) {
        rocksdb::Status s = rocksdb::DestroyDB(resident_ghost_links_path_, rocksdb::Options{});
        if (!s.ok()) {
            std::cerr << "[Worker " << worker_id_ << "] Warning: DestroyDB failed for "
                      << resident_ghost_links_path_ << ": " << s.ToString()
                      << " — continuing anyway\n";
        } else {
            std::cout << "[Worker " << worker_id_ << "] Cleared stale ghost links DB at " << resident_ghost_links_path_ << '\n';
        }
    }
    resident_ghost_links_on_disk_ = false;
    std::remove(("/tmp/worker" + std::to_string(worker_id_) + "_batched_insert.csv").c_str());
    std::remove(("/tmp/worker" + std::to_string(worker_id_) + "_add_ghost_edges.csv").c_str());
}

template<typename dist_t>
bool WorkerServiceImpl<dist_t>::ensureResidentGhostLinksLoaded(const std::string& reason) {
    // load ghost links in memory, index should be locked 
    auto* hnsw = cluster_index_->getHNSWIndex();
    if (hnsw->ghost_links_in_memory_) return false;
    if (!resident_ghost_links_on_disk_) return false;

    rocksdb::DB* db = ghostDb();
    if (!db) return false;

    logMemoryUsage("Ghost load / before file read (" + reason + ")", worker_id_);
    hnsw->loadGhostLinks(db);
    logMemoryUsage("Ghost load / after file read (" + reason + ")", worker_id_);
    
    if (hnsw->ghost_links_in_memory_) {
        const uintmax_t ghost_file_size = getDirSize(resident_ghost_links_path_);
        std::cout << "[Worker " << worker_id_ << "][Ghost] Load (" << reason
                  << "): vectors=" << hnsw->getGhostLinkedVectorCount()
                //   << ", edges=" << hnsw->getGhostLinkCount()
                  << ", size=" << ghost_file_size << " bytes"
                  << '\n';
    }

    return hnsw->ghost_links_in_memory_;
}

template<typename dist_t>
bool WorkerServiceImpl<dist_t>::persistResidentGhostLinks(const std::string& reason) {

    // save ghost links to disk, index should be locked
    auto* hnsw = cluster_index_->getHNSWIndex();

    if (!hnsw->ghost_links_in_memory_) {
        return false;
    }

    std::filesystem::create_directories(scratch_dir_);
    logMemoryUsage("Ghost save / before file write (" + reason + ")", worker_id_);

    size_t active_nodes_stored = 0;
    rocksdb::DB* db = ghostDb();
    if (!db) return false;

    resident_ghost_links_on_disk_ = hnsw->saveGhostLinks(db, &active_nodes_stored);

    const uint64_t ghost_file_size = resident_ghost_links_on_disk_
        ? getDirSize(resident_ghost_links_path_)
        : 0;
    logMemoryUsage("Ghost save / after file write (" + reason + ")", worker_id_);

    // once saved to disk, clear from memory
    hnsw->clearGhostLinksMemory();
    logMemoryUsage("Ghost save / after clear (" + reason + ")", worker_id_);

    if (resident_ghost_links_on_disk_) {
        std::cout << "[Worker " << worker_id_ << "][Ghost] Saved (" << reason
                  << "): size=" << ghost_file_size << 
                  " bytes, active_nodes_stored=" << active_nodes_stored <<
                  ", path="
                  << resident_ghost_links_path_
                  << '\n';
    } else {
        std::cout << "[Worker " << worker_id_ << "][Ghost] Did not save to disk (" << reason
                  << "): size=" << ghost_file_size << " bytes, path="
                  << resident_ghost_links_path_
                  << '\n';
    }

    return true;
}

// The one ghost DB handle for this worker. RocksDB allows a single writer per directory, so
// append / save / load all go through this instead of opening their own, which is what used to
// force a close+reopen (118-233s on a 20GB+ store) at every rebalance.
// Only the open/close is serialised, and only because sharing one handle turned the old
// microsecond check-then-set window into a DB::Open that takes minutes on a 20GB+ store —
// two callers racing it would both open, and the loser fails on the LOCK file.
// RocksDB's DB object is itself thread safe for concurrent Write/Get/NewIterator, so appends
// still do not wait behind a save or a load.
template<typename dist_t>
rocksdb::DB* WorkerServiceImpl<dist_t>::ghostDb() {
    std::lock_guard<std::mutex> lock(ghost_db_mutex_);
    if (ghost_insertion_db_) return ghost_insertion_db_;

    std::filesystem::create_directories(scratch_dir_);
    rocksdb::Options opts;
    opts.create_if_missing = true;
    opts.error_if_exists   = false;
    rocksdb::DB* raw_db = nullptr;
    rocksdb::Status s = rocksdb::DB::Open(opts, resident_ghost_links_path_, &raw_db);
    if (!s.ok()) {
        std::cerr << "[Worker " << worker_id_ << "][Ghost] Cannot open ghost DB: "
                  << s.ToString() << '\n';
        return nullptr;
    }

    ghost_insertion_db_ = raw_db;
    return ghost_insertion_db_;
}

template<typename dist_t>
void WorkerServiceImpl<dist_t>::closeGhostInsertionDb() {
    std::lock_guard<std::mutex> lock(ghost_db_mutex_);
    if (ghost_insertion_db_) {
        delete ghost_insertion_db_;
        ghost_insertion_db_ = nullptr;
    }
}

template<typename dist_t>
void WorkerServiceImpl<dist_t>::flushInsertionGhostBuf(
    const std::vector<std::pair<int64_t, std::vector<GhostEdge>>>& new_entries)
{
    if (new_entries.empty()) return;

    auto* hnsw = cluster_index_->getHNSWIndex();

    // Convert to hnswalg GhostLink pairs
    using GL = typename hnswlib::HierarchicalNSW<dist_t>::GhostLink;
    std::vector<std::pair<hnswlib::labeltype, std::vector<GL>>> to_write;
    to_write.reserve(new_entries.size());

    for (const auto& [label, edges] : new_entries) {
        std::vector<GL> links;
        links.reserve(edges.size());
        for (const auto& e : edges) {
            GL gl;
            gl.label    = static_cast<hnswlib::labeltype>(e.neighbor_id);
            gl.distance = static_cast<dist_t>(e.distance);
            links.push_back(gl);
        }
        to_write.push_back({static_cast<hnswlib::labeltype>(label), std::move(links)});
    }

    std::filesystem::create_directories(scratch_dir_);
    bool wrote = false;
    try {
        rocksdb::DB* db = ghostDb();
        if (!db) return;

        wrote = hnsw->appendInsertionGhostLinks(db, to_write);
    } catch (const std::exception& e) {
        std::cerr << "[Worker " << worker_id_ << "][Ghost] appendInsertionGhostLinks failed: "
                  << e.what() << '\n';
        return;
    }

    if (wrote) {
        resident_ghost_links_on_disk_ = true;
        std::cout << "[Worker " << worker_id_ << "][Ghost] Appended " << to_write.size()
                  << " insertion ghost entries to " << resident_ghost_links_path_ << '\n';
    }
}

template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::Search(grpc::ServerContext* context, const Query* request, WorkerResult* response) {

    auto t0 = std::chrono::steady_clock::now();
    response->set_worker_id(worker_id_);
    const float* query_data = as_floats(request->vector_data());
    const int dim = static_cast<int>(request->vector_data().size() / sizeof(float));

    int local_k = request->k();
    auto neighbors = cluster_index_->searchKNN(query_data, local_k);

    for (const auto& [distance, id] : neighbors) {
        auto* neighbor = response->add_results();
        neighbor->set_id(id);
        neighbor->set_distance(distance);
    }
    
    auto* hnsw_index = cluster_index_->getHNSWIndex();
    int current_nodes = hnsw_index->getCurrentElementCount();

    auto t1 = std::chrono::steady_clock::now();
    const double compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    {
        std::string filename = "/tmp/worker" + std::to_string(worker_id_) + "_search.csv";
        std::ofstream ws_csv(filename, std::ios::app);
        if (ws_csv.tellp() == 0) ws_csv << "query_id,worker_id,worker_search_compute_ms\n";
        ws_csv << request->query_id() << ',' << worker_id_ << ',' << compute_ms << '\n';
    }
    return grpc::Status::OK;
}

// TODO: remove
template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::Split(grpc::ServerContext* context, const SplitRequest* request, SplitResponse* response) {
    try {
        std::cout << "[Worker " << worker_id_ << "] Split requested. Free worker address=\""
                  << request->free_worker_address() << "\"" << '\n';
        
        auto* hnsw = cluster_index_->getHNSWIndex();
        const size_t N = hnsw->getCurrentElementCount();
  
        // Sample 50k points for k-means centroids
        size_t sample_n = std::min<size_t>(50000, N);
        std::vector<hnswlib::tableint> sample_ids(sample_n);
        std::mt19937 gen(42);
        
        std::vector<hnswlib::tableint> all_indices(N);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        
        for (size_t i = 0; i < sample_n; ++i) {
            std::uniform_int_distribution<size_t> dis(i, N - 1);
            size_t j = dis(gen);
            std::swap(all_indices[i], all_indices[j]);
            sample_ids[i] = all_indices[i];
        }

        std::vector<std::vector<float>> sample_data;
        sample_data.reserve(sample_n);
        for (size_t i = 0; i < sample_n; ++i) {
            if constexpr (std::is_same_v<dist_t, int>) {
                const unsigned char* raw = reinterpret_cast<const unsigned char*>(hnsw->getDataByInternalId(sample_ids[i]));
                std::vector<float> fvec(dim_);
                for (int d = 0; d < dim_; ++d) fvec[d] = static_cast<float>(raw[d]);
                sample_data.push_back(std::move(fvec));
            } else {
                const float* vec = reinterpret_cast<const float*>(hnsw->getDataByInternalId(sample_ids[i]));
                sample_data.emplace_back(vec, vec + dim_);
            }
        }

        kmeans::KMeans<float> km(2);
        km.fit(sample_data);
        const auto& centroids = km.get_centroids();

        std::cout << "[Worker " << worker_id_ << "] Creating space with dim=" << dim_ << '\n';
        auto owned_spaceB = std::make_unique<typename hnswlib::SpaceFor<dist_t>::type>(dim_);

        std::vector<uint32_t> assignments(N, 0);
        if constexpr (std::is_same_v<dist_t, int>) {
            // Quantize the 2 centroids to uint8 once, then compare stored uint8 data directly
            typename hnswlib::SpaceFor<dist_t>::type int_space(dim_);
            auto dist_func_i = int_space.get_dist_func();
            void* dist_param_i = int_space.get_dist_func_param();
            std::array<std::vector<unsigned char>, 2> uint8_centroids;
            for (int c = 0; c < 2; ++c) {
                uint8_centroids[c].resize(dim_);
                for (int d = 0; d < dim_; ++d)
                    uint8_centroids[c][d] = static_cast<unsigned char>(
                        std::max(0L, std::min(255L, std::lround(static_cast<double>(centroids[c][d])))));
            }
            for (size_t i = 0; i < N; ++i) {
                const unsigned char* vec = reinterpret_cast<const unsigned char*>(hnsw->getDataByInternalId(static_cast<hnswlib::tableint>(i)));
                int d0 = dist_func_i(vec, uint8_centroids[0].data(), dist_param_i);
                int d1 = dist_func_i(vec, uint8_centroids[1].data(), dist_param_i);
                assignments[i] = (d0 <= d1) ? 0U : 1U;
            }
        } else {
            auto space = std::make_unique<hnswlib::L2Space>(dim_);
            auto dist_func = space->get_dist_func();
            void* dist_param = space->get_dist_func_param();
            for (size_t i = 0; i < N; ++i) {
                const float* vec = reinterpret_cast<const float*>(hnsw->getDataByInternalId(static_cast<hnswlib::tableint>(i)));
                float d0 = dist_func(vec, centroids[0].data(), dist_param);
                float d1 = dist_func(vec, centroids[1].data(), dist_param);
                assignments[i] = (d0 <= d1) ? 0U : 1U;
            }
        }

        hnsw_splitter::HNSWSplitter<dist_t> splitter(hnsw);
        std::vector<hnswlib::SpaceInterface<dist_t>*> transfer_spaces = {owned_spaceB.get()};
        std::cout << "[Worker " << worker_id_ << "] Performing in-place HNSW split..." << '\n';
        auto split_children = splitter.splitInPlace(assignments, transfer_spaces, true);

        const int64_t child_a_count = hnsw->getCurrentElementCount();
        const int64_t child_b_count = split_children[0]->getCurrentElementCount();
        std::cout << "[Worker " << worker_id_ << "] Split produced counts A="
                  << child_a_count << ", B=" << child_b_count << '\n';

        // Transfer child B
        const std::string free_worker_address = request->free_worker_address();
        if (!free_worker_address.empty()) {
            std::cout << "[Worker " << worker_id_ << "] Streaming child B to "
                      << free_worker_address << '\n';
            
            try {
                auto channel = grpc::CreateChannel(free_worker_address, grpc::InsecureChannelCredentials());
                auto stub = WorkerService::NewStub(channel);
                grpc::ClientContext uctx;
                LoadIndexResponse uresp;
                auto writer = stub->TransferIndex(&uctx, &uresp);

                std::filesystem::create_directories(scratch_dir_);

                std::string save_path_B = scratch_dir_ + "/worker_" + std::to_string(worker_id_) + "_childB.bin";
                std::cout << "[Worker " << worker_id_ << "] Saving child B index (nodes=" << child_b_count 
                          << ") to " << save_path_B << "..." << '\n';
                
                split_children[0]->saveIndex(save_path_B);
                split_children[0].reset();
                
                std::ifstream in(save_path_B, std::ios::binary);
                if (!in.is_open()) {
                    std::cout << "[Worker " << worker_id_ << "] Failed to open temp index file for streaming: "
                              << save_path_B << '\n';
                    response->set_success(false);
                    return grpc::Status::OK;
                }
                
                const size_t chunk = 1 << 20;
                std::vector<char> buf(chunk);
                while (in) {
                    in.read(buf.data(), buf.size());
                    std::streamsize got = in.gcount();
                    if (got <= 0) break;
                    IndexChunk ch;
                    ch.set_data(buf.data(), (int)got);
                    if (!writer->Write(ch)) {
                        std::cout << "[Worker " << worker_id_ << "] Stream write returned false" << '\n';
                        break;
                    }
                }
                writer->WritesDone();
                auto st = writer->Finish();
                if (!st.ok() || !uresp.success()) {
                    std::cout << "[Worker " << worker_id_ << "] TransferIndex failed: code="
                              << st.error_code() << ", msg=\"" << st.error_message() << "\"" << '\n';
                    response->set_success(false);
                    return grpc::Status::OK;
                }
                std::cout << "[Worker " << worker_id_ << "] TransferIndex succeeded" << '\n';
            } catch (const std::exception& ex) {
                std::cout << "[Worker " << worker_id_ << "] Exception during TransferIndex: "
                          << ex.what() << '\n';
                response->set_success(false);
                return grpc::Status::OK;
            }
        } else {
            std::cout << "[Worker " << worker_id_ << "] No free worker address provided; skipping transfer" << '\n';
        }

        std::cout << "[Worker " << worker_id_ << "] Split complete. Index count: " 
                  << hnsw->getCurrentElementCount() << '\n';

        response->set_success(true);

        set_float_bytes(response->mutable_old_centroid_data(), centroids[0].data(), dim_);
        set_float_bytes(response->mutable_new_centroid_data(), centroids[1].data(), dim_);

        response->set_old_nodes(child_a_count);
        response->set_new_nodes(child_b_count);
        
        // logMemoryUsage("Split complete", worker_id_);
        std::cout << "[Worker " << worker_id_ << "] About to return from Split function..." << '\n';
        return grpc::Status::OK;
    } catch (const std::exception& e) {
        std::cout << "[Worker " << worker_id_ << "] Split failed with exception: " << e.what() << '\n';
        response->set_success(false);
        return grpc::Status::OK;
    }
}

// TODO: remove
template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::TransferIndex(grpc::ServerContext* context, grpc::ServerReader<IndexChunk>* reader, LoadIndexResponse* response) {

    std::string peer = context->peer();  // e.g. "ipv4:127.0.0.1:50051"
    std::cout << "[Worker " << worker_id_ << "] TransferIndex initiated by worker at: " << peer << '\n';

    // Receive bytes and write to a temp file, then load
    std::filesystem::create_directories(scratch_dir_);
    
    std::string tmp_path = scratch_dir_ + "/uploaded_index_" + std::to_string(worker_id_) + ".bin";
    std::cout << "[Worker " << worker_id_ << "] Writing received data to: " << tmp_path << '\n';
    
    std::ofstream out(tmp_path, std::ios::binary);
    if (!out.is_open()) {
        std::cout << "[Worker " << worker_id_ << "] Failed to open temp file for writing: " << tmp_path << '\n';
        response->set_success(false);
        return grpc::Status::OK;
    }
    
    IndexChunk chunk;
    size_t total_bytes = 0;
    size_t chunk_count = 0;
    
    while (reader->Read(&chunk)) {
        const std::string& d = chunk.data();
        out.write(d.data(), (std::streamsize)d.size());
        total_bytes += d.size();
        chunk_count++;
        
        if (chunk_count % 100 == 0) {
            std::cout << "[Worker " << worker_id_ << "] Received " << chunk_count << " chunks, " << total_bytes << " bytes so far" << '\n';
        }
    }
    
    out.close();
    
    std::cout << "[Worker " << worker_id_ << "] Loading index from file..." << '\n';
    bool ok = false;
    try {
        ok = cluster_index_->loadIndex(tmp_path);
        if (ok) {
            std::lock_guard<std::mutex> lock(index_mutex_);
            auto* hnsw = cluster_index_->getHNSWIndex();
            size_t loaded_nodes = hnsw->getCurrentElementCount();
            if (hnsw->getMaxElements() < (size_t)max_elements_) {
                logMemoryUsage("before resizeIndex (TransferIndex load)", worker_id_);
                hnsw->resizeIndex(max_elements_);
                logMemoryUsage("after resizeIndex (TransferIndex load)", worker_id_);
            }
            if (!hnsw->ghost_links_in_memory_ && resident_ghost_links_on_disk_) {
                closeGhostInsertionDb();
                std::filesystem::remove_all(resident_ghost_links_path_);
                resident_ghost_links_on_disk_ = false;
            }
            auto repair_t0 = std::chrono::steady_clock::now();
            hnsw->repairL0ZeroDegreeParallel();
            auto repair_t1 = std::chrono::steady_clock::now();
            double repair_ms = std::chrono::duration<double, std::milli>(repair_t1 - repair_t0).count();
            bool ghost_saved = false;
            double ghost_save_ms = 0.0;
            if (hnsw->ghost_links_in_memory_) {
                auto ghost_save_t0 = std::chrono::steady_clock::now();
                ghost_saved = persistResidentGhostLinks("split target load");
                auto ghost_save_t1 = std::chrono::steady_clock::now();
                if (ghost_saved) {
                    ghost_save_ms = std::chrono::duration<double, std::milli>(ghost_save_t1 - ghost_save_t0).count();
                }
            }
            std::cout << "[Worker " << worker_id_ << "] Successfully loaded index with "
                      << loaded_nodes << " nodes (deferred_zero_deg_repair=" << repair_ms << "ms)";
            if (ghost_saved) {
                const uintmax_t ghost_save_bytes = resident_ghost_links_on_disk_
                    ? getDirSize(resident_ghost_links_path_)
                    : 0;
                std::cout << " (resident_ghost_size=" << ghost_save_bytes
                          << " bytes, ghost_save=" << ghost_save_ms << "ms)";
            }
            std::cout << '\n';
        } else {
            std::cout << "[Worker " << worker_id_ << "] Failed to load index from file (loadIndex returned false)" << '\n';
        }
    } catch (const std::exception& e) {
        std::cout << "[Worker " << worker_id_ << "] Exception while loading index: " << e.what() << '\n';
        std::cerr << "[Worker " << worker_id_ << "] Exception while loading index: " << e.what() << '\n';
        ok = false;
    } catch (...) {
        std::cout << "[Worker " << worker_id_ << "] Unknown exception while loading index" << '\n';
        std::cerr << "[Worker " << worker_id_ << "] Unknown exception while loading index" << '\n';
        ok = false;
    }
    
    response->set_success(ok);
    return grpc::Status::OK;
}

// TODO: remove
template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::SamplePoints(grpc::ServerContext* context, const SamplePointsRequest* request, grpc::ServerWriter<SamplePointsChunk>* writer) {
    try {
        auto* hnsw = cluster_index_->getHNSWIndex();
        const size_t N = hnsw->getCurrentElementCount();
        const int num_points_requested = request->num_points();
        
        if (N == 0) {
            std::cout << "[Worker " << worker_id_ << "] SamplePoints: Index is empty" << '\n';
            return grpc::Status::OK;
        }
        
        // Sample min(num_points_requested, N) points
        const size_t sample_n = std::min<size_t>(static_cast<size_t>(num_points_requested), N);
        
        std::cout << "[Worker " << worker_id_ << "] SamplePoints: Sampling " << sample_n 
                  << " points from " << N << " total points" << '\n';
        
        // Generate random indices
        std::vector<hnswlib::tableint> sample_ids(sample_n);
        std::mt19937 gen(std::random_device{}());
        
        std::vector<hnswlib::tableint> all_indices(N);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        
        // Fisher-Yates shuffle to get random sample
        for (size_t i = 0; i < sample_n; ++i) {
            std::uniform_int_distribution<size_t> dis(i, N - 1);
            size_t j = dis(gen);
            std::swap(all_indices[i], all_indices[j]);
            sample_ids[i] = all_indices[i];
        }
        
        // Stream points in chunks to avoid gRPC message size limits
        const size_t chunk_size = 100; // points per chunk
        size_t points_sent = 0;
        
        for (size_t chunk_start = 0; chunk_start < sample_n; chunk_start += chunk_size) {
            SamplePointsChunk chunk;
            size_t chunk_end = std::min(chunk_start + chunk_size, sample_n);
            
            // Add points to this chunk
            for (size_t i = chunk_start; i < chunk_end; ++i) {
                auto* point = chunk.add_points();
                if constexpr (std::is_same_v<dist_t, int>) {
                    const unsigned char* raw = reinterpret_cast<const unsigned char*>(hnsw->getDataByInternalId(sample_ids[i]));
                    std::vector<float> tmp(dim_);
                    for (int d = 0; d < dim_; ++d) tmp[d] = static_cast<float>(raw[d]);
                    point->set_vector_data(reinterpret_cast<const char*>(tmp.data()), static_cast<size_t>(dim_) * sizeof(float));
                } else {
                    const float* vec = reinterpret_cast<const float*>(hnsw->getDataByInternalId(sample_ids[i]));
                    point->set_vector_data(reinterpret_cast<const char*>(vec), static_cast<size_t>(dim_) * sizeof(float));
                }
            }
            
            // Mark if this is the last chunk
            chunk.set_is_last_chunk(chunk_end >= sample_n);
            
            // Write chunk to stream
            if (!writer->Write(chunk)) {
                std::cout << "[Worker " << worker_id_ << "] SamplePoints: Failed to write chunk, stream may be closed" << '\n';
                return grpc::Status::OK;
            }
            
            points_sent += (chunk_end - chunk_start);
        }
        
        std::cout << "[Worker " << worker_id_ << "] SamplePoints: Successfully streamed " 
                  << points_sent << " points in chunks" << '\n';
        
        return grpc::Status::OK;
    } catch (const std::exception& e) {
        std::cout << "[Worker " << worker_id_ << "] SamplePoints exception: " << e.what() << '\n';
        std::cerr << "[Worker " << worker_id_ << "] SamplePoints exception: " << e.what() << '\n';
        return grpc::Status(grpc::StatusCode::INTERNAL, "Exception in SamplePoints: " + std::string(e.what()));
    } catch (...) {
        std::cout << "[Worker " << worker_id_ << "] SamplePoints: Unknown exception" << '\n';
        std::cerr << "[Worker " << worker_id_ << "] SamplePoints: Unknown exception" << '\n';
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown exception in SamplePoints");
    }
}

template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::AnalyzeMigration(grpc::ServerContext* context,
                                                          const AnalyzeMigrationRequest* request,
                                                          AnalyzeMigrationResponse* response) {
    try {
        auto* hnsw = cluster_index_->getHNSWIndex();
        const size_t N = hnsw->getCurrentElementCount();
        const int k = request->num_centroids();
        const int dim = request->dimension();
        const int my_centroid = request->assigned_centroid_idx();

        // Unflatten centroids
        std::vector<const float*> centroid_ptrs(k);
        const float* flat = as_floats(request->centroids_data());
        for (int c = 0; c < k; ++c) {
            centroid_ptrs[c] = flat + (size_t)c * dim;
        }

        // Per-centroid price bias; missing entries default to 0 so legacy callers are unaffected
        std::vector<double> price(k, 0.0);
        for (int c = 0; c < std::min(k, request->cluster_price_size()); ++c) price[c] = request->cluster_price(c);

        std::cout << "[Worker " << worker_id_ << "] AnalyzeMigration: "
                  << N << " points, k=" << k << ", assigned_centroid=" << my_centroid
                  << '\n';

        int64_t staying = 0;
        int64_t migrating = 0;
        std::vector<int64_t> migration_per_centroid(k, 0);
        double distortion = 0.0;
        const bool want_vec_sums = request->compute_vector_sums();
        std::vector<double> vec_sum;
        if (want_vec_sums) vec_sum.assign((size_t)k * dim, 0.0);

        // Quantize centroids to the index's native type once, then compare stored data directly.
        // Scores are computed in double (not float/int) so that with all-zero price the chosen
        // nearest centroid is identical to the pre-price behavior even at high dimensions.
        if constexpr (std::is_same_v<dist_t, int>) {
            typename hnswlib::SpaceFor<dist_t>::type int_space(dim_);
            auto dist_func_i = int_space.get_dist_func();
            void* dist_param_i = int_space.get_dist_func_param();
            std::vector<std::vector<unsigned char>> uint8_centroids(k, std::vector<unsigned char>(dim));
            for (int c = 0; c < k; ++c)
                for (int d = 0; d < dim; ++d)
                    uint8_centroids[c][d] = static_cast<unsigned char>(
                        std::max(0L, std::min(255L, std::lround(static_cast<double>(centroid_ptrs[c][d])))));

            #pragma omp parallel for schedule(static) reduction(+:staying,migrating,distortion)
            for (size_t i = 0; i < N; ++i) {
                if (hnsw->isMarkedDeleted(static_cast<hnswlib::tableint>(i))) continue;
                const unsigned char* vec = reinterpret_cast<const unsigned char*>(
                    hnsw->getDataByInternalId(static_cast<hnswlib::tableint>(i)));
                int nearest = 0;
                double best_score = std::numeric_limits<double>::max();
                for (int c = 0; c < k; ++c) {
                    double score = static_cast<double>(dist_func_i(vec, uint8_centroids[c].data(), dist_param_i)) + price[c];
                    if (score < best_score) { best_score = score; nearest = c; }
                }
                distortion += best_score;
                if (nearest == my_centroid) {
                    staying++;
                } else {
                    migrating++;
                    #pragma omp atomic
                    migration_per_centroid[nearest]++;
                }
                if (want_vec_sums) {
                    for (int d = 0; d < dim; ++d) {
                        #pragma omp atomic
                        vec_sum[(size_t)nearest * dim + d] += static_cast<double>(vec[d]);
                    }
                }
            }
        } else {
            hnswlib::L2Space l2_space(static_cast<size_t>(dim));
            auto dist_func = l2_space.get_dist_func();
            void* dist_param = l2_space.get_dist_func_param();

            #pragma omp parallel for schedule(static) reduction(+:staying,migrating,distortion)
            for (size_t i = 0; i < N; ++i) {
                if (hnsw->isMarkedDeleted(static_cast<hnswlib::tableint>(i))) continue;
                const float* vec = reinterpret_cast<const float*>(
                    hnsw->getDataByInternalId(static_cast<hnswlib::tableint>(i)));
                int nearest = 0;
                double best_score = std::numeric_limits<double>::max();
                for (int c = 0; c < k; ++c) {
                    double score = static_cast<double>(dist_func(vec, centroid_ptrs[c], dist_param)) + price[c];
                    if (score < best_score) { best_score = score; nearest = c; }
                }
                distortion += best_score;
                if (nearest == my_centroid) {
                    staying++;
                } else {
                    migrating++;
                    #pragma omp atomic
                    migration_per_centroid[nearest]++;
                }
                if (want_vec_sums) {
                    for (int d = 0; d < dim; ++d) {
                        #pragma omp atomic
                        vec_sum[(size_t)nearest * dim + d] += static_cast<double>(vec[d]);
                    }
                }
            }
        }

        response->set_total_points(staying + migrating);
        response->set_staying_points(staying);
        response->set_migrating_points(migrating);
        response->set_distortion(distortion);

        for (int c = 0; c < k; ++c) {
            if (migration_per_centroid[c] > 0) {
                auto* mc = response->add_migration_breakdown();
                mc->set_target_centroid_idx(c);
                mc->set_count(migration_per_centroid[c]);
            }
        }

        if (want_vec_sums) {
            for (int c = 0; c < k; ++c) {
                int64_t count_c = (c == my_centroid) ? staying : migration_per_centroid[c];
                if (count_c == 0) continue;
                std::vector<float> sum_f(dim);
                for (int d = 0; d < dim; ++d) sum_f[d] = static_cast<float>(vec_sum[(size_t)c * dim + d]);
                auto* cvs = response->add_centroid_vector_sums();
                cvs->set_centroid_idx(c);
                cvs->set_vector_sum_data(reinterpret_cast<const char*>(sum_f.data()), sum_f.size() * sizeof(float));
            }
        }

        // Write log file
        std::string log_dir = "indexes";
        std::filesystem::create_directories(log_dir);
        std::string log_path = log_dir + "/worker" + std::to_string(worker_id_) + "_migration_analysis.log";
        std::ofstream log_file(log_path);
        log_file << "worker_id=" << worker_id_ << "\n";
        log_file << "total_points=" << (staying + migrating) << "\n";
        log_file << "staying=" << staying << "\n";
        log_file << "migrating=" << migrating << "\n";
        double pct = (staying + migrating > 0) ? (100.0 * migrating / (staying + migrating)) : 0.0;
        log_file << "migration_pct=" << pct << "\n";
        log_file << "assigned_centroid_idx=" << my_centroid << "\n";
        log_file << "\n# breakdown: target_centroid_idx -> count\n";
        for (int c = 0; c < k; ++c) {
            if (migration_per_centroid[c] > 0) {
                log_file << "to_centroid_" << c << "=" << migration_per_centroid[c] << "\n";
            }
        }
        log_file.close();

        std::cout << "[Worker " << worker_id_ << "] AnalyzeMigration complete: "
                  << "total=" << (staying + migrating)
                  << ", staying=" << staying
                  << ", migrating=" << migrating
                  << " (" << pct << "%)"
                  << '\n';

        return grpc::Status::OK;
    } catch (const std::exception& e) {
        std::cout << "[Worker " << worker_id_ << "] AnalyzeMigration exception: "
                  << e.what() << '\n';
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "Exception in AnalyzeMigration: " + std::string(e.what()));
    }
}

template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::Rebalance(grpc::ServerContext* context,
                                                   const RebalanceRequest* request,
                                                   RebalanceResponse* response) {
    try {
        const int k = request->num_centroids();
        const int dim = request->dimension();
        const int my_centroid = request->assigned_centroid_idx();


        const float* centroids_flat = as_floats(request->centroids_data());

        // Per-centroid price bias; missing entries default to 0 so legacy callers are unaffected
        std::vector<double> price(k, 0.0);
        for (int c = 0; c < std::min(k, request->cluster_price_size()); ++c) price[c] = request->cluster_price(c);

        // Build centroid_idx -> worker_address map
        std::unordered_map<int, std::string> centroid_to_worker;
        for (const auto& m : request->centroid_to_worker()) {
            centroid_to_worker[m.centroid_idx()] = m.worker_address();
        }

        auto rebalance_t0 = std::chrono::steady_clock::now();

        // Step 1: Split index under lock so MergeIndex cannot materialize temp indexes
        // while the resident split children are still in memory. Merge RPCs may still
        // receive bytes to disk concurrently, but their loads wait on index_mutex_.
        struct TransferInfo {
            int centroid_idx;
            std::string worker_address;
            std::string file_path;
        };
        std::vector<TransferInfo> transfers;

        {
            std::lock_guard<std::mutex> lock(index_mutex_);
            auto* hnsw = cluster_index_->getHNSWIndex();
            const size_t N = hnsw->getCurrentElementCount();

            std::cout << "[Worker " << worker_id_ << "][Rebalance] Start: nodes="
                      << N << ", num_centroids=" << k << ", centroid=" << my_centroid
                      << '\n';

            if (N == 0) {
                std::cout << "[Worker " << worker_id_ << "] Index is empty, nothing to rebalance"
                          << '\n';
                response->set_success(true);
                return grpc::Status::OK;
            }

            // Classify all points by nearest centroid 
            auto classify_t0 = std::chrono::steady_clock::now();

            std::vector<int> point_assignment(N);

            if constexpr (std::is_same_v<dist_t, int>) {
                // Quantize k centroids to uint8 once, then compare stored uint8 data directly
                auto quant_t0 = std::chrono::steady_clock::now();
                typename hnswlib::SpaceFor<dist_t>::type int_space(dim_);
                auto dist_func_i = int_space.get_dist_func();
                void* dist_param_i = int_space.get_dist_func_param();
                std::vector<std::vector<unsigned char>> uint8_centroids(k, std::vector<unsigned char>(dim));
                for (int c = 0; c < k; ++c)
                    for (int d = 0; d < dim; ++d)
                        uint8_centroids[c][d] = static_cast<unsigned char>(
                            std::max(0L, std::min(255L, std::lround(static_cast<double>(centroids_flat[c * dim + d])))));
                auto quant_t1 = std::chrono::steady_clock::now();
                std::cout << "[Worker " << worker_id_ << "][Rebalance] Quantize: time="
                          << std::chrono::duration<double, std::milli>(quant_t1 - quant_t0).count()
                          << "ms, num_centroids=" << k << ", dim=" << dim << '\n';

                // Scores computed in double so all-zero price reproduces the pre-price nearest-centroid
                // choice exactly, even at high dimensions where the raw int distance exceeds float precision.
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < N; ++i) {
                    const unsigned char* vec = reinterpret_cast<const unsigned char*>(
                        hnsw->getDataByInternalId(static_cast<hnswlib::tableint>(i)));
                    int nearest = 0;
                    double best_score = std::numeric_limits<double>::max();
                    for (int c = 0; c < k; ++c) {
                        double score = static_cast<double>(dist_func_i(vec, uint8_centroids[c].data(), dist_param_i)) + price[c];
                        if (score < best_score) { best_score = score; nearest = c; }
                    }
                    point_assignment[i] = nearest;
                }
            } else {
                hnswlib::L2Space l2_space(static_cast<size_t>(dim));
                auto dist_func = l2_space.get_dist_func();
                void* dist_param = l2_space.get_dist_func_param();
                const float* flat_ptr = centroids_flat;

                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < N; ++i) {
                    const float* vec = reinterpret_cast<const float*>(
                        hnsw->getDataByInternalId(static_cast<hnswlib::tableint>(i)));
                    int nearest = 0;
                    double best_score = std::numeric_limits<double>::max();
                    for (int c = 0; c < k; ++c) {
                        const float* cent = flat_ptr + c * dim;
                        double score = static_cast<double>(dist_func(vec, cent, dist_param)) + price[c];
                        if (score < best_score) { best_score = score; nearest = c; }
                    }
                    point_assignment[i] = nearest;
                }
            }

            auto classify_t1 = std::chrono::steady_clock::now();
            double classify_ms = std::chrono::duration<double, std::milli>(classify_t1 - classify_t0).count();

            // Count per-destination
            std::set<int> destinations;
            size_t this_worker_count = 0;
            for (size_t i = 0; i < N; ++i) {
                if (point_assignment[i] == my_centroid) {
                    this_worker_count++;
                } else {
                    destinations.insert(point_assignment[i]);
                }
            }

            std::cout << "[Worker " << worker_id_ << "][Rebalance] Classify: stay="
                      << this_worker_count << ", move=" << (N - this_worker_count)
                      << ", destinations=" << destinations.size()
                      << ", time=" << classify_ms << "ms"
                      << '\n';

            // resident ghost edges loaded from disk
            bool ghost_loaded = false;
            double ghost_load_ms = 0.0;
            uintmax_t ghost_load_bytes = 0;
            {
                if (resident_ghost_links_on_disk_) {
                    ghost_load_bytes = getDirSize(resident_ghost_links_path_);
                }
                auto ghost_load_t0 = std::chrono::steady_clock::now();
                ghost_loaded = ensureResidentGhostLinksLoaded("rebalance split");
                auto ghost_load_t1 = std::chrono::steady_clock::now();
                if (ghost_loaded) {
                    ghost_load_ms = std::chrono::duration<double, std::milli>(ghost_load_t1 - ghost_load_t0).count();
                } else {
                    ghost_load_bytes = 0;
                }
            }

            std::cout << "[Worker " << worker_id_ << "][Rebalance] Resident ghost links: "
                      << (ghost_loaded ? "loaded" : "not loaded")
                      << ", size=" << ghost_load_bytes
                      << " bytes, load_time=" << ghost_load_ms << "ms"
                      << '\n';

            // Single-pass N-way split: child 0 is this worker, remaining children are destinations.
            auto split_t0 = std::chrono::steady_clock::now();
            std::vector<int> child_to_centroid;
            child_to_centroid.reserve(destinations.size() + 1);
            child_to_centroid.push_back(my_centroid);
            for (int dest_centroid : destinations) child_to_centroid.push_back(dest_centroid);

            std::unordered_map<int, uint32_t> centroid_to_child;
            centroid_to_child.reserve(child_to_centroid.size());
            for (uint32_t child = 0; child < child_to_centroid.size(); ++child) {
                centroid_to_child[child_to_centroid[child]] = child;
            }

            std::vector<uint32_t> split_assignment(N, 0);
            for (size_t i = 0; i < N; ++i) {
                split_assignment[i] = centroid_to_child[point_assignment[i]];
            }

            // K-1 spaces for transfer children (child 0 reuses original in-place)
            std::vector<std::unique_ptr<typename hnswlib::SpaceFor<dist_t>::type>> split_spaces;
            split_spaces.reserve(child_to_centroid.size() - 1);
            std::vector<hnswlib::SpaceInterface<dist_t>*> transfer_space_ptrs;
            transfer_space_ptrs.reserve(child_to_centroid.size() - 1);
            for (size_t child = 1; child < child_to_centroid.size(); ++child) {
                split_spaces.emplace_back(std::make_unique<typename hnswlib::SpaceFor<dist_t>::type>(dim));
                transfer_space_ptrs.push_back(split_spaces.back().get());
            }

            hnsw_splitter::HNSWSplitter<dist_t> splitter(hnsw);
            logMemoryUsage("Rebalance / before split", worker_id_);
            auto split1_t0 = std::chrono::steady_clock::now();
            auto split_children = splitter.splitInPlace(split_assignment, transfer_space_ptrs);
            auto split1_t1 = std::chrono::steady_clock::now();
            double split1_ms = std::chrono::duration<double, std::milli>(split1_t1 - split1_t0).count();
            logMemoryUsage("Rebalance / after split peak", worker_id_);

            const int64_t points_kept = hnsw->getCurrentElementCount();

            std::cout << "[Worker " << worker_id_ << "][Rebalance] Split: kept=" << points_kept
                      << ", split time=" << split1_ms << "ms"
                      << '\n';

            std::filesystem::create_directories(scratch_dir_);

            struct SavedChildIndex {
                bool success = false;
                int dest_centroid = -1;
                int64_t count = 0;
                size_t ghost_vectors = 0;
                std::string dest_addr;
                std::string path;
            };

            auto save_phase_t0 = std::chrono::steady_clock::now();
            std::vector<SavedChildIndex> saved_children(split_children.size());
            std::vector<std::future<void>> save_futures(split_children.size());

            // Save child indexes to disk in parallel. 
            for (size_t ci = 0; ci < split_children.size(); ++ci) {
                auto& out_idx = split_children[ci];
                auto& saved = saved_children[ci];
                saved.dest_centroid = child_to_centroid[ci + 1];
                saved.count = out_idx->getCurrentElementCount();
                if (saved.count == 0) continue;

                saved.ghost_vectors = out_idx->getGhostLinkedVectorCount();
                saved.dest_addr = centroid_to_worker.count(saved.dest_centroid)
                    ? centroid_to_worker[saved.dest_centroid] : "";
                saved.path = scratch_dir_ + "/worker_" + std::to_string(worker_id_)
                           + "_rebal_to_c" + std::to_string(saved.dest_centroid) + ".bin";

                auto* out_idx_ptr = out_idx.get();
                const std::string path = saved.path;
                save_futures[ci] = std::async(std::launch::async, [out_idx_ptr, path]() {
                    out_idx_ptr->saveIndex(path);
                });
                saved.success = true;
            }

            for (size_t ci = 0; ci < saved_children.size(); ++ci) {
                auto& saved = saved_children[ci];
                if (!saved.success) continue;

                save_futures[ci].get();
                transfers.push_back({saved.dest_centroid, saved.dest_addr, saved.path});

                std::cout << "[Worker " << worker_id_ << "][Rebalance] Export: centroid="
                          << saved.dest_centroid << ", points=" << saved.count
                          << ", ghost_vectors=" << saved.ghost_vectors
                          << ", dest=" << saved.dest_addr
                          << '\n';
            }
            auto save_phase_t1 = std::chrono::steady_clock::now();
            const double save_ms =
                std::chrono::duration<double, std::milli>(save_phase_t1 - save_phase_t0).count();

            auto split_t1 = std::chrono::steady_clock::now();
            double split_ms = std::chrono::duration<double, std::milli>(split_t1 - split_t0).count();

            // Free child indexes now that they're saved to disk
            split_children.clear();
            logMemoryUsage("Rebalance / after child save", worker_id_);

            std::cout << "[Worker " << worker_id_ << "][Rebalance] Split done: kept="
                      << points_kept << ", total transfers=" << transfers.size()
                      << ", split=" << split1_ms << "ms"
                      << ", save=" << save_ms << "ms"
                      << ", total (including split and save)=" << split_ms << "ms"
                      << '\n';
        } // index_mutex_ released here; MergeIndex can now load and apply temp indexes

        // ===== Step 2: Stream sub-indexes to destinations in parallel (no lock needed) =====
        auto transfer_t0 = std::chrono::steady_clock::now();

        auto streamToWorker = [this](const TransferInfo& t) -> bool {
            const auto transfer_started_at = std::chrono::steady_clock::now();
            auto transfer_elapsed_ms = [&]() {
                return std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - transfer_started_at).count();
            };

            if (t.worker_address.empty()) {
                std::cout << "[Worker " << worker_id_ << "][Rebalance] Transfer skip: centroid="
                          << t.centroid_idx << ", reason=no worker address"
                          << ", time=" << transfer_elapsed_ms() << "ms" << '\n';
                return false;
            }

            std::cout << "[Worker " << worker_id_ << "][Rebalance] Transfer start: centroid="
                      << t.centroid_idx << ", dest=" << t.worker_address
                      << '\n';

            try {
                auto channel = grpc::CreateChannel(t.worker_address, grpc::InsecureChannelCredentials());
                auto stub = WorkerService::NewStub(channel);
                grpc::ClientContext ctx;
                LoadIndexResponse merge_resp;
                auto writer = stub->MergeIndex(&ctx, &merge_resp);

                std::ifstream in(t.file_path, std::ios::binary);
                if (!in.is_open()) {
                    std::cout << "[Worker " << worker_id_ << "][Rebalance] Transfer error: centroid="
                              << t.centroid_idx << ", reason=failed to open " << t.file_path
                              << ", time=" << transfer_elapsed_ms() << "ms" << '\n';
                    return false;
                }

                const size_t chunk_size = 1 << 20; // 1MB
                std::vector<char> buf(chunk_size);
                while (in) {
                    in.read(buf.data(), buf.size());
                    std::streamsize got = in.gcount();
                    if (got <= 0) break;
                    IndexChunk ch;
                    ch.set_data(buf.data(), (int)got);
                    if (!writer->Write(ch)) {
                        std::cout << "[Worker " << worker_id_ << "][Rebalance] Transfer stream failed: centroid="
                                  << t.centroid_idx << ", dest=" << t.worker_address
                                  << ", time=" << transfer_elapsed_ms() << "ms" << '\n';
                        break;
                    }
                }
                writer->WritesDone();
                auto st = writer->Finish();

                std::filesystem::remove(t.file_path);

                if (st.ok() && merge_resp.success()) {
                    std::cout << "[Worker " << worker_id_ << "][Rebalance] Transfer ok: centroid="
                              << t.centroid_idx << ", dest=" << t.worker_address
                              << ", time=" << transfer_elapsed_ms() << "ms" << '\n';
                    return true;
                }
                std::cout << "[Worker " << worker_id_ << "][Rebalance] Transfer failed: centroid="
                          << t.centroid_idx << ", dest=" << t.worker_address
                          << ", error=" << st.error_message()
                          << ", time=" << transfer_elapsed_ms() << "ms"
                          << '\n';
                return false;
            } catch (const std::exception& e) {
                std::cout << "[Worker " << worker_id_ << "][Rebalance] Transfer exception: centroid="
                          << t.centroid_idx << ", dest=" << t.worker_address
                          << ", error=" << e.what()
                          << ", time=" << transfer_elapsed_ms() << "ms" << '\n';
                return false;
            }
        };

        bool transfers_ok = true;
        if (transfers.size() == 1) {
            transfers_ok = streamToWorker(transfers[0]);
        } else if (transfers.size() > 1) {
            std::vector<std::future<bool>> transfer_futures;
            transfer_futures.reserve(transfers.size());
            for (const auto& t : transfers) {
                transfer_futures.emplace_back(std::async(std::launch::async, streamToWorker, std::cref(t)));
            }
            for (auto& f : transfer_futures) {
                transfers_ok = f.get() && transfers_ok;
            }
        }

        auto transfer_t1 = std::chrono::steady_clock::now();
        double transfer_ms = std::chrono::duration<double, std::milli>(transfer_t1 - transfer_t0).count();

        std::cout << "[Worker " << worker_id_ << "][Rebalance] Done: transfer took=" << transfer_ms << '\n';

        auto rebalance_t1 = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(rebalance_t1 - rebalance_t0).count();

        response->set_success(transfers_ok);

        std::cout << "[Worker " << worker_id_ << "][Rebalance] Done: total time for split and transfer=" << total_ms << "ms, success="
                  << (transfers_ok ? "true" : "false")
                  << '\n';

        return grpc::Status::OK;
    } catch (const std::exception& e) {
        std::cout << "[Worker " << worker_id_ << "] Rebalance exception: "
                  << e.what() << '\n';
        response->set_success(false);
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "Exception in Rebalance: " + std::string(e.what()));
    }
}

template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::GetCurrentElementCount(grpc::ServerContext* context,
                                                                const GetCurrentElementCountRequest* request,
                                                                GetCurrentElementCountResponse* response) {
    (void)context;
    (void)request;

    // Signifies end of rebalance operation; resident ghost edges can be saved to disk
    logMemoryUsage("before file write start of GetCurrentElementCount", worker_id_);
    // no close here any more: saveGhostLinks shares ghostDb(), so the handle stays open and the
    // next AddGhostEdges does not pay a reopen
    std::lock_guard<std::mutex> lock(index_mutex_);
    auto* hnsw = cluster_index_->getHNSWIndex();
    bool ghost_saved = false;
    double ghost_save_ms = 0.0;
    uintmax_t ghost_save_bytes = 0;
    if (hnsw->ghost_links_in_memory_) {
        auto ghost_save_t0 = std::chrono::steady_clock::now();
        ghost_saved = persistResidentGhostLinks("rebalance finalization");
        auto ghost_save_t1 = std::chrono::steady_clock::now();
        if (ghost_saved) {
            ghost_save_ms = std::chrono::duration<double, std::milli>(ghost_save_t1 - ghost_save_t0).count();
            ghost_save_bytes = resident_ghost_links_on_disk_
                ? getDirSize(resident_ghost_links_path_)
                : 0;
        }
    }
    logMemoryUsage("after ghost file write; end of GetCurrentElementCount", worker_id_);
    response->set_element_count(static_cast<int64_t>(hnsw->getCurrentElementCount()));
    if (ghost_saved) {
        std::cout << "[Worker " << worker_id_ << "][Ghost] Finalized (rebalance finalization): size="
                  << ghost_save_bytes << " bytes, save=" << ghost_save_ms << "ms"
                  << '\n';
    }
    return grpc::Status::OK;
}

template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::GetEdgeStats(grpc::ServerContext* context,
                                                      const GetEdgeStatsRequest* request,
                                                      EdgeStatsResponse* response) {
    (void)context;
    (void)request;

    // Release the persistent write handle so the read-only scan below doesn't
    // collide with it (same pattern GetCurrentElementCount uses).
    
    std::lock_guard<std::mutex> lock(index_mutex_);
    auto* hnsw = cluster_index_->getHNSWIndex();

    response->set_element_count(static_cast<int64_t>(hnsw->getCurrentElementCount()));
    response->set_normal_edge_count(static_cast<int64_t>(hnsw->getNormalLinkCount()));

    // Ghost edges: scanned directly off the on-disk RocksDB file rather than the
    // live in-memory ghost_links_ structure, since incremental AddGhostEdges
    // appends write straight to disk and never touch that in-memory copy (see
    // ghost_db_stats.h for the full reasoning).
    if (resident_ghost_links_on_disk_) {
        GhostDbStats ghost_stats = scanGhostLinkDb(ghostDb());
        response->set_ghost_edge_count(static_cast<int64_t>(ghost_stats.ghost_edge_count));
        response->set_ghost_linked_vector_count(static_cast<int64_t>(ghost_stats.ghost_linked_vector_count));
    } else {
        response->set_ghost_edge_count(0);
        response->set_ghost_linked_vector_count(0);
    }

    std::cout << "[Worker " << worker_id_ << "][EdgeStats] element_count=" << response->element_count()
              << " normal_edge_count=" << response->normal_edge_count()
              << " ghost_edge_count=" << response->ghost_edge_count()
              << " ghost_linked_vector_count=" << response->ghost_linked_vector_count()
              << '\n';

    return grpc::Status::OK;
}

template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::MergeIndex(grpc::ServerContext* context,
                                                    grpc::ServerReader<IndexChunk>* reader,
                                                    LoadIndexResponse* response) {
    try {
        std::string peer = context->peer();
        auto merge_total_t0 = std::chrono::steady_clock::now();
        std::cout << "[Worker " << worker_id_ << "][Merge] Start: peer=" << peer
                  << '\n';

        // --- Step 1: Receive the index file (no lock needed, runs concurrently) ---
        std::filesystem::create_directories(scratch_dir_);

        uint64_t file_id = g_merge_file_counter.fetch_add(1);
        std::string tmp_path = scratch_dir_ + "/merge_incoming_" + std::to_string(worker_id_)
                             + "_" + std::to_string(file_id) + ".bin";

        size_t total_bytes = 0;
        double receive_ms = 0.0;
        {
            std::ofstream out(tmp_path, std::ios::binary);
            if (!out.is_open()) {
                std::cout << "[Worker " << worker_id_ << "] Failed to open temp file: " << tmp_path
                          << '\n';
                response->set_success(false);
                return grpc::Status::OK;
            }

            auto receive_t0 = std::chrono::steady_clock::now();
            IndexChunk chunk;
            while (reader->Read(&chunk)) {
                const std::string& d = chunk.data();
                out.write(d.data(), (std::streamsize)d.size());
                total_bytes += d.size();
            }
            out.close();
            receive_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - receive_t0).count();

            std::cout << "[Worker " << worker_id_ << "][Merge] Receive: bytes=" << total_bytes
                      << ", receive_ms=" << receive_ms << '\n';
        }

        // --- Step 2: Load temp HNSW and merge while holding index_mutex_ ---
        // This prevents large incoming temp indexes from being materialized in memory
        // while Rebalance is holding split children in memory.
        {
            std::lock_guard<std::mutex> lock(index_mutex_);
            
            auto load_t0 = std::chrono::steady_clock::now();
            auto temp_space = std::make_unique<typename hnswlib::SpaceFor<dist_t>::type>(dim_);
            auto temp_hnsw = std::make_unique<hnswlib::HierarchicalNSW<dist_t>>(
                temp_space.get(), tmp_path); // loads both hnsw and ghost links
            size_t incoming_count = temp_hnsw->getCurrentElementCount();
            size_t incoming_ghost_vectors = temp_hnsw->getGhostLinkedVectorCount();
            std::filesystem::remove(tmp_path);
            auto load_t1 = std::chrono::steady_clock::now();
            double load_ms = std::chrono::duration<double, std::milli>(load_t1 - load_t0).count();

            auto repair_t0 = std::chrono::steady_clock::now();
            temp_hnsw->repairL0ZeroDegreeParallel();
            auto repair_t1 = std::chrono::steady_clock::now();
            double repair_ms = std::chrono::duration<double, std::milli>(repair_t1 - repair_t0).count();

            std::cout << "[Worker " << worker_id_ << "][Merge] Incoming: points="
                    << incoming_count << ", ghost_vectors=" << incoming_ghost_vectors
                    << ", L0 zero-degree repair time=" << repair_ms << "ms"
                    << ", load incoming index time (both hnsw and ghost links)=" << load_ms << "ms"
                    << '\n';

        // --- Step 3: Merge into main index ---
        
            auto* hnsw = cluster_index_->getHNSWIndex();
            size_t current_count = hnsw->getCurrentElementCount();

            // load resident ghost links from disk
            // if this is a new machine in a merge operation, there won't be any resident ghost links on disk
            bool resident_ghost_loaded = false;
            double resident_ghost_load_ms = 0.0;
            uintmax_t resident_ghost_load_bytes = 0;
            if (current_count > 0) {
                if (resident_ghost_links_on_disk_) {
                    resident_ghost_load_bytes = getDirSize(resident_ghost_links_path_);
                }
                auto ghost_load_t0 = std::chrono::steady_clock::now();
                resident_ghost_loaded = ensureResidentGhostLinksLoaded("merge");
                auto ghost_load_t1 = std::chrono::steady_clock::now();
                if (resident_ghost_loaded) {
                    resident_ghost_load_ms = std::chrono::duration<double, std::milli>(ghost_load_t1 - ghost_load_t0).count();
                } else {
                    resident_ghost_load_bytes = 0;
                }
            }

            if (current_count == 0) {
                std::cout << "[Worker " << worker_id_ << "] Index is empty, loading as initial index"
                          << '\n';
                // clean up any resident ghost links on disk from a previous run, if any
                // std::filesystem::remove(resident_ghost_links_path_);
                // resident_ghost_links_on_disk_ = false;
            
                cluster_index_->replaceIndex(std::move(temp_space), std::move(temp_hnsw));
                logMemoryUsage("before resizeIndex (Merge replace)", worker_id_);
                cluster_index_->getHNSWIndex()->resizeIndex(max_elements_);
                logMemoryUsage("after resizeIndex (Merge replace)", worker_id_);

                std::cout << "[Worker " << worker_id_ << "][Merge] Apply: mode=replace (ghost edges come from incoming index, are not loaded from the resident ghost file), add="
                          << incoming_count << ", resident_ghost_load="
                          << resident_ghost_load_ms << "ms, resident_ghost_size="
                          << resident_ghost_load_bytes << " bytes" << '\n';
            } else {
                std::cout << "[Worker " << worker_id_ << "][Merge] Apply: before="
                          << current_count << ", add=" << incoming_count
                          << '\n';
                auto merge_t0 = std::chrono::steady_clock::now();
                hnsw->mergeIndex(*temp_hnsw);
                auto merge_t1 = std::chrono::steady_clock::now();
                double merge_ms = std::chrono::duration<double, std::milli>(merge_t1 - merge_t0).count();
                std::cout << "[Worker " << worker_id_ << "][Merge] After merging incoming index into resident index: after="
                          << hnsw->getCurrentElementCount()
                          << " (resident_ghost_load=" << resident_ghost_load_ms
                          << "ms, resident_ghost_size=" << resident_ghost_load_bytes
                          << " bytes, merge=" << merge_ms << "ms)" << '\n';
            }
        }

        double merge_total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - merge_total_t0).count();
        std::cout << "[Worker " << worker_id_ << "][Merge] Done: bytes=" << total_bytes
                  << ", receive_ms=" << receive_ms
                  << ", total_ms=" << merge_total_ms << '\n';

        response->set_success(true);
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        std::cout << "[Worker " << worker_id_ << "] MergeIndex exception: "
                  << e.what() << '\n';
        response->set_success(false);
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "Exception in MergeIndex: " + std::string(e.what()));
    }
}

struct QueryResult {
    int query_id;
    std::vector<std::pair<float, int>> neighbors;
};

template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::BatchedSearch(grpc::ServerContext* context, const BatchedQuery* request, BatchedWorkerResult* response) {
    auto total_t0 = std::chrono::steady_clock::now();
    response->set_worker_id(worker_id_);

    const int num_queries = request->batch_size();
    const float* all_q   = as_floats(request->vectors_data());
    const int    q_dim   = num_queries > 0
        ? static_cast<int>(request->vectors_data().size()) / (num_queries * static_cast<int>(sizeof(float)))
        : 0;
    const int    local_k = request->k();
    double pb_construction_ms = 0.0;

    std::vector<QueryResult> query_results(num_queries);

    double search_ms = 0.0;

    // Parallel search
    auto search_start = std::chrono::steady_clock::now();
    if (request->has_efsearch()) {
        cluster_index_->getHNSWIndex()->setEf(static_cast<size_t>(request->efsearch()));
    }
    // std::cout << "[Worker " << worker_id_ << "] BatchedSearch batch_size=" << num_queries
    //           << " efsearch=";
    if (request->has_efsearch()) {
        std::cout << request->efsearch() << " (explicit)";
    } else {
        std::cout << cluster_index_->getHNSWIndex()->ef_ << " (current/default)";
    }
    std::cout << '\n';
    #pragma omp parallel for
    for (int i = 0; i < num_queries; ++i) {
        auto neighbors = cluster_index_->searchKNN(all_q + i * q_dim, local_k);
        query_results[i].query_id = request->query_ids(i);
        query_results[i].neighbors = std::move(neighbors);
    }

    auto search_end = std::chrono::steady_clock::now();
    search_ms += std::chrono::duration<double, std::milli>(search_end - search_start).count();
    
    auto pb_construction_start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_queries; ++i) {
        WorkerResult* worker_result = response->add_results();
        worker_result->set_worker_id(worker_id_);
        worker_result->set_query_id(query_results[i].query_id);

        for (const auto& [distance, id] : query_results[i].neighbors) {
            auto* neighbor = worker_result->add_results();
            neighbor->set_id(id);
            neighbor->set_distance(distance);
        }
    }
    auto pb_construction_end = std::chrono::steady_clock::now();
    pb_construction_ms += std::chrono::duration<double, std::milli>(pb_construction_end - pb_construction_start).count();
    auto total_t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();
        
    // std::cout << "[Worker " << worker_id_ << "] BatchedSearch timing: total_ms=" << total_ms
    //           << " search_ms=" << search_ms
    //           << " pb_construction_ms=" << pb_construction_ms
    //           << " batch_size=" << num_queries << '\n';
    
    return grpc::Status::OK;
}

template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::Insert(grpc::ServerContext* context, const InsertData* request, InsertAck* response) {
    // Measure compute time for local insert
    auto t0 = std::chrono::steady_clock::now();
    
    const float* vector_data = as_floats(request->vector_data());

    bool success = cluster_index_->insertVector(request->id(), vector_data);
    
    if (success) {
        auto* hnsw_index = cluster_index_->getHNSWIndex();
        int current_nodes = hnsw_index->getCurrentElementCount();
        if (request->id() % 10000 == 0) {
            std::cout << "[Worker " << worker_id_ << "] Insert successful for id=" << request->id() << ", current nodes=" << current_nodes << '\n';
        }
    } else {
        std::cout << "[Worker " << worker_id_ << "] Insert failed for id=" 
                  << request->id() << '\n';
    }
    response->set_success(success);

    auto t1 = std::chrono::steady_clock::now();
    const double insert_compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    std::string filename = "/tmp/worker" + std::to_string(worker_id_) + "_insert.csv";
    std::ofstream wi_csv(filename, std::ios::app);
    if (wi_csv.tellp() == 0) {
        wi_csv << "query_id,worker_id,worker_insert_compute_ms" << '\n';
    }
    wi_csv << request->id() << ',' << worker_id_ << ',' << insert_compute_ms << '\n';
    return grpc::Status::OK;
}

template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::BatchedInsert(grpc::ServerContext* context, const BatchedInsertData* request, BatchedInsertAck* response) {
    auto total_t0 = std::chrono::steady_clock::now();
    const int num_inserts = request->batch_size();
    double capacity_check_ms = 0.0;
    double resize_ms = 0.0;
    double insert_ms = 0.0;
    double pb_construction_ms = 0.0;
    auto pb_construction_start = std::chrono::steady_clock::now();
    std::vector<unsigned char> results(num_inserts, 0);
    auto pb_construction_end = std::chrono::steady_clock::now();
    pb_construction_ms += std::chrono::duration<double, std::milli>(pb_construction_end - pb_construction_start).count();

    auto capacity_check_start = std::chrono::steady_clock::now();
    auto* hnsw_index = cluster_index_->getHNSWIndex();
    size_t current_count = hnsw_index->getCurrentElementCount();
    size_t max_elems = hnsw_index->getMaxElements();
    if (current_count + static_cast<size_t>(num_inserts) >= max_elems) {
        size_t new_max = static_cast<size_t>(max_elems * 1.2);
        std::cout << "[Worker " << worker_id_ << "] Resizing index from " << max_elems
                    << " to " << new_max << " (current=" << current_count
                    << ", batch=" << num_inserts << ")" << '\n';
        logMemoryUsage("before resizeIndex (BatchedInsert growth)", worker_id_);
        auto resize_start = std::chrono::steady_clock::now();
        hnsw_index->resizeIndex(new_max);
        resize_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - resize_start).count();
        logMemoryUsage("after resizeIndex (BatchedInsert growth)", worker_id_);
        max_elements_ = static_cast<int>(new_max);
    }
    capacity_check_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - capacity_check_start).count();

    auto insert_start = std::chrono::steady_clock::now();
    const float* all_vecs = as_floats(request->vectors_data());

    #pragma omp parallel for
    for (int i = 0; i < num_inserts; ++i) {
        const int id = request->ids(i);
        results[i] = cluster_index_->insertVector(id, all_vecs + i * dim_) ? 1 : 0;

        if (results[i] != 0) {
            auto* hnsw_index = cluster_index_->getHNSWIndex();
            int current_nodes = hnsw_index->getCurrentElementCount();
            if(current_nodes % 10000000 == 0){
                logMemoryUsage("after insert id=" + std::to_string(id), worker_id_);
            }
            if (id % 100000 == 0) {
                std::cout << "[Worker " << worker_id_ << "] Insert successful for id=" << id << ", current nodes=" << current_nodes << '\n';
            }
        } else {
            std::cout << "[Worker " << worker_id_ << "] Insert failed for id="
                      << id << '\n';
        }
    }

    auto insert_end = std::chrono::steady_clock::now();
    insert_ms += std::chrono::duration<double, std::milli>(insert_end - insert_start).count();

    pb_construction_start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_inserts; ++i) {
        InsertAck* ack = response->add_acks();
        ack->set_success(results[i] != 0);
    }

    pb_construction_end = std::chrono::steady_clock::now();
    pb_construction_ms += std::chrono::duration<double, std::milli>(pb_construction_end - pb_construction_start).count();

    auto total_t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();

     // write timings to csv

    {
        static std::atomic<uint64_t> batch_counter{0};
        const uint64_t batch_idx = batch_counter.fetch_add(1) + 1;
        std::string path = "/tmp/worker" + std::to_string(worker_id_) + "_batched_insert.csv";
        if (batch_idx == 1) {
            std::ofstream csv(path);
            csv << "batch_idx,worker_id,batch_size,total_ms,capacity_check_ms,resize_ms,insert_ms,pb_construction_ms\n";
        }
        std::ofstream csv(path, std::ios::app);
        csv << batch_idx << ',' << worker_id_ << ',' << num_inserts << ',' << total_ms << ','
            << capacity_check_ms << ',' << resize_ms << ',' << insert_ms << ',' << pb_construction_ms << '\n';
    }

    // std::cout << "[Worker " << worker_id_ << "] BatchedInsert timing: total_ms=" << total_ms
    //           << " capacity_check_ms=" << capacity_check_ms
    //           << " resize_ms=" << resize_ms
    //           << " insert_ms=" << insert_ms
    //           << " pb_construction_ms=" << pb_construction_ms
    //           << " batch_size=" << num_inserts << '\n';

    return grpc::Status::OK;
}

template<typename dist_t>
grpc::Status WorkerServiceImpl<dist_t>::AddGhostEdges(grpc::ServerContext* context,
                                                       const BatchedAddGhostEdgesRequest* request,
                                                       BatchedAddGhostEdgesAck* response) {
    auto t0 = std::chrono::steady_clock::now();
    const int n = request->entries_size();
    std::vector<std::pair<int64_t, std::vector<GhostEdge>>> batch_ghost;    // ext label, ghost edges
    batch_ghost.reserve(n);
    for (const auto& entry : request->entries()) {
        if (entry.ghost_neighbors_size() == 0) continue;
        std::vector<GhostEdge> edges;
        edges.reserve(entry.ghost_neighbors_size());
        for (const auto& nb : entry.ghost_neighbors()) {
            edges.push_back({nb.id(), nb.distance()});
        }
        batch_ghost.push_back({entry.inserted_id(), std::move(edges)});
    }
    if (!batch_ghost.empty()) {
        flushInsertionGhostBuf(batch_ghost);
    }
    double flush_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // write timings to csv

    {
        static std::atomic<uint64_t> batch_counter{0};
        const uint64_t batch_idx = batch_counter.fetch_add(1) + 1;
        std::string path = "/tmp/worker" + std::to_string(worker_id_) + "_add_ghost_edges.csv";
        if (batch_idx == 1) {
            std::ofstream csv(path);
            csv << "batch_idx,worker_id,entries,non_empty,flush_ms\n";
        }
        std::ofstream csv(path, std::ios::app);
        csv << batch_idx << ',' << worker_id_ << ',' << n << ',' << batch_ghost.size() << ',' << flush_ms << '\n';
    }

    std::cout << "[Worker " << worker_id_ << "] AddGhostEdges: entries=" << n
              << " non_empty=" << batch_ghost.size()
              << " flush_ms=" << flush_ms << '\n';
    response->set_success(true);
    return grpc::Status::OK;
}

// ---------------- Explicit instantiations ----------------

template class WorkerServiceImpl<float>;
template class WorkerServiceImpl<int>;

// ---------------- Worker Server Wrapper ----------------

WorkerServer::WorkerServer(const std::string& server_address, int worker_id, int dim, int max_elements, const std::string& vector_type, std::string scratch_dir)
    : server_address_(server_address), worker_id_(worker_id) {
    if (vector_type == "uint8") {
        service_ = std::make_unique<WorkerServiceImpl<int>>(
            std::make_unique<ClusterIndex<int>>(dim, max_elements, 32, 100, 100),
            worker_id, dim, max_elements, scratch_dir);
    } else {
        service_ = std::make_unique<WorkerServiceImpl<float>>(
            std::make_unique<ClusterIndex<float>>(dim, max_elements, 32, 100, 100),
            worker_id, dim, max_elements, scratch_dir);
    }

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());

    grpc_server_ = builder.BuildAndStart();
    if (!grpc_server_) {
        throw std::runtime_error("Failed to start worker server");
    }
}

WorkerServer::~WorkerServer() {
    if (grpc_server_) {
        grpc_server_->Shutdown();
    }
}

void WorkerServer::Run() {
    std::cout << "[Worker " << worker_id_ << "] Listening on " 
              << server_address_ << '\n';
    grpc_server_->Wait();
}

void WorkerServer::Shutdown() {
    if (grpc_server_) {
        grpc_server_->Shutdown();
    }
}


} // namespace ann

#ifndef ANN_BUILDING_TESTS
// ---------------- Main with Signal Handling ----------------

static ann::WorkerServer* global_server = nullptr;

void signalHandler(int signum) {
    std::cerr << "[Worker] Signal " << signum << " received. ";
    if (signum == SIGSEGV) {
        std::cerr << "SEGMENTATION FAULT DETECTED!" << '\n';
        std::cerr << "[Worker] This is likely a memory corruption or null pointer dereference." << '\n';
        std::cerr << "[Worker] Check core dump if enabled. Use: gdb build/worker_server core.XXXX" << '\n';
        std::cerr.flush();
        // Re-raise signal to get core dump
        std::signal(signum, SIG_DFL);
        std::raise(signum);
    } else if (signum == SIGABRT) {
        std::cerr << "ABORT SIGNAL DETECTED!" << '\n';
        std::cerr.flush();
        // Re-raise signal to get core dump
        std::signal(signum, SIG_DFL);
        std::raise(signum);
    } else if (signum == SIGINT || signum == SIGTERM) {
        std::cout << "Shutting down worker..." << '\n';
        // Request shutdown (non-blocking, safe from signal handler)
        if (global_server) {
            global_server->Shutdown();
        }
    } else {
        std::cout << "Shutting down worker..." << '\n';
        if (global_server) {
            global_server->Shutdown();
        }
    }
}

static std::string getDefaultScratchDir() {
    const char* home = std::getenv("HOME");
    return std::string(home ? home : "") + "/indexes";
}

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    int worker_id = 1;
    std::string server_address = "0.0.0.0:50052";
    int dim = 128;
    int max_elements = 110000000;
    std::string scratch_dir = getDefaultScratchDir();
    std::string vector_type = "float32";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--worker_id" && i + 1 < argc) {
            worker_id = std::stoi(argv[++i]);
        } else if (arg == "--port" && i + 1 < argc) {
            server_address = "0.0.0.0:" + std::string(argv[++i]);
        } else if (arg == "--dim" && i + 1 < argc) {
            dim = std::stoi(argv[++i]);
        } else if (arg == "--max-elements" && i + 1 < argc) {
            max_elements = std::stoi(argv[++i]);
        } else if (arg == "--scratch-dir" && i + 1 < argc) {
            scratch_dir = argv[++i];
        } else if (arg == "--vector-type" && i + 1 < argc) {
            vector_type = argv[++i];
        }
    }

    try {
        ann::WorkerServer server(server_address, worker_id, dim, max_elements, vector_type, scratch_dir);
        global_server = &server;

        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);
        std::signal(SIGSEGV, signalHandler);  
        std::signal(SIGABRT, signalHandler);  

        server.Run();
        // After Wait() returns (due to Shutdown()), cleanup happens in destructor
    } catch (const std::exception& e) {
        std::cerr << "Worker server failed: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
#endif // ANN_BUILDING_TESTS
