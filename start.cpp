#include <grpcpp/grpcpp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>

#include "proto/ann.pb.h"
#include "proto/ann.grpc.pb.h"
#include "include/sift_utils.h"
#include "include/bin_utils.h"
#include "include/vec_bytes.h"

using Clock = std::chrono::steady_clock;

struct BenchArgs {
    std::string master_address = "127.0.0.1:50051";
    std::string base_path;   // base vectors: .fvecs | .bvecs | directory (SpaceV bin parts)
    std::string query_path;  // query vectors: .fvecs | .bvecs | .bin (SpaceV)
    std::string gt_path;     // ground truth:  .ivecs | .bin (SpaceV) — optional
    int k = 10;
    int max_insert = -1;
    int max_query  = -1;
    int batch_size = 1;
    int insert_batch_size = 1;
    int efsearch = -1;
    int router_k = -1;
    std::vector<int> efsearch_values;
    std::vector<int> router_k_values;
    std::string csv_path;  // optional: write sweep results here
};

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << "\n"
        << "  --master HOST:PORT\n"
        << "  --base   PATH          base vectors (.fvecs | .bvecs | directory for SpaceV bin)\n"
        << "  --query  PATH          query vectors (.fvecs | .bvecs | .bin)\n"
        << "  --gt     PATH          ground truth  (.ivecs | .bin) [optional]\n"
        << "  --k      K             number of neighbors (default 10)\n"
        << "  --max-insert N         limit insertions (-1 = all)\n"
        << "  --max-query  N         limit queries    (-1 = all)\n"
        << "  --batch-size N         query batch size (default 1)\n"
        << "  --insert-batch-size N  insert batch size (default 1)\n"
        << "  --efsearch N           worker ef_search override\n"
        << "  --router-k N           router HNSW k override\n"
        << "  --efsearch-values V,.. comma-separated efsearch sweep\n"
        << "  --router-k-values V,.. comma-separated router_k sweep\n";
}

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ','))
        if (!token.empty()) result.push_back(std::stoi(token));
    return result;
}

static bool parse_args(int argc, char** argv, BenchArgs& out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--master"             && i+1 < argc) out.master_address    = argv[++i];
        else if (a == "--base"               && i+1 < argc) out.base_path         = argv[++i];
        else if (a == "--query"              && i+1 < argc) out.query_path        = argv[++i];
        else if (a == "--gt"                 && i+1 < argc) out.gt_path           = argv[++i];
        else if (a == "--k"                  && i+1 < argc) { out.k = std::atoi(argv[++i]); if (out.k <= 0) out.k = 10; }
        else if (a == "--max-insert"         && i+1 < argc) out.max_insert        = std::atoi(argv[++i]);
        else if (a == "--max-query"          && i+1 < argc) out.max_query         = std::atoi(argv[++i]);
        else if (a == "--batch-size"         && i+1 < argc) { out.batch_size = std::atoi(argv[++i]); if (out.batch_size <= 0) out.batch_size = 1; }
        else if (a == "--insert-batch-size"  && i+1 < argc) { out.insert_batch_size = std::atoi(argv[++i]); if (out.insert_batch_size <= 0) out.insert_batch_size = 1; }
        else if (a == "--efsearch"           && i+1 < argc) out.efsearch          = std::atoi(argv[++i]);
        else if (a == "--router-k"           && i+1 < argc) out.router_k          = std::atoi(argv[++i]);
        else if (a == "--efsearch-values"    && i+1 < argc) out.efsearch_values   = parse_int_list(argv[++i]);
        else if (a == "--router-k-values"    && i+1 < argc) out.router_k_values   = parse_int_list(argv[++i]);
        else if (a == "--csv"                && i+1 < argc) out.csv_path          = argv[++i];
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return false; }
    }
    if (out.base_path.empty() || out.query_path.empty()) {
        std::cerr << "Error: --base and --query are required.\n";
        print_usage(argv[0]);
        return false;
    }
    return true;
}

// Return lowercase extension including the dot, or "" for directories.
static std::string file_ext(const std::string& path) {
    if (std::filesystem::is_directory(path)) return "";
    std::string e = std::filesystem::path(path).extension().string();
    for (auto& c : e) c = static_cast<char>(std::tolower(c));
    return e;
}

int main(int argc, char** argv) {
    BenchArgs args;
    if (!parse_args(argc, argv, args)) return 1;

    const int max_insert = args.max_insert;
    const int max_query  = args.max_query;

    // gRPC channel and stub
    auto channel = grpc::CreateChannel(args.master_address, grpc::InsecureChannelCredentials());
    auto stub    = ann::MasterService::NewStub(channel);

    // ── Insert base vectors (always chunked) ─────────────────────────────
    size_t inserted_ok = 0;
    size_t total_inserted = 0;
    double insert_sec = 0.0;
    double insert_eps = 0.0;

    {
        const size_t chunk_size = 10000;

        // Build a unified (done, next) interface over all three formats.
        std::shared_ptr<sift_utils::bvecs_chunks> bvecs_rd;
        std::shared_ptr<sift_utils::fvecs_chunks> fvecs_rd;
        std::shared_ptr<bin_utils::bin_chunks>    bin_rd;

        std::function<bool()>                              reader_done;
        std::function<std::vector<std::vector<float>>()>   reader_next;

        const std::string base_ext = file_ext(args.base_path);

        if (base_ext.empty() || base_ext == ".bin") {
            // Directory → SpaceV bin parts; single .bin → one contiguous blob
            std::cout << "Base format: SpaceV bin ("
                      << (base_ext.empty() ? "directory" : "single file") << ")" << std::endl;
            bin_rd      = std::make_shared<bin_utils::bin_chunks>(args.base_path, chunk_size);
            reader_done = [&]() { return bin_rd->done(); };
            reader_next = [&]() -> std::vector<std::vector<float>> { return bin_rd->next(); };
        } else if (base_ext == ".bvecs") {
            std::cout << "Base format: bvecs" << std::endl;
            bvecs_rd    = std::make_shared<sift_utils::bvecs_chunks>(args.base_path, chunk_size);
            reader_done = [&]() { return bvecs_rd->done(); };
            reader_next = [&]() -> std::vector<std::vector<float>> { return bvecs_rd->next(); };
        } else if (base_ext == ".fvecs") {
            std::cout << "Base format: fvecs" << std::endl;
            fvecs_rd    = std::make_shared<sift_utils::fvecs_chunks>(args.base_path, chunk_size);
            reader_done = [&]() { return fvecs_rd->done(); };
            reader_next = [&]() -> std::vector<std::vector<float>> { return fvecs_rd->next(); };
        } else {
            std::cerr << "Unknown base file extension '" << base_ext
                      << "'. Expected .fvecs, .bvecs, .bin, or a directory.\n";
            return 1;
        }

        std::cout << "Processing base vectors from: " << args.base_path;
        if (max_insert >= 0) std::cout << " (limited to " << max_insert << " vectors)";
        std::cout << std::endl;

        if (args.insert_batch_size == 1)
            std::cout << "Inserting vectors (single inserts, chunked reading)..." << std::endl;
        else
            std::cout << "Inserting vectors (batch size=" << args.insert_batch_size << ", chunked reading)..." << std::endl;

        const auto t0_insert = Clock::now();
        double total_client_insert_ms = 0.0;
        size_t global_idx = 0;

        while (!reader_done() && (max_insert < 0 || total_inserted < static_cast<size_t>(max_insert))) {
            auto chunk = reader_next();
            if (chunk.empty()) break;

            if (args.insert_batch_size == 1) {
                for (size_t i = 0; i < chunk.size() && (max_insert < 0 || total_inserted < static_cast<size_t>(max_insert)); ++i) {
                    const auto& v = chunk[i];
                    auto ci_t0 = Clock::now();
                    ann::InsertData req;
                    req.set_id(static_cast<int64_t>(global_idx));
                    ann::set_float_bytes(req.mutable_vector_data(), v.data(), static_cast<int>(v.size()));
                    ann::InsertAck ack;
                    grpc::ClientContext ctx;
                    auto s = stub->Insert(&ctx, req, &ack);
                    auto ci_t1 = Clock::now();
                    total_client_insert_ms += std::chrono::duration<double, std::milli>(ci_t1 - ci_t0).count();
                    if (s.ok() && ack.success()) ++inserted_ok;
                    total_inserted++;
                    global_idx++;
                }
            } else {
                for (size_t i = 0; i < chunk.size() && (max_insert < 0 || total_inserted < static_cast<size_t>(max_insert)); ) {
                    auto ci_t0 = Clock::now();
                    ann::BatchedInsertData batched_req;
                    size_t batch_start_idx = global_idx;
                    size_t batch_size = 0;

                    while (batch_size < static_cast<size_t>(args.insert_batch_size) &&
                           i < chunk.size() &&
                           (max_insert < 0 || total_inserted < static_cast<size_t>(max_insert))) {
                        const auto& v = chunk[i];
                        batched_req.add_ids(static_cast<int64_t>(global_idx));
                        ann::append_float_bytes(batched_req.mutable_vectors_data(), v.data(), static_cast<int>(v.size()));
                        batch_size++; i++; global_idx++;
                    }
                    if (batch_size == 0) break;
                    batched_req.set_batch_size(static_cast<int>(batch_size));

                    ann::BatchedInsertAck batched_ack;
                    grpc::ClientContext ctx;
                    auto s = stub->BatchedInsert(&ctx, batched_req, &batched_ack);
                    auto ci_t1 = Clock::now();
                    const double ms = std::chrono::duration<double, std::milli>(ci_t1 - ci_t0).count();

                    if (s.ok()) {
                        for (int bi = 0; bi < batched_ack.acks_size(); ++bi) {
                            if (batched_ack.acks(bi).success()) ++inserted_ok;
                            total_client_insert_ms += ms;
                            total_inserted++;
                        }
                    } else {
                        std::cout << "Batch starting at insert " << batch_start_idx
                                  << " failed: " << s.error_code() << " - " << s.error_message() << std::endl;
                        global_idx = batch_start_idx;
                    }
                }
            }
        }

        const auto t1_insert = Clock::now();
        insert_sec = std::chrono::duration<double>(t1_insert - t0_insert).count();
        insert_eps = inserted_ok / std::max(1e-9, insert_sec);
        std::cout << "Insertions: ok=" << inserted_ok << "/" << total_inserted
                  << ", time=" << insert_sec << " s, eps=" << insert_eps
                  << ", avg_client_insert_ms=" << (inserted_ok ? total_client_insert_ms / inserted_ok : 0.0)
                  << std::endl;
    }

    // ── Load query vectors ────────────────────────────────────────────────
    std::vector<std::vector<float>> query_vecs;
    {
        const std::string qext = file_ext(args.query_path);
        std::cout << "Loading query vectors from: " << args.query_path;
        if (max_query >= 0) std::cout << " (limited to " << max_query << " vectors)";
        std::cout << std::endl;

        auto collect_chunks = [&](auto& reader) {
            while (!reader.done() && (max_query < 0 || query_vecs.size() < static_cast<size_t>(max_query))) {
                auto chunk = reader.next();
                if (chunk.empty()) break;
                for (auto& vec : chunk) {
                    if (max_query >= 0 && query_vecs.size() >= static_cast<size_t>(max_query)) break;
                    query_vecs.push_back(std::move(vec));
                }
            }
        };

        if (qext == ".bvecs") {
            sift_utils::bvecs_chunks reader(args.query_path, 10000);
            collect_chunks(reader);
        } else if (qext == ".fvecs") {
            sift_utils::fvecs_chunks reader(args.query_path, 10000);
            collect_chunks(reader);
        } else if (qext == ".bin") {
            query_vecs = bin_utils::read_bin_vectors(args.query_path, max_query);
        } else {
            std::cerr << "Unknown query file extension '" << qext
                      << "'. Expected .fvecs, .bvecs, or .bin.\n";
            return 1;
        }
        std::cout << "Loaded " << query_vecs.size() << " query vectors." << std::endl;
    }

    // ── Load ground truth ─────────────────────────────────────────────────
    std::vector<std::vector<int>> groundtruth;
    const size_t num_queries = query_vecs.size();
    if (!args.gt_path.empty()) {
        const std::string gext = file_ext(args.gt_path);
        if (gext == ".ivecs") {
            groundtruth = sift_utils::read_ivecs(args.gt_path, static_cast<int>(num_queries));
            std::cout << "Loaded ground truth from: " << args.gt_path << std::endl;
        } else if (gext == ".bin") {
            groundtruth = bin_utils::read_bin_groundtruth(args.gt_path, static_cast<int>(num_queries));
            std::cout << "Loaded ground truth from: " << args.gt_path << std::endl;
        } else {
            std::cerr << "Unknown ground truth extension '" << gext
                      << "'. Expected .ivecs or .bin. Skipping recall computation.\n";
        }
    } else {
        std::cout << "No ground truth provided. Skipping recall computation." << std::endl;
    }

    // ── Search sweep ──────────────────────────────────────────────────────
    std::vector<int> ef_list = args.efsearch_values.empty()
        ? std::vector<int>{args.efsearch} : args.efsearch_values;
    std::vector<int> rk_list = args.router_k_values.empty()
        ? std::vector<int>{args.router_k} : args.router_k_values;
    bool is_sweep = !args.efsearch_values.empty() || !args.router_k_values.empty();

    if (num_queries == 0) {
        std::cout << "Searching skipped (0 queries).\nDone." << std::endl;
        return 0;
    }

    // Open CSV for sweep results (truncate at start so each run is clean)
    std::ofstream sweep_csv;
    if (is_sweep && !args.csv_path.empty()) {
        sweep_csv.open(args.csv_path, std::ios::out | std::ios::trunc);
        if (sweep_csv.is_open()) {
            sweep_csv << "efsearch,router_k,qps,recall_at_" << args.k << ",avg_client_ms\n";
            std::cout << "Sweep results will be written to: " << args.csv_path << std::endl;
        } else {
            std::cerr << "Warning: could not open CSV file: " << args.csv_path << std::endl;
        }
    }

    if (is_sweep) {
        std::cout << "\n=== Search Sweep (" << ef_list.size() * rk_list.size() << " combinations) ===" << std::endl;
        std::cout << std::left
                  << std::setw(12) << "efsearch"
                  << std::setw(12) << "router_k"
                  << std::setw(14) << "QPS"
                  << std::setw(16) << ("Recall@" + std::to_string(args.k))
                  << "avg_client_ms" << std::endl;
    }

    const int bsz = std::max(1, args.batch_size);

    for (int ef : ef_list) {
        for (int rk : rk_list) {
            size_t search_ok   = 0;
            double total_recall = 0.0;
            double client_total_ms = 0.0;
            const auto t0_search = Clock::now();

            if (args.batch_size == 1 && !is_sweep) {
                std::cout << "Searching " << num_queries << " queries with k=" << args.k
                          << " (single queries)..." << std::endl;
                for (size_t i = 0; i < num_queries; ++i) {
                    const auto& v = query_vecs[i];
                    auto q_t0 = Clock::now();
                    ann::Query req;
                    ann::set_float_bytes(req.mutable_vector_data(), v.data(), static_cast<int>(v.size()));
                    req.set_k(args.k);
                    req.set_query_id(static_cast<int>(i));
                    ann::AggregatedResult out;
                    grpc::ClientContext ctx;
                    auto s = stub->Search(&ctx, req, &out);
                    auto q_t1 = Clock::now();
                    const double q_ms = std::chrono::duration<double, std::milli>(q_t1 - q_t0).count();
                    std::ofstream cs_csv("/tmp/client_search.csv", std::ios::app);
                    if (cs_csv.tellp() == 0) cs_csv << "query_id,client_rpc_ms\n";
                    cs_csv << req.query_id() << ',' << q_ms << '\n';
                    if (s.ok()) {
                        ++search_ok;
                        client_total_ms += q_ms;
                        if (!groundtruth.empty()) {
                            const auto& gt = groundtruth[i];
                            const int gt_size  = std::min(args.k, static_cast<int>(gt.size()));
                            const int pred_size = out.topk_size();
                            if (pred_size != args.k)
                                std::cerr << "Warning: Predicted results size does not match k=" << args.k << std::endl;
                            int matches = 0;
                            for (int j = 0; j < gt_size; ++j) {
                                int gt_id = gt[j];
                                for (int kk = 0; kk < pred_size; ++kk)
                                    if (out.topk(kk).id() == gt_id) { matches++; break; }
                            }
                            total_recall += static_cast<double>(matches) / gt_size;
                        }
                    } else {
                        std::cout << "Query " << i << " failed: "
                                  << s.error_code() << " - " << s.error_message() << std::endl;
                    }
                }
            } else {
                if (!is_sweep)
                    std::cout << "Searching " << num_queries << " queries with k=" << args.k
                              << " (batch size=" << bsz << ")..." << std::endl;

                for (size_t i = 0; i < num_queries; i += bsz) {
                    auto q_t0 = Clock::now();
                    ann::BatchedQuery batched_req;
                    if (ef > 0) batched_req.set_efsearch(ef);
                    if (rk > 0) batched_req.set_router_k(rk);
                    const size_t batch_end = std::min(i + static_cast<size_t>(bsz), num_queries);
                    if (!is_sweep) {
                        std::cout << "[bench_client] Sending batch starting at query " << i
                                  << " size=" << (batch_end - i)
                                  << " router_k=" << (rk > 0 ? std::to_string(rk) : std::string("default"))
                                  << " efsearch=" << (ef > 0 ? std::to_string(ef) : std::string("default"))
                                  << std::endl;
                    }
                    batched_req.set_k(args.k);
                    batched_req.set_batch_size(static_cast<int>(batch_end - i));
                    for (size_t j = i; j < batch_end; ++j) {
                        const auto& v = query_vecs[j];
                        ann::append_float_bytes(batched_req.mutable_vectors_data(), v.data(), static_cast<int>(v.size()));
                    }
                    ann::BatchedAggregatedResult batched_out;
                    grpc::ClientContext ctx;
                    auto s = stub->BatchedSearch(&ctx, batched_req, &batched_out);
                    auto q_t1 = Clock::now();
                    const double q_ms = std::chrono::duration<double, std::milli>(q_t1 - q_t0).count();

                    if (s.ok()) {
                        for (int bi = 0; bi < batched_out.results_size(); ++bi) {
                            const auto& out = batched_out.results(bi);
                            size_t query_idx = i + bi;
                            ++search_ok;
                            client_total_ms += q_ms;
                            if (!groundtruth.empty()) {
                                const auto& gt = groundtruth[query_idx];
                                const int gt_size   = std::min(args.k, static_cast<int>(gt.size()));
                                const int pred_size = out.topk_size();
                                if (pred_size != args.k)
                                    std::cerr << "Warning: Predicted results size does not match k=" << args.k << std::endl;
                                int matches = 0;
                                for (int j = 0; j < gt_size; ++j) {
                                    int gt_id = gt[j];
                                    for (int kk = 0; kk < pred_size; ++kk)
                                        if (out.topk(kk).id() == gt_id) { matches++; break; }
                                }
                                total_recall += static_cast<double>(matches) / gt_size;
                            }
                        }
                        if (!is_sweep) {
                            std::ofstream cs_csv("/tmp/client_search.csv", std::ios::app);
                            if (cs_csv.tellp() == 0) cs_csv << "batch_id,client_rpc_ms,batch_size\n";
                            cs_csv << (i / bsz) << ',' << q_ms << ',' << (batch_end - i) << '\n';
                        }
                    } else {
                        std::cout << "Batch starting at query " << i << " failed: "
                                  << s.error_code() << " - " << s.error_message() << std::endl;
                    }
                }
            }

            const auto t1_search = Clock::now();
            const double search_sec  = std::chrono::duration<double>(t1_search - t0_search).count();
            const double qps         = search_ok / std::max(1e-9, search_sec);
            const double avg_recall  = search_ok > 0 ? total_recall   / search_ok : 0.0;
            const double avg_ms      = search_ok > 0 ? client_total_ms / search_ok : 0.0;

            if (is_sweep) {
                std::cout << std::left
                          << std::setw(12) << (ef > 0 ? std::to_string(ef) : "default")
                          << std::setw(12) << (rk > 0 ? std::to_string(rk) : "default")
                          << std::setw(14) << std::fixed << std::setprecision(1) << qps
                          << std::setw(16) << std::fixed << std::setprecision(4) << avg_recall
                          << std::fixed << std::setprecision(2) << avg_ms << std::endl;
                if (sweep_csv.is_open()) {
                    sweep_csv << (ef > 0 ? std::to_string(ef) : "default") << ','
                              << (rk > 0 ? std::to_string(rk) : "default") << ','
                              << std::fixed << std::setprecision(1) << qps << ','
                              << std::fixed << std::setprecision(4) << avg_recall << ','
                              << std::fixed << std::setprecision(2) << avg_ms << '\n';
                    sweep_csv.flush();
                }
            } else {
                std::cout << "Search: ok=" << search_ok << "/" << num_queries
                          << ", time=" << search_sec << " s, QPS=" << qps
                          << ", Recall@" << args.k << "=" << avg_recall
                          << ", avg_client_rpc_ms=" << avg_ms << std::endl;
            }
        }
    }

    std::cout << "Done." << std::endl;
    return 0;
}
