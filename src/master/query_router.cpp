#include "query_router.h"

namespace ann {

QueryRouter::QueryRouter(int dim, int n, int M, int efConstruction, const std::string& filepath, const std::vector<std::string> worker_addresses): dim_(dim), M_(M), efConstruction_(efConstruction) {
    space_ = std::make_unique<hnswlib::L2Space>(dim_);
    max_nodes = n;
    //hparams are n, M, efConstruction, n is the maximum number of points, will need to resize later
    routing_index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), max_nodes, M, efConstruction);
    hnswlib::labeltype curr_label = 0;
    
    //if there is a filepath initialize the routing index otherwise return a 0-node routing index
    if (!filepath.empty() && std::filesystem::exists(filepath)) {
        //read from the file path, check extension (bvecs/fvecs/ivecs) to figure out vector sizes
        std::filesystem::path p(filepath);
        std::string ext = p.extension().string();

        if (ext == ".fvecs") {
            // Load at most max_nodes vectors
            vecs = sift_utils::read_fvecs<float>(filepath, max_nodes);
            for (const auto& v : vecs) {
                if (static_cast<int>(v.size()) != dim_) {
                    throw std::runtime_error("Dim mismatch in .fvecs");
                }
                routing_index_->addPoint(v.data(), curr_label++);
            }
        } else if (ext == ".ivecs") {
            // Load at most max_nodes vectors
            auto ivecs = sift_utils::read_ivecs(filepath, max_nodes); 

            vecs.reserve(vecs.size() + ivecs.size());
            std::vector<float> tmp;
            tmp.reserve(dim_);

            for (const auto& v : ivecs) {
                if (static_cast<int>(v.size()) != dim_) {
                    std::cout << "[QueryRouter] Dim mismatch in .ivecs, expected dim=" << dim_ << ", got dim=" << v.size() << std::endl << std::flush;
                    throw std::runtime_error("Dim mismatch in .ivecs");
                }

                for (int b : v) {tmp.push_back(static_cast<float>(b));}
                vecs.push_back(std::move(tmp));
                routing_index_->addPoint(vecs.back().data(), curr_label++);
                tmp.clear();
            }
        } 

        // if vecs isn't empty we can run kmeans
        if (!vecs.empty()) {
            int k = worker_addresses.size();
            if (k > 0) { //continue only if we have worker addresses
                kmeans::KMeans<float> km(k);
                km.fit(vecs);
                auto centroids = km.get_centroids(); //we will get k centroids

                int starting_id = 0;
                for (auto& centroid: centroids) { //add ClusterInfo
                    clusters_.emplace(starting_id, ClusterInfo{starting_id, centroid, worker_addresses[starting_id]});
                    starting_id++;
                }
                //the last value of starting id will be (k-1)
            }
        }
    } else {
        // Initialize with one default cluster (incremental approach)
        if (!worker_addresses.empty()) {
            clusters_.emplace(0, ClusterInfo{0, std::vector<float>(dim_, 0.0), worker_addresses[0]});
        }
    }

    // COMMENTED OUT: split history tree replaced by balanced k-means rebalancing
    // std::vector<float> dummy_centroid(dim_, 0.0f);
    // split_history_root_ = std::make_unique<SplitHistoryNode>(dummy_centroid, 0);
}

QueryRouter::~QueryRouter() = default;



void QueryRouter::addCluster(int cluster_id, const std::vector<float>& centroid, const std::string& worker_address) {
    if (centroid.size() != static_cast<size_t>(dim_)) {
        throw std::invalid_argument("addCluster: centroid size does not match index dimensionality");
    }
    
    // Update or insert cluster
    clusters_[cluster_id] = ClusterInfo{cluster_id, centroid, worker_address};
}

ClusterInfo QueryRouter::insertQuery(const float* query_data, int dim) {
    if (query_data == nullptr) {
        throw std::invalid_argument("insertQuery: query_data is null");
    }
    if (dim != dim_) {
        throw std::invalid_argument("insertQuery: dim mismatch");
    }
    if (clusters_.empty()) {
        throw std::out_of_range("insertQuery: no clusters configured");
    }

    if (clusters_.size() == 1) {
        return clusters_.begin()->second;
    }

    // Route to closest centroid
    int best_cluster_id = -1;
    float best_dist = std::numeric_limits<float>::infinity();

    const auto dist_fn = space_->get_dist_func();
    const auto dist_param = space_->get_dist_func_param();

    for (const auto& kv : clusters_) {
        const int cid = kv.first;
        const ClusterInfo& ci = kv.second;

        float d = dist_fn(static_cast<const void*>(query_data),
                          static_cast<const void*>(ci.centroid.data()),
                          dist_param);
        if (d < best_dist) {
            best_dist = d;
            best_cluster_id = cid;
        }
    }
    return clusters_.at(best_cluster_id);
}


 std::vector<ClusterInfo> QueryRouter::routeQuery(const float* query_data, int dim, int k) {

    if (routing_index_->cur_element_count == 0) {
        // If routing index is empty, return all clusters
        std::vector<ClusterInfo> out;
        out.reserve(clusters_.size());
        for (const auto& kv : clusters_) {
            out.push_back(kv.second);
        }
        return out;
    }

    // Search HNSW index to find top k neighbors
    const int search_k = k;
    auto pq = routing_index_->searchKnn(query_data, static_cast<size_t>(search_k));
    
    // Collect all distinct partitions that have at least one nearest neighbor
    std::unordered_set<int> cluster_ids_seen;
    while (!pq.empty()) {
        hnswlib::labeltype label = pq.top().second;  // (distance, label)
        pq.pop();
        auto it = label_to_cluster_.find(label);
        if (it != label_to_cluster_.end()) {
            cluster_ids_seen.insert(it->second);
        }
    }

    std::vector<ClusterInfo> out;
    out.reserve(cluster_ids_seen.size());
    for (int cid : cluster_ids_seen) {
        auto it = clusters_.find(cid);
        if (it != clusters_.end()) {
            out.push_back(it->second);
        }
    }
    
    // Fallback: if no partitions found, return all clusters
    if (out.empty()) {
        for (const auto& kv : clusters_) {
            out.push_back(kv.second);
        }
    }

    return out;
}

ClusterInfo QueryRouter::getClusterInfo(int cluster_id) const {
   auto it = clusters_.find(cluster_id);
    if (it == clusters_.end()) {
        throw std::out_of_range("getClusterInfo: cluster_id not found");
    }
    return it->second;
}

std::vector<ClusterInfo> QueryRouter::getAllClusters() const {
    std::vector<ClusterInfo> all;
    all.reserve(clusters_.size());
    for (const auto& element : clusters_) {
        all.push_back(element.second);
    }
    return all;
}

void QueryRouter::addVectorsToRoutingIndex(const float* data, int n, int dim) {
    for (int i = 0; i < n; ++i) {
        routing_index_->addPoint(data + i * dim, static_cast<hnswlib::labeltype>(i));
    }
    routing_index_->setEf(100);
}

int QueryRouter::getMaxNodes() const {
    return max_nodes;
}

void QueryRouter::rebuildRoutingIndex(const std::unordered_map<int, std::vector<std::vector<float>>>& cluster_samples) {
    // Clear existing index and mappings
    vecs.clear();
    label_to_cluster_.clear();
    
    // Calculate total number of points
    size_t total_points = 0;
    for (const auto& kv : cluster_samples) {
        total_points += kv.second.size();
    }
    
    // Update max_nodes if needed (resize if necessary)
    if (total_points > static_cast<size_t>(max_nodes)) {
        max_nodes = static_cast<int>(total_points * 1.2); // Add 20% buffer
    }
    
    // Create a fresh HNSW index
    routing_index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), max_nodes, M_, efConstruction_);
    routing_index_->setEf(100);  
    
    // Flatten samples to parallelize HNSW insertions
    vecs.reserve(total_points);
    std::vector<int> flat_cluster_ids;
    flat_cluster_ids.reserve(total_points);

    for (const auto& kv : cluster_samples) {
        const int cluster_id = kv.first;
        const auto& points = kv.second;
        for (const auto& point : points) {
            if (point.size() != dim_) {
                throw std::invalid_argument("rebuildRoutingIndex: point dimension mismatch");
            }
            vecs.push_back(point);
            flat_cluster_ids.push_back(cluster_id);
        }
    }

    // Build point label -> cluster mapping.
    for (hnswlib::labeltype lbl = 0; lbl < static_cast<hnswlib::labeltype>(flat_cluster_ids.size()); ++lbl) {
        label_to_cluster_[lbl] = flat_cluster_ids[static_cast<size_t>(lbl)];
    }

    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < vecs.size(); ++i) {
        routing_index_->addPoint(vecs[i].data(), static_cast<hnswlib::labeltype>(i));
    }
    
    std::cout << "[QueryRouter] Rebuilt routing index with " << total_points 
              << " points from " << cluster_samples.size() << " clusters" << std::endl;
}

// COMMENTED OUT: split history tree replaced by balanced k-means rebalancing
// SplitHistoryNode* QueryRouter::findLeafByClusterId(...) { ... }
// void QueryRouter::updateSplitHistory(...) { ... }
// int QueryRouter::traverseSplitHistory(...) { ... }

} // namespace ann
