#include "MockQueryRouter.h"
#include <stdexcept>
#include <limits>

namespace ann {

MockQueryRouter::MockQueryRouter(int dim, const std::vector<std::string>& worker_addresses)
    : dim_(dim) {
    
    // initialize the clusters
    clusters_.reserve(worker_addresses.size());
    int value = 1;
    for (int i = 0; i < worker_addresses.size(); ++i) {
        std::vector<float> centroid(dim_, static_cast<float>(value));
        clusters_.push_back({i + 1, centroid, worker_addresses[i]});
        value += 100;
    }
}

void MockQueryRouter::addCluster(int cluster_id, const std::vector<float>& centroid, const std::string& worker_address) {
    if (static_cast<int>(centroid.size()) != dim_) {
        throw std::invalid_argument("Centroid dimension mismatch");
    }
    if (static_cast<size_t>(cluster_id) >= clusters_.size()) {
        clusters_.resize(static_cast<size_t>(cluster_id) + 1);
    }
    clusters_[static_cast<size_t>(cluster_id)] = {cluster_id, centroid, worker_address};
}

ClusterInfo MockQueryRouter::insertQuery(const float* query_data, int dim) {
    if (dim != dim_) {
        throw std::invalid_argument("Query vector dimension mismatch");
    }
    // return the closest cluster
    int closest_cluster = 0;
    float closest_distance = std::numeric_limits<float>::max();
    for (int i = 0; i < clusters_.size(); ++i) {
        float distance = 0;
        for (int j = 0; j < dim_; ++j) {
            distance += (query_data[j] - clusters_[i].centroid[j]) * (query_data[j] - clusters_[i].centroid[j]);
        }
        if (distance < closest_distance) {
            closest_distance = distance;
            closest_cluster = i;
        }
    }
    return clusters_[closest_cluster];
}


std::vector<ClusterInfo> MockQueryRouter::routeQuery(const float* /*query_data*/, int dim, int num_clusters) {
    if (dim != dim_) {
        throw std::invalid_argument("Query vector dimension mismatch");
    }
    std::vector<ClusterInfo> workers;
    for (int i = 0; i < num_clusters && i < clusters_.size(); ++i) {
        workers.push_back(clusters_[i]);
    }
    return workers;
}

ClusterInfo MockQueryRouter::getClusterInfo(int cluster_id) const {
    if (cluster_id < 0 || static_cast<size_t>(cluster_id) >= clusters_.size()) {
        throw std::out_of_range("Cluster not found");
    }
    return clusters_[static_cast<size_t>(cluster_id)];
}

std::vector<ClusterInfo> MockQueryRouter::getAllClusters() const {
    std::vector<ClusterInfo> result;
    for (const auto& info : clusters_) {
        if (!info.worker_address.empty()) result.push_back(info);
    }
    return result;
}

void MockQueryRouter::addVectorsToRoutingIndex(const float* /*data*/, int /*n*/, int /*dim*/) {
    // No-op for mock router
}

void MockQueryRouter::rebuildRoutingIndex(const std::unordered_map<int, std::vector<std::vector<float>>>& /*cluster_samples*/) {
    // No-op for mock router
}

int MockQueryRouter::getMaxNodes() const {
    return 0; // Mock router doesn't use max_nodes
}

// COMMENTED OUT: split history tree replaced by balanced k-means rebalancing
// void MockQueryRouter::updateSplitHistory(...) { }

} // namespace ann


