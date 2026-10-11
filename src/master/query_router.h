#pragma once

#include "i_query_router.h"
#include "include/hnswlib.h"
#include "include/kmeans.h"
#include "include/space_l2.h"
#include "include/sift_utils.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include <filesystem>


namespace ann {

// COMMENTED OUT: split history tree replaced by balanced k-means rebalancing
// struct SplitHistoryNode {
//     std::vector<float> centroid;
//     int cluster_id;
//     std::unique_ptr<SplitHistoryNode> left;
//     std::unique_ptr<SplitHistoryNode> right;
//     SplitHistoryNode(const std::vector<float>& c, int cid = -1)
//         : centroid(c), cluster_id(cid), left(nullptr), right(nullptr) {}
//     bool isLeaf() const { return !left && !right; }
// };

class QueryRouter : public IQueryRouter {
public:
//intialize space and index, run kmeans
//if we pass in a list of worker address (one cluster per worker address)
    explicit QueryRouter(int dim, int n = 10000, int M = 16, int efConstruction = 200, const std::string& filepath = "", const std::vector<std::string> worker_addresses = {});
    ~QueryRouter();

    // Add a cluster centroid to the routing index
    void addCluster(int cluster_id, const std::vector<float>& centroid, const std::string& worker_address) override;
    
    ClusterInfo insertQuery(const float* query_data, int dim) override;

    // Route a query to relevant clusters based on nearest centroids
     std::vector<ClusterInfo> routeQuery(const float* query_data, int dim, int k) override;
    
    // Get cluster info by ID
    ClusterInfo getClusterInfo(int cluster_id) const override;
    
    // Get all clusters
    std::vector<ClusterInfo> getAllClusters() const override;

    void addVectorsToRoutingIndex(const float* data, int n, int dim) override;
    int getMaxNodes() const override;

    // Rebuild the routing index from scratch with sampled points from all clusters
    // This clears the existing index and creates a fresh one
    void rebuildRoutingIndex(const std::unordered_map<int, std::vector<std::vector<float>>>& cluster_samples);

    // COMMENTED OUT: split history tree replaced by balanced k-means rebalancing
    // void updateSplitHistory(int old_cluster_id, const std::vector<float>& old_centroid,
    //                        int new_cluster_id, const std::vector<float>& new_centroid) override;
    // int traverseSplitHistory(const float* query_data) const;

private:
    int max_nodes; //the maximum number of nodes our routing index can handle
                  // we can resize the index later
    int dim_;
    int M_; // HNSW parameter M
    int efConstruction_; // HNSW parameter efConstruction
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_; 
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> routing_index_;

    //cluster_id -> ClusterInfo mapping (worker address, centroid value, cluster id)
    std::unordered_map<int, ClusterInfo> clusters_;

    //the 2-D array of floats which were used to build the small HNSW
    std::vector<std::vector<float>> vecs;

    // Mapping from label (point index in routing HNSW) to cluster_id (partition)
    std::unordered_map<hnswlib::labeltype, int> label_to_cluster_;

    // COMMENTED OUT: split history tree replaced by balanced k-means rebalancing
    // std::unique_ptr<SplitHistoryNode> split_history_root_;
    // SplitHistoryNode* findLeafByClusterId(SplitHistoryNode* node, int cluster_id) const;

};

} // namespace ann
