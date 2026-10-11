#pragma once

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>

namespace ann {

struct ClusterInfo {
    int cluster_id;
    std::vector<float> centroid;
    std::string worker_address;
};

class IQueryRouter {
public:
    virtual ~IQueryRouter() = default;

    virtual void addCluster(int cluster_id, const std::vector<float>& centroid, const std::string& worker_address) = 0;
    virtual ClusterInfo insertQuery(const float* query_data, int dim) = 0;
    virtual std::vector<ClusterInfo> routeQuery(const float* query_data, int dim, int k) = 0;
    virtual ClusterInfo getClusterInfo(int cluster_id) const = 0;
    virtual std::vector<ClusterInfo> getAllClusters() const = 0;
    virtual void addVectorsToRoutingIndex(const float* data, int n, int dim) = 0;
    virtual void rebuildRoutingIndex(const std::unordered_map<int, std::vector<std::vector<float>>>& cluster_samples) = 0;
    virtual int getMaxNodes() const = 0;
    
    // COMMENTED OUT: split history tree replaced by balanced k-means rebalancing
    // virtual void updateSplitHistory(int old_cluster_id, const std::vector<float>& old_centroid,
    //                                 int new_cluster_id, const std::vector<float>& new_centroid) = 0;
};

} // namespace ann


