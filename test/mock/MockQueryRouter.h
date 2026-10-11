#pragma once

#include <vector>
#include <string>
#include "master/i_query_router.h"

namespace ann {

class MockQueryRouter : public IQueryRouter {
public:
    MockQueryRouter(int dim, const std::vector<std::string>& worker_addresses);
    ~MockQueryRouter() override = default;

    void addCluster(int cluster_id, const std::vector<float>& centroid, const std::string& worker_address) override;
    std::vector<ClusterInfo> routeQuery(const float* query_data, int dim, int k) override;
    ClusterInfo insertQuery(const float* query_data, int dim) override;
    ClusterInfo getClusterInfo(int cluster_id) const override;
    std::vector<ClusterInfo> getAllClusters() const override;
    void addVectorsToRoutingIndex(const float* data, int n, int dim) override;
    void rebuildRoutingIndex(const std::unordered_map<int, std::vector<std::vector<float>>>& cluster_samples) override;
    int getMaxNodes() const override;
    // COMMENTED OUT: split history tree replaced by balanced k-means rebalancing
    // void updateSplitHistory(int old_cluster_id, const std::vector<float>& old_centroid,
    //                        int new_cluster_id, const std::vector<float>& new_centroid) override;

private:
    int dim_;
    std::vector<ClusterInfo> clusters_;
};

} // namespace ann


