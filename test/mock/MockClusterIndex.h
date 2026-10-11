#pragma once

#include "worker/i_cluster_index.h"
#include "include/hnswlib.h"

namespace ann {

class MockClusterIndex : public IClusterIndex<float> {
public:
    explicit MockClusterIndex(int dim, int max_elements, int M, int ef_construction);
    ~MockClusterIndex() override = default;

    bool insertVector(int id, const float* vector) override;
    std::vector<std::pair<float, int>> searchKNN(const float* query_data, int k) override;
    bool saveIndex(const std::string& filename) override;
    bool loadIndex(const std::string& filename) override;
    hnswlib::HierarchicalNSW<float>* getHNSWIndex() override;

private:
    int dim_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index_;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
};

} // namespace ann
