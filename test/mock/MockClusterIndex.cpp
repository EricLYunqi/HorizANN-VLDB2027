#include "MockClusterIndex.h"
#include "include/hnswlib.h"
#include <fstream>
#include <stdexcept>

namespace ann {

MockClusterIndex::MockClusterIndex(int dim, int max_elements, int M, int ef_construction)
    : dim_(dim)
{
    space_ = std::make_unique<hnswlib::L2Space>(dim);
    hnsw_index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), max_elements, M, ef_construction);
}

bool MockClusterIndex::insertVector(int id, const float* vector) {
    hnsw_index_->addPoint(vector, id);
    return true;
}

std::vector<std::pair<float, int>> MockClusterIndex::searchKNN(const float* /*query_vector*/, int k) {
    std::vector<std::pair<float, int>> out;
    for (int i = 0; i < k; ++i) {
        out.emplace_back(static_cast<float>(i) * 0.1f, i);
    }
    return out;
}

bool MockClusterIndex::saveIndex(const std::string& /*filename*/) { return true; }
bool MockClusterIndex::loadIndex(const std::string& /*filename*/) { return true; }

hnswlib::HierarchicalNSW<float>* MockClusterIndex::getHNSWIndex() {
    return hnsw_index_.get();
}

} // namespace ann
