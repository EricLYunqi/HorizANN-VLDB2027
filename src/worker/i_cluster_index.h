#pragma once

#include <vector>
#include <utility>
#include <string>
#include "include/hnswlib.h"

namespace ann {

template<typename dist_t>
class IClusterIndex {
public:
    virtual ~IClusterIndex() = default;

    virtual bool insertVector(int id, const float* vector) = 0;
    virtual std::vector<std::pair<float, int>> searchKNN(const float* query_data, int k) = 0;
    virtual bool saveIndex(const std::string& filename) = 0;
    virtual bool loadIndex(const std::string& filename) = 0;

    virtual hnswlib::HierarchicalNSW<dist_t>* getHNSWIndex() = 0;

    virtual void replaceIndex(std::unique_ptr<hnswlib::SpaceInterface<dist_t>> new_space,
                              std::unique_ptr<hnswlib::HierarchicalNSW<dist_t>> new_index) = 0;
};

} // namespace ann
