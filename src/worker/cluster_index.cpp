#include "cluster_index.h"
#include "include/hnswlib.h"
#include "include/space_traits.h"
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <cmath>
#include <algorithm>

namespace ann
{

    template<typename dist_t>
    ClusterIndex<dist_t>::ClusterIndex(int dim, int max_elements, int M, int ef_construction, int ef_search)
        : dim_(dim)
    {
        space_ = std::make_unique<typename hnswlib::SpaceFor<dist_t>::type>(dim);
        hnsw_index_ = std::make_unique<hnswlib::HierarchicalNSW<dist_t>>(space_.get(), max_elements, M, ef_construction);
        hnsw_index_->setEf(ef_search);
    }

    template<typename dist_t>
    ClusterIndex<dist_t>::~ClusterIndex() = default;

    template<typename dist_t>
    bool ClusterIndex<dist_t>::insertVector(int id, const float* vector)
    {
        try
        {
            if constexpr (std::is_same_v<dist_t, int>) {
                // Convert float input to uint8 for storage
                std::vector<unsigned char> buf(dim_);
                for (int d = 0; d < dim_; ++d) {
                    buf[d] = static_cast<unsigned char>(
                        std::max(0L, std::min(255L, std::lround(static_cast<double>(vector[d])))));
                }
                hnsw_index_->addPoint(buf.data(), id);
            } else {
                hnsw_index_->addPoint(vector, id);
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            return false;
        }
        return true;
    }

    template<typename dist_t>
    std::vector<std::pair<float, int>> ClusterIndex<dist_t>::searchKNN(const float* query_data, int k)
    {
        std::priority_queue<std::pair<dist_t, size_t>> knn;

        if constexpr (std::is_same_v<dist_t, int>) {
            // Convert float query to uint8
            std::vector<unsigned char> buf(dim_);
            for (int d = 0; d < dim_; ++d) {
                buf[d] = static_cast<unsigned char>(
                    std::max(0L, std::min(255L, std::lround(static_cast<double>(query_data[d])))));
            }
            knn = hnsw_index_->searchKnn(buf.data(), k);
        } else {
            knn = hnsw_index_->searchKnn(query_data, k);
        }

        std::vector<std::pair<float, int>> result;
        result.reserve(knn.size());
        while (!knn.empty())
        {
            auto [dist, id] = knn.top();
            knn.pop();
            result.emplace_back(static_cast<float>(dist), static_cast<int>(id));
        }

        std::reverse(result.begin(), result.end());
        return result;
    }

    template<typename dist_t>
    bool ClusterIndex<dist_t>::saveIndex(const std::string &filename)
    {
        try
        {
            hnsw_index_->saveIndex(filename);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            return false;
        }
        return true;
    }

    template<typename dist_t>
    bool ClusterIndex<dist_t>::loadIndex(const std::string &filename)
    {
        try
        {
            hnsw_index_.reset();
            hnsw_index_ = std::make_unique<hnswlib::HierarchicalNSW<dist_t>>(space_.get(), filename);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ClusterIndex] Exception in loadIndex: " << e.what() << std::endl;
            std::cout << "[ClusterIndex] Exception in loadIndex: " << e.what() << std::endl << std::flush;
            return false;
        }

        return true;
    }

    // Explicit instantiations
    template class ClusterIndex<float>;
    template class ClusterIndex<int>;

} // namespace ann
