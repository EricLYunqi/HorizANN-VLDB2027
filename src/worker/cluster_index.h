#pragma once

#include <vector>
#include <memory>
#include "include/hnswlib.h"
#include "include/space_traits.h"
#include "i_cluster_index.h"

namespace ann
{

    template<typename dist_t>
    class ClusterIndex : public IClusterIndex<dist_t>
    {
    public:
        /**
         * @brief Construct a new ClusterIndex containing a HNSW L2 index.
         *
         * @param dim Vector space dimension.
         * @param max_elements Maximum number of elements in index.
         * @param M Tightly connected with internal dimensionality of the data, strongly affects memory consumption.
         * @param ef_construction Controls index search/build speed tradeoff.
         */
        ClusterIndex(int dim, int max_elements = 1000000, int M = 16, int ef_construction = 200, int ef_search = 200);
        ~ClusterIndex();

        /**
         * @brief Insert a single vector into the index
         *
         * @param id
         * @param vector
         * @return true
         * @return false
         */
        bool insertVector(int id, const float *vector) override;

        /**
         * @brief Search for k nearest neighbors
         *
         * @param query_vector
         * @param k
         * @return std::vector<std::pair<int, float>>
         */
        std::vector<std::pair<float, int>> searchKNN(const float *query_data, int k) override;

        /**
         * @brief Get internal HNSW index object
         *
         * @return hnswlib::HierarchicalNSW<dist_t>*
         */
        hnswlib::HierarchicalNSW<dist_t>* getHNSWIndex() override { return hnsw_index_.get(); }

        /**
         * @brief Serialize index to disk
         *
         * @param filename Path to index .bin file
         * @return true
         * @return false
         */
        bool saveIndex(const std::string &filename) override;

        /**
         * @brief Deserialize and load index from disk
         *
         * @param filename Path to index .bin file
         * @return true
         * @return false
         */
        bool loadIndex(const std::string &filename) override;

        void replaceIndex(std::unique_ptr<hnswlib::SpaceInterface<dist_t>> new_space,
                          std::unique_ptr<hnswlib::HierarchicalNSW<dist_t>> new_index) override {
            space_ = std::move(new_space);
            hnsw_index_ = std::move(new_index);
        }

    private:
        int dim_;
        std::unique_ptr<hnswlib::HierarchicalNSW<dist_t>> hnsw_index_;
        std::unique_ptr<hnswlib::SpaceInterface<dist_t>> space_;
    };

} // namespace ann
