#include <gtest/gtest.h>
#include "../src/worker/cluster_index.h"

namespace ann
{

    class ClusterIndexTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            dim = 4;
            max_elements = 1000000;
            M = 5;
            ef_construction = 10;
            cluster_index = std::make_unique<ClusterIndex<float>>(dim, max_elements, M, ef_construction);

            vec1 = {1.0f, 2.0f, 3.0f, 4.0f};
            vec2 = {2.0f, 3.0f, 4.0f, 5.0f};
            vec3 = {5.0f, 6.0f, 7.0f, 8.0f};
        }

        int dim;
        int max_elements;
        int M;
        int ef_construction;
        std::unique_ptr<ClusterIndex<float>> cluster_index;
        std::vector<float> vec1, vec2, vec3;
    };

    // Basic insertion
    TEST_F(ClusterIndexTest, InsertVectorSuccess)
    {
        EXPECT_TRUE(cluster_index->insertVector(1, vec1.data()));
    }

    // KNN returns correct number of results
    TEST_F(ClusterIndexTest, SearchKNNReturnsCorrectSize)
    {
        cluster_index->insertVector(1, vec1.data());
        cluster_index->insertVector(2, vec2.data());

        auto results = cluster_index->searchKNN(vec1.data(), 2);
        EXPECT_EQ(results.size(), 2);
    }

    // KNN returns nearest neighbor first
    TEST_F(ClusterIndexTest, SearchKNNReturnsNearestFirst)
    {
        cluster_index->insertVector(1, vec1.data());
        cluster_index->insertVector(2, vec2.data());

        auto results = cluster_index->searchKNN(vec1.data(), 2);
        EXPECT_EQ(results.front().second, 1);
        EXPECT_LE(results[0].first, results[1].first);
    }

    // Search with k > number of elements
    TEST_F(ClusterIndexTest, SearchKNNWithKGreaterThanSize)
    {
        cluster_index->insertVector(1, vec1.data());

        auto results = cluster_index->searchKNN(vec1.data(), 5);
        EXPECT_EQ(results.size(), 1);
    }

    // Search before any insertion
    TEST_F(ClusterIndexTest, SearchBeforeInsertionReturnsEmpty)
    {
        auto results = cluster_index->searchKNN(vec1.data(), 1);
        EXPECT_TRUE(results.empty());
    }

    // Save and load index round-trip
    TEST_F(ClusterIndexTest, SaveAndLoadIndex)
    {
        cluster_index->insertVector(1, vec1.data());
        cluster_index->insertVector(2, vec2.data());

        std::string filename = "test_index.bin";
        EXPECT_TRUE(cluster_index->saveIndex(filename));

        auto new_index = std::make_unique<ClusterIndex<float>>(dim, max_elements, M, ef_construction);
        EXPECT_TRUE(new_index->loadIndex(filename));

        auto results = new_index->searchKNN(vec1.data(), 2);
        EXPECT_EQ(results.size(), 2);
        EXPECT_EQ(results.front().second, 1);
    }

    // Load non-existent file should fail gracefully
    TEST_F(ClusterIndexTest, LoadNonExistentFileFails)
    {
        EXPECT_FALSE(cluster_index->loadIndex("nonexistent_index.bin"));
    }

    // KNN with k = 0 returns empty
    TEST_F(ClusterIndexTest, SearchKNNWithZeroKReturnsEmpty)
    {
        cluster_index->insertVector(1, vec1.data());
        auto results = cluster_index->searchKNN(vec1.data(), 0);
        EXPECT_TRUE(results.empty());
    }

    // Stress test: insert many vectors and search
    TEST_F(ClusterIndexTest, LargeNumberOfVectors)
    {
        for (int i = 0; i < 100000; ++i)
        {
            std::vector<float> v = {float(i), float(i + 1), float(i + 2), float(i + 3)};
            EXPECT_TRUE(cluster_index->insertVector(i, v.data()));
        }

        auto results = cluster_index->searchKNN(vec1.data(), 5);
        EXPECT_EQ(results.size(), 5);
    }

} // namespace ann
