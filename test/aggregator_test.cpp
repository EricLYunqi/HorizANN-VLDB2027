#include <gtest/gtest.h>
#include "../src/master/aggregator.h"
#include "proto/ann.pb.h"

namespace ann {
namespace test {

class AggregatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        aggregator_ = std::make_unique<Aggregator>();
    }
    
    void TearDown() override {
        aggregator_.reset();
    }
    
    // Helper to convert owned results to pointer vector for aggregateResults API
    std::vector<const WorkerResult*> toPtrs(const std::vector<WorkerResult>& v) {
        std::vector<const WorkerResult*> ptrs;
        ptrs.reserve(v.size());
        for (const auto& r : v) ptrs.push_back(&r);
        return ptrs;
    }

    // Helper function to create a WorkerResult
    WorkerResult CreateWorkerResult(int worker_id, int query_id, 
                                   const std::vector<std::pair<int32_t, float>>& neighbors) {
        WorkerResult result;
        result.set_worker_id(worker_id);
        result.set_query_id(query_id);
        
        for (const auto& neighbor_pair : neighbors) {
            Neighbor* neighbor = result.add_results();
            neighbor->set_id(neighbor_pair.first);
            neighbor->set_distance(neighbor_pair.second);
        }
        
        return result;
    }
    
    // Helper function to extract results as vector of pairs for easy comparison
    std::vector<std::pair<int32_t, float>> ExtractResults(const AggregatedResult& result) {
        std::vector<std::pair<int32_t, float>> extracted;
        for (const auto& neighbor : result.topk()) {
            extracted.emplace_back(neighbor.id(), neighbor.distance());
        }
        return extracted;
    }
    
    std::unique_ptr<Aggregator> aggregator_;
};

// Test basic functionality with non-overlapping results
TEST_F(AggregatorTest, BasicAggregationNoOverlap) {
    std::vector<WorkerResult> worker_results;
    
    // Worker 1: {id:1, dist:0.1}, {id:3, dist:0.3}
    worker_results.push_back(CreateWorkerResult(1, 100, {{1, 0.1f}, {3, 0.3f}}));
    
    // Worker 2: {id:2, dist:0.2}, {id:4, dist:0.4}
    worker_results.push_back(CreateWorkerResult(2, 100, {{2, 0.2f}, {4, 0.4f}}));
    
    AggregatedResult result = aggregator_->aggregateResults(toPtrs(worker_results), 3);
    auto extracted = ExtractResults(result);
    
    EXPECT_EQ(extracted.size(), 3);
    EXPECT_EQ(extracted[0], std::make_pair(1, 0.1f)); // Smallest distance
    EXPECT_EQ(extracted[1], std::make_pair(2, 0.2f));
    EXPECT_EQ(extracted[2], std::make_pair(3, 0.3f));
}

// Test when k is larger than available results
TEST_F(AggregatorTest, KLargerThanAvailableResults) {
    std::vector<WorkerResult> worker_results;
    
    worker_results.push_back(CreateWorkerResult(1, 100, {{1, 0.1f}, {2, 0.2f}}));
    worker_results.push_back(CreateWorkerResult(2, 100, {{3, 0.3f}}));
    
    AggregatedResult result = aggregator_->aggregateResults(toPtrs(worker_results), 10); // Ask for more than available
    auto extracted = ExtractResults(result);
    
    EXPECT_EQ(extracted.size(), 3); // Should return all available results
    EXPECT_EQ(extracted[0], std::make_pair(1, 0.1f));
    EXPECT_EQ(extracted[1], std::make_pair(2, 0.2f));
    EXPECT_EQ(extracted[2], std::make_pair(3, 0.3f));
}

// Test with k=1 (single result)
TEST_F(AggregatorTest, SingleResult) {
    std::vector<WorkerResult> worker_results;
    
    worker_results.push_back(CreateWorkerResult(1, 100, {{1, 0.5f}, {2, 0.6f}}));
    worker_results.push_back(CreateWorkerResult(2, 100, {{3, 0.1f}, {4, 0.7f}}));
    
    AggregatedResult result = aggregator_->aggregateResults(toPtrs(worker_results), 1);
    auto extracted = ExtractResults(result);
    
    EXPECT_EQ(extracted.size(), 1);
    EXPECT_EQ(extracted[0], std::make_pair(3, 0.1f)); // Globally best result
}

// Test empty worker results
TEST_F(AggregatorTest, EmptyWorkerResults) {
    std::vector<WorkerResult> worker_results;
    
    AggregatedResult result = aggregator_->aggregateResults(toPtrs(worker_results), 5);
    
    EXPECT_EQ(result.topk_size(), 0);
}

// Test workers with empty result lists
TEST_F(AggregatorTest, WorkersWithEmptyResults) {
    std::vector<WorkerResult> worker_results;
    
    // Worker 1: empty results
    worker_results.push_back(CreateWorkerResult(1, 100, {}));
    
    // Worker 2: has results
    worker_results.push_back(CreateWorkerResult(2, 100, {{1, 0.1f}, {2, 0.2f}}));
    
    // Worker 3: empty results
    worker_results.push_back(CreateWorkerResult(3, 100, {}));
    
    AggregatedResult result = aggregator_->aggregateResults(toPtrs(worker_results), 3);
    auto extracted = ExtractResults(result);
    
    EXPECT_EQ(extracted.size(), 2);
    EXPECT_EQ(extracted[0], std::make_pair(1, 0.1f));
    EXPECT_EQ(extracted[1], std::make_pair(2, 0.2f));
}

// Test invalid k values
TEST_F(AggregatorTest, InvalidKValue) {
    std::vector<WorkerResult> worker_results;
    worker_results.push_back(CreateWorkerResult(1, 100, {{1, 0.1f}}));
    
    EXPECT_THROW(aggregator_->aggregateResults(toPtrs(worker_results), 0), std::invalid_argument);
    EXPECT_THROW(aggregator_->aggregateResults(toPtrs(worker_results), -1), std::invalid_argument);
}

// Test complex merge scenario with multiple workers
TEST_F(AggregatorTest, ComplexMergeScenario) {
    std::vector<WorkerResult> worker_results;
    
    // Worker 1: distances in ascending order
    worker_results.push_back(CreateWorkerResult(1, 100, {
        {10, 0.05f}, {11, 0.15f}, {12, 0.25f}, {13, 0.35f}
    }));
    
    // Worker 2: distances in ascending order  
    worker_results.push_back(CreateWorkerResult(2, 100, {
        {20, 0.10f}, {21, 0.20f}, {22, 0.30f}, {23, 0.40f}
    }));
    
    // Worker 3: distances in ascending order
    worker_results.push_back(CreateWorkerResult(3, 100, {
        {30, 0.08f}, {31, 0.18f}, {32, 0.28f}
    }));
    
    AggregatedResult result = aggregator_->aggregateResults(toPtrs(worker_results), 6);
    auto extracted = ExtractResults(result);
    
    EXPECT_EQ(extracted.size(), 6);
    // Should be sorted by distance: 0.05, 0.08, 0.10, 0.15, 0.18, 0.20
    EXPECT_EQ(extracted[0], std::make_pair(10, 0.05f));
    EXPECT_EQ(extracted[1], std::make_pair(30, 0.08f));
    EXPECT_EQ(extracted[2], std::make_pair(20, 0.10f));
    EXPECT_EQ(extracted[3], std::make_pair(11, 0.15f));
    EXPECT_EQ(extracted[4], std::make_pair(31, 0.18f));
    EXPECT_EQ(extracted[5], std::make_pair(21, 0.20f));
}


// Performance test with large number of results
TEST_F(AggregatorTest, LargeResultSet) {
    std::vector<WorkerResult> worker_results;
    
    // Create 5 workers each with 1000 results
    for (int worker = 0; worker < 5; ++worker) {
        std::vector<std::pair<int32_t, float>> neighbors;
        for (int i = 0; i < 1000; ++i) {
            // Non-overlapping IDs, ascending distances
            neighbors.emplace_back(worker * 1000 + i, i * 0.001f);
        }
        worker_results.push_back(CreateWorkerResult(worker, 100, neighbors));
    }
    
    AggregatedResult result = aggregator_->aggregateResults(toPtrs(worker_results), 100);
    auto extracted = ExtractResults(result);
    
    EXPECT_EQ(extracted.size(), 100);
    // First 100 results should be the smallest distances from all workers
    // Since each worker starts from distance 0.0, we should get first 20 from each worker
    for (int i = 0; i < 100; ++i) {
        float expected_distance = (i / 5) * 0.001f;
        EXPECT_FLOAT_EQ(extracted[i].second, expected_distance);
    }
}

} // namespace test
} // namespace ann