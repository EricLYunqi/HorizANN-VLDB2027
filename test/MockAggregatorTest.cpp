#include <gtest/gtest.h>
#include "mock/MockAggregator.h"
#include "proto/ann.pb.h"

namespace ann {

class MockAggregatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        aggregator = std::make_unique<MockAggregator>();
    }
    
    std::unique_ptr<MockAggregator> aggregator;

    std::vector<const WorkerResult*> toPtrs(const std::vector<WorkerResult>& v) {
        std::vector<const WorkerResult*> ptrs;
        ptrs.reserve(v.size());
        for (const auto& r : v) ptrs.push_back(&r);
        return ptrs;
    }
};

TEST_F(MockAggregatorTest, AggregateEmptyResults) {
    std::vector<WorkerResult> worker_results;
    int k = 5;
    
    AggregatedResult result = aggregator->aggregateResults(toPtrs(worker_results), k);
    
    EXPECT_EQ(result.topk_size(), 0);
}

TEST_F(MockAggregatorTest, AggregateSingleWorkerResult) {
    std::vector<WorkerResult> worker_results;
    
    // Create a mock worker result
    WorkerResult worker_result;
    worker_result.set_worker_id(1);
    
    auto* neighbor1 = worker_result.add_results();
    neighbor1->set_id(100);
    neighbor1->set_distance(0.5f);
    
    auto* neighbor2 = worker_result.add_results();
    neighbor2->set_id(200);
    neighbor2->set_distance(1.0f);
    
    worker_results.push_back(worker_result);
    
    AggregatedResult result = aggregator->aggregateResults(toPtrs(worker_results), 2);
    
    EXPECT_EQ(result.topk_size(), 2);
    EXPECT_EQ(result.topk(0).id(), 100);
    EXPECT_FLOAT_EQ(result.topk(0).distance(), 0.5f);
    EXPECT_EQ(result.topk(1).id(), 200);
    EXPECT_FLOAT_EQ(result.topk(1).distance(), 1.0f);
}

TEST_F(MockAggregatorTest, AggregateMultipleWorkerResults) {
    std::vector<WorkerResult> worker_results;
    
    // Worker 1 result
    WorkerResult worker1_result;
    worker1_result.set_worker_id(1);
    auto* neighbor1 = worker1_result.add_results();
    neighbor1->set_id(100);
    neighbor1->set_distance(0.5f);
    worker_results.push_back(worker1_result);
    
    // Worker 2 result
    WorkerResult worker2_result;
    worker2_result.set_worker_id(2);
    auto* neighbor2 = worker2_result.add_results();
    neighbor2->set_id(200);
    neighbor2->set_distance(0.3f);
    worker_results.push_back(worker2_result);
    
    AggregatedResult result = aggregator->aggregateResults(toPtrs(worker_results), 2);
    
    EXPECT_EQ(result.topk_size(), 2);
    // Results should be sorted by distance (ascending)
    EXPECT_EQ(result.topk(0).id(), 200);  // distance 0.3
    EXPECT_FLOAT_EQ(result.topk(0).distance(), 0.3f);
    EXPECT_EQ(result.topk(1).id(), 100);  // distance 0.5
    EXPECT_FLOAT_EQ(result.topk(1).distance(), 0.5f);
}

} // namespace ann
