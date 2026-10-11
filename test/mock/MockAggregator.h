#pragma once

#include "master/i_aggregator.h"

namespace ann {

class MockAggregator : public IAggregator {
public:
    ~MockAggregator() override = default;
    AggregatedResult aggregateResults(const std::vector<const WorkerResult*>& worker_results, int k) override;
};

} // namespace ann


