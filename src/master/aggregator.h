#pragma once

#include <vector>
#include "proto/ann.pb.h"
#include "proto/ann.grpc.pb.h"
#include "i_aggregator.h"

namespace ann {

class Aggregator : public IAggregator {
public:
    Aggregator();
    ~Aggregator() = default;

    // Aggregate results from multiple workers and return top-k
    AggregatedResult aggregateResults(const std::vector<const WorkerResult*>& worker_results, int k) override;

};

} // namespace ann
