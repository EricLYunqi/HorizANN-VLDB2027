#pragma once

#include <vector>
#include "proto/ann.pb.h"
#include "proto/ann.grpc.pb.h"

namespace ann {

class IAggregator {
public:
    virtual ~IAggregator() = default;

    virtual AggregatedResult aggregateResults(const std::vector<const WorkerResult*>& worker_results, int k) = 0;
};

} // namespace ann


