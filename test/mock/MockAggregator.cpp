#include "MockAggregator.h"
#include <algorithm>

namespace ann {

AggregatedResult MockAggregator::aggregateResults(const std::vector<const WorkerResult*>& worker_results, int k) {
    AggregatedResult result;
    std::vector<Neighbor> all;
    for (const auto* wr : worker_results) {
        for (const auto& n : wr->results()) all.push_back(n);
    }
    std::sort(all.begin(), all.end(), [](const Neighbor& a, const Neighbor& b){ return a.distance() < b.distance(); });
    if (k > static_cast<int>(all.size())) k = static_cast<int>(all.size());
    for (int i = 0; i < k; ++i) {
        auto* out = result.add_topk();
        out->set_id(all[static_cast<size_t>(i)].id());
        out->set_distance(all[static_cast<size_t>(i)].distance());
    }
    return result;
}

} // namespace ann


