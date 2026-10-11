#include "aggregator.h"
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace ann {

struct WorkerIterator {
    int worker_idx;
    int neighbor_idx;
    float distance;
    int64_t id;

    WorkerIterator(int w_idx, int n_idx, const WorkerResult& worker_result) 
        : worker_idx(w_idx), neighbor_idx(n_idx) {
        if (n_idx < worker_result.results_size()) {
            const auto& neighbor = worker_result.results(n_idx);
            distance = neighbor.distance();
            id = neighbor.id();
        }
    }

    bool operator>(const WorkerIterator& other) const {
        return distance > other.distance;
    }
};

Aggregator::Aggregator() {
    
}

AggregatedResult Aggregator::aggregateResults(const std::vector<const WorkerResult*>& worker_results, int k) {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }

    if (worker_results.empty()) {
        return AggregatedResult();
    }

    std::priority_queue<WorkerIterator, std::vector<WorkerIterator>, std::greater<WorkerIterator>> pq;

    for (int i = 0; i < static_cast<int>(worker_results.size()); ++i) {
        const auto& worker_result = *worker_results[static_cast<size_t>(i)];
        if (worker_result.results_size() > 0) {
            pq.emplace(i, 0, worker_result);
        }
    }
    
    
    
    // Create the result with top-k neighbors
    AggregatedResult result;
    
    int collected = 0;
    
    // K-way merge with duplicate handling
    while (!pq.empty() && collected < k) {
        WorkerIterator current = pq.top();
        pq.pop();
        
        Neighbor* neighbor = result.add_topk();
        neighbor->set_id(current.id);
        neighbor->set_distance(current.distance);
        collected++;

        // Advance iterator for this worker and add back to heap if more elements exist
        int next_idx = current.neighbor_idx + 1;
        const auto& worker_result = *worker_results[static_cast<size_t>(current.worker_idx)];
        if (next_idx < worker_result.results_size()) {
            pq.emplace(current.worker_idx, next_idx, worker_result);
        }
    }
    
    return result;
}

} // namespace ann
