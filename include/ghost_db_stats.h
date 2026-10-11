#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <rocksdb/db.h>

namespace ann {

struct GhostDbStats {
    uint64_t ghost_linked_vector_count = 0;
    uint64_t ghost_edge_count = 0;
};

// Lightweight read-only scan of a ghost-link RocksDB directory, as written by
// HierarchicalNSW::saveGhostLinks / appendInsertionGhostLinks (see hnswalg.h).
// Sums edge counts directly from each value's header without deserializing
// labels/distances, and without touching the live in-memory ghost_links_
// structure. Returns all-zero stats if the DB doesn't exist yet (worker never
// wrote any ghost links).
// Takes the worker's live handle rather than opening its own: a second open would need the
// caller to close the long-lived one first, and reopening a 20GB+ store costs minutes.
inline GhostDbStats scanGhostLinkDb(rocksdb::DB* db) {
    GhostDbStats stats;
    if (!db) return stats;

    std::string meta_val;
    rocksdb::Status meta_status = db->Get(rocksdb::ReadOptions(), "__meta__", &meta_val);
    if (meta_status.ok() && meta_val.size() >= sizeof(uint64_t) * 2) {
        uint64_t active_nodes = 0;
        memcpy(&active_nodes, meta_val.data() + sizeof(uint64_t), sizeof(uint64_t));
        stats.ghost_linked_vector_count = active_nodes;
    }

    // Per-edge payload is [size_t label][dist_t distance]. dist_t is either
    // int or float in this codebase, both 4 bytes, so the skip size below is
    // correct regardless of which one wrote this particular file.
    constexpr size_t kPerEdgeBytes = sizeof(size_t) + sizeof(uint32_t);

    std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(rocksdb::ReadOptions()));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        if (it->key() == "__meta__") continue;

        rocksdb::Slice val = it->value();
        const char* ptr = val.data();
        const char* end = ptr + val.size();

        if (ptr + sizeof(uint32_t) > end) continue; // corrupted/truncated entry, skip
        uint32_t num_levels = 0;
        memcpy(&num_levels, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        for (uint32_t level = 0; level < num_levels; ++level) {
            if (ptr + sizeof(uint32_t) > end) break;
            uint32_t count = 0;
            memcpy(&count, ptr, sizeof(uint32_t));
            ptr += sizeof(uint32_t);

            stats.ghost_edge_count += count;
            ptr += static_cast<size_t>(count) * kPerEdgeBytes;
        }
    }

    return stats;
}

} // namespace ann
