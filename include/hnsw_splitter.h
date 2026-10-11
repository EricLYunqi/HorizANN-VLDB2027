#pragma once

#include "hnswalg.h"
#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <limits>
#include <utility>
#include <thread>
#include <chrono>

namespace hnsw_splitter {

template <typename dist_t>
class HNSWSplitter {
public:
    explicit HNSWSplitter(hnswlib::HierarchicalNSW<dist_t>* original)
        : orig_(original) {}

    // In-place N-way split. `assignments[i]` is the destination child id
    // (0..num_children-1) for original internal id i.
    // Child 0 stays in orig_ (modified in-place).
    // child_spaces has num_children-1 entries for children 1..num_children-1
    // (the indexes to be transferred).
    // Returns children 1..num_children-1 as newly allocated indexes.
    // After return, orig_ has been compacted to contain only child-0 elements.
    //
    // dont_create_structures_unused_in_save: when true (production), transfer children omit
    // link_list_locks_ / VisitedListPool / label_lookup_ until saveIndex+loadIndex on the worker.
    // When false (typical unit tests), transfer children are fully materialized in memory.
    std::vector<std::unique_ptr<hnswlib::HierarchicalNSW<dist_t>>>
    splitInPlace(const std::vector<uint32_t>& assignments,
                 const std::vector<hnswlib::SpaceInterface<dist_t>*>& child_spaces,
                 bool enable_repair = true,
                 bool dont_create_structures_unused_in_save = true) {
        using tableint = hnswlib::tableint;
        using linklistsizeint = hnswlib::linklistsizeint;
        using labeltype = hnswlib::labeltype;
        // using GhostLink = typename hnswlib::HierarchicalNSW<dist_t>::GhostLink;

        const size_t N = orig_->getCurrentElementCount();
        if (assignments.size() != N) {
            throw std::runtime_error("Assignment vector size must match index element count");
        }
        const size_t num_children = child_spaces.size() + 1;
        if (num_children < 2) {
            throw std::runtime_error("splitInPlace requires at least one child space");
        }

        // ---- Pass 1: count + map ----
        auto t_pass1_start = std::chrono::steady_clock::now();

        std::vector<tableint> old_to_new(N, (tableint)-1);
        std::vector<size_t> child_counts(num_children, 0);

        for (tableint oldId = 0; oldId < (tableint)N; ++oldId) {
            const uint32_t child_id = assignments[oldId];
            if (child_id >= num_children) {
                throw std::runtime_error("Assignment value out of range");
            }
            old_to_new[oldId] = static_cast<tableint>(child_counts[child_id]++);
        }

        auto t_pass1_end = std::chrono::steady_clock::now();
        double pass1_ms = std::chrono::duration<double, std::milli>(t_pass1_end - t_pass1_start).count();

        // ---- Allocate children 1..num_children-1 ----
        std::vector<std::unique_ptr<hnswlib::HierarchicalNSW<dist_t>>> children(num_children - 1);
        for (size_t c = 0; c < num_children - 1; ++c) {
            if (!child_spaces[c]) {
                throw std::runtime_error("Child space pointer cannot be null");
            }
            children[c] = cloneSaveSkeleton(child_spaces[c], child_counts[c + 1],
                                            dont_create_structures_unused_in_save);
            children[c]->maxlevel_ = -1;
            children[c]->enterpoint_node_ = (tableint)-1;
            children[c]->ghost_links_in_memory_ = true;
            // children[c]->ghost_links_.resize(child_counts[c + 1]);
            // new: populate the edge vectors later
            children[c]->initGhostEdges(child_counts[c + 1]);
        }

        // Old: Ensure orig_ has ghost_links_ initialized
        // if (!orig_->ghost_links_in_memory_) {
        //     orig_->ghost_links_in_memory_ = true;
        //     orig_->ghost_links_.resize(N);
        // } else if (orig_->ghost_links_.size() < N) {
        //     orig_->ghost_links_.resize(N);
        // }

        // new: Ensure orig_ has empty ghost CSR initialized.
        if (!orig_->ghost_links_in_memory_) {
            orig_->ghost_links_in_memory_ = true;
            orig_->initGhostEdges(N);
        }

        // this shouldn't happen in normal cases
        const size_t cur_orig_node_cnt = orig_->node_level_offsets_.empty() ? 0 : orig_->node_level_offsets_.size() - 1;
        if (cur_orig_node_cnt < N) {
            std::cout << "[HNSWSplitter] ghost edges storage mismatch: current capacity: " << cur_orig_node_cnt << " nodes, total: " << N
                      << " nodes in index (the missing nodes start with no ghost edges)"
                      << std::endl;
        }

        // ---- Pass 2: parallel copy + adjacency rewrite ----
        auto t_pass2_start = std::chrono::steady_clock::now();

        // L0 zero-degree internal ids 
        std::vector<tableint> zero_deg_child0;

        const size_t stride = orig_->size_data_per_element_;

        const unsigned hw = std::thread::hardware_concurrency();
        const size_t num_workers = std::max<size_t>(1, std::min(N, hw == 0 ? 1u : static_cast<size_t>(hw)));
        const tableint n = static_cast<tableint>(N);
        std::vector<std::vector<tableint>> zero_deg_child0_by_worker(num_workers);

        // 2-a
        auto t_pass2a_start = std::chrono::steady_clock::now();

        // tmp struct
        struct GhostCsr {
            std::vector<uint32_t>  node_off;
            std::vector<uint64_t>  edge_off;
            std::vector<uint16_t>  edge_cnt;
            std::vector<labeltype> label;
            std::vector<dist_t>    distance;
        };
        std::vector<GhostCsr> ghost_build(num_children);

        // the old implementation assumes there may be stale ghost edges in memory from last run, so we need to "append" new edges in CSR.
        // counting and filling both read the live prefix through getGhostEdges*, so they cannot disagree.
        // activateGhostLinks leaves a dead tail behind, it is dropped here because the new slots are sized from the live count.

        // assign offsets
        for (size_t c = 0; c < num_children; ++c)
            ghost_build[c].node_off.assign(child_counts[c] + 1, 0);

        {
            std::vector<uint64_t> running(num_children, 0);
            const tableint tmpN = static_cast<tableint>(N); 

            for (tableint oldId = 0; oldId < tmpN; ++oldId) {
                const uint32_t c = assignments[oldId];
                running[c] += static_cast<uint64_t>(orig_->element_levels_[oldId]) + 1;

                if (running[c] > std::numeric_limits<uint32_t>::max()) {
                    std::string err_msg = "[HNSWSplitter] ghost CSR level-slot count overflows uint32_t for child " + std::to_string(c);
                    throw std::runtime_error(err_msg);
                }

                ghost_build[c].node_off[old_to_new[oldId] + 1] = static_cast<uint32_t>(running[c]);
            }

            for (size_t c = 0; c < num_children; ++c) {
                ghost_build[c].edge_off.assign((size_t)ghost_build[c].node_off.back() + 1, 0);
            }
        }

        // count edges
        auto ghost_count_task = [&](size_t worker_id) {
            const tableint begin = static_cast<tableint>((worker_id * (size_t)n) / num_workers);
            const tableint end = static_cast<tableint>(((worker_id + 1) * (size_t)n) / num_workers);

            for (tableint oldId = begin; oldId < end; ++oldId) {
                const uint32_t c = assignments[oldId];
                auto& csr = ghost_build[c];
                const uint32_t base = csr.node_off[old_to_new[oldId]];
                const int lvl = orig_->element_levels_[oldId];

                for (int level = 0; level <= lvl; ++level) {
                    linklistsizeint* ll = get_linklist_at_level(*orig_, oldId, level);
                    const unsigned short sz = ll ? get_list_count(*orig_, ll) : 0;
                    const tableint* nei = ll ? (const tableint*)(ll + 1) : nullptr;

                    uint64_t crossing = 0;
                    for (unsigned short j = 0; j < sz; ++j) {
                        if (assignments[nei[j]] != c) 
                            ++ crossing;
                    }

                    const uint64_t total = orig_->getGhostEdgesCount(oldId, level) + crossing;
                    csr.edge_off[(size_t)base + level + 1] = std::min<uint64_t>(total, maxGhostDegree(*orig_, level));
                }
            }
        };

        {
            std::vector<std::thread> threads;
            threads.reserve(num_workers - 1);
            for (size_t w = 1; w < num_workers; ++w) {
                threads.emplace_back(ghost_count_task, w);
            }
            ghost_count_task(0);
            for (auto& t : threads) {
                t.join();
            }
        }

        // turn per-slot counts into offsets, then allocate the edge arrays.
        // the fill writes every slot to capacity, so live count == capacity coming out of a split
        for (size_t c = 0; c < num_children; ++c) {
            auto& eo = ghost_build[c].edge_off;
            auto& ec = ghost_build[c].edge_cnt;
            ec.assign(eo.size() - 1, 0);

            uint64_t running = 0;
            for (size_t slot = 1; slot < eo.size(); ++slot) {
                ec[slot - 1] = static_cast<uint16_t>(eo[slot]);
                running += eo[slot];
                eo[slot] = running;
            }

            ghost_build[c].label.resize(running);
            ghost_build[c].distance.resize(running);
        }

        auto t_pass2a_end = std::chrono::steady_clock::now();
        double pass2a_ms = std::chrono::duration<double, std::milli>(t_pass2a_end - t_pass2a_start).count();

        // gather the edges carried over from the last split, they always come first
        auto take_old_ghost = [&](tableint oldId, int level, std::vector<GhostCand>& cands) {
            unsigned short length = 0;
            const labeltype* label = orig_->getGhostEdgesLabel(oldId, level, length);
            const dist_t* distance = orig_->getGhostEdgesDistance(oldId, level, length);

            for (unsigned short k = 0; k < length; ++k)
                cands.push_back(GhostCand{label[k], distance[k]});
        };

        // the slot was sized min(previous + crossing, cap), so this only trims
        // when the union overflowed the cap
        auto write_ghost_slot = [&](GhostCsr& csr, size_t slot, std::vector<GhostCand>& cands) {
            const uint64_t lo = csr.edge_off[slot];
            capGhostCands(cands, (size_t)(csr.edge_off[slot + 1] - lo));

            for (size_t k = 0; k < cands.size(); ++k) {
                csr.label[lo + k] = cands[k].label;
                csr.distance[lo + k] = cands[k].distance;
            }
        };

        // fill edges
        auto worker_task = [&](size_t worker_id) {
            const tableint begin = static_cast<tableint>((worker_id * (size_t)n) / num_workers);
            const tableint end = static_cast<tableint>(((worker_id + 1) * (size_t)n) / num_workers);
            auto& local_zero_deg = zero_deg_child0_by_worker[worker_id];

            // reused across nodes so the gather never allocates per level
            std::vector<GhostCand> cands;
            cands.reserve(2 * orig_->maxM0_);

            for (tableint oldId = begin; oldId < end; ++oldId) {
                const uint32_t c = assignments[oldId];
                const tableint newId = old_to_new[oldId];
                const int lvl = orig_->element_levels_[oldId];

                auto& csr = ghost_build[c];
                const uint32_t base = csr.node_off[newId];

                if (c == 0) {
                    bool level0_zero = false;

                    for (int level = 0; level <= lvl; ++level) {
                        linklistsizeint* ll = get_linklist_at_level(*orig_, oldId, level);
                        const unsigned short sz = ll ? get_list_count(*orig_, ll) : 0;
                        auto* nei = ll ? (tableint*)(ll + 1) : nullptr;

                        cands.clear();
                        take_old_ghost(oldId, level, cands);

                        unsigned short kept = 0;
                        for (unsigned short j = 0; j < sz; ++j) {
                            const tableint nb = nei[j];
                            if (assignments[nb] != 0) {
                                cands.push_back(GhostCand{
                                    orig_->getExternalLabel(nb),
                                    orig_->fstdistfunc_(
                                        orig_->getDataByInternalId(oldId),
                                        orig_->getDataByInternalId(nb),
                                        orig_->dist_func_param_)});
                                continue;
                            }
                            nei[kept++] = old_to_new[nb];
                        }

                        write_ghost_slot(csr, (size_t)base + level, cands);
                        set_list_count(*orig_, ll, kept);

                        if (level == 0 && kept == 0)
                            level0_zero = true;
                    }

                    if (level0_zero)
                        local_zero_deg.push_back(newId);
                } else {
                    auto& dst = *children[c - 1];

                    std::memcpy(dst.data_level0_memory_ + (size_t)newId * stride,
                                orig_->data_level0_memory_ + (size_t)oldId * stride,
                                stride);

                    dst.element_levels_[newId] = lvl;

                    if (lvl > 0) {
                        const size_t bytes = (size_t)lvl * orig_->size_links_per_element_;
                        dst.linkLists_[newId] = (char*)std::malloc(bytes);
                        if (!dst.linkLists_[newId]) throw std::bad_alloc();
                        std::memcpy(dst.linkLists_[newId], orig_->linkLists_[oldId], bytes);
                    } else {
                        dst.linkLists_[newId] = nullptr;
                    }

                    const labeltype lbl = orig_->getExternalLabel(oldId);
                    dst.setExternalLabel(newId, lbl);

                    for (int level = 0; level <= lvl; ++level) {
                        linklistsizeint* src_ll = get_linklist_at_level(*orig_, oldId, level);
                        const unsigned short sz = src_ll ? get_list_count(*orig_, src_ll) : 0;
                        auto* src_nei = src_ll ? (tableint*)(src_ll + 1) : nullptr;

                        linklistsizeint* dst_ll = (level == 0)
                            ? dst.get_linklist0(newId)
                            : dst.get_linklist(newId, level);
                        auto* dst_nei = (tableint*)(dst_ll + 1);

                        cands.clear();
                        take_old_ghost(oldId, level, cands);

                        unsigned short kept = 0;
                        for (unsigned short j = 0; j < sz; ++j) {
                            const tableint nb = src_nei[j];
                            if (assignments[nb] != c) {
                                cands.push_back(GhostCand{
                                    orig_->getExternalLabel(nb),
                                    orig_->fstdistfunc_(
                                        orig_->getDataByInternalId(oldId),
                                        orig_->getDataByInternalId(nb),
                                        orig_->dist_func_param_)});
                                continue;
                            }
                            dst_nei[kept++] = old_to_new[nb];
                        }

                        write_ghost_slot(csr, (size_t)base + level, cands);
                        set_list_count(dst, dst_ll, kept);
                    }
                }
            }
        };

        {
            std::vector<std::thread> threads;
            threads.reserve(num_workers - 1);
            for (size_t w = 1; w < num_workers; ++w) {
                threads.emplace_back(worker_task, w);
            }
            worker_task(0);
            for (auto& t : threads) {
                t.join();
            }
        }

        // the incoming CSR is dead once the fill is done, so install the rebuilt ones.
        // they are written at newId, so unlike the nested form nothing is compacted in pass 3.
        // in_node_off / in_edge_off / old_ghost_range must not be touched past this point.
        orig_->node_level_offsets_ = std::move(ghost_build[0].node_off);
        orig_->edge_offsets_ = std::move(ghost_build[0].edge_off);
        orig_->edge_counts_ = std::move(ghost_build[0].edge_cnt);
        orig_->ghost_links_label_ = std::move(ghost_build[0].label);
        orig_->ghost_links_distance_ = std::move(ghost_build[0].distance);

        for (size_t c = 1; c < num_children; ++c) {
            auto& dst = *children[c - 1];
            dst.node_level_offsets_ = std::move(ghost_build[c].node_off);
            dst.edge_offsets_ = std::move(ghost_build[c].edge_off);
            dst.edge_counts_ = std::move(ghost_build[c].edge_cnt);
            dst.ghost_links_label_ = std::move(ghost_build[c].label);
            dst.ghost_links_distance_ = std::move(ghost_build[c].distance);
        }

        zero_deg_child0.reserve(child_counts[0]);
        for (auto& bucket : zero_deg_child0_by_worker) {
            zero_deg_child0.insert(zero_deg_child0.end(), bucket.begin(), bucket.end());
        }

        // Finalize metadata for children: counts + enterpoint / maxlevel
        {
            std::vector<std::thread> threads;
            threads.reserve(num_children - 1);
            for (size_t c = 1; c < num_children; ++c) {
                threads.emplace_back([&, c]() {
                    auto& dst = *children[c - 1];
                    dst.cur_element_count = child_counts[c];
                    int max_lvl = -1;
                    tableint ep = (tableint)-1;
                    for (tableint i = 0; i < (tableint)child_counts[c]; ++i) {
                        if (dst.element_levels_[i] > max_lvl) {
                            max_lvl = dst.element_levels_[i];
                            ep = i;
                        }
                    }
                    dst.maxlevel_ = max_lvl;
                    dst.enterpoint_node_ = ep;
                });
            }
            for (auto& t : threads) {
                t.join();
            }
        }

        auto t_pass2_end = std::chrono::steady_clock::now();
        double pass2_ms = std::chrono::duration<double, std::milli>(t_pass2_end - t_pass2_start).count();

        // ---- Pass 3: compact child-0 in orig_ ----
        auto t_pass3_start = std::chrono::steady_clock::now();

        const size_t n0 = child_counts[0];

        // 3a + 3b: move child-0 elements to contiguous positions, free non-child-0 memory.
        // Note: linkLists_[i] for level-0-only elements is uninitialized (garbage from
        // the original malloc), so we must guard free() with element_levels_ > 0.
        for (tableint oldId = 0; oldId < (tableint)N; ++oldId) {
            if (assignments[oldId] != 0) {
                if (orig_->element_levels_[oldId] > 0) {
                    std::free(orig_->linkLists_[oldId]);
                }
                orig_->linkLists_[oldId] = nullptr;
                continue;
            }

            const tableint newId = old_to_new[oldId];
            if (newId != oldId) {
                std::memcpy(orig_->data_level0_memory_ + (size_t)newId * stride,
                            orig_->data_level0_memory_ + (size_t)oldId * stride,
                            stride);
                orig_->linkLists_[newId] = (orig_->element_levels_[oldId] > 0)
                    ? orig_->linkLists_[oldId] : nullptr;
                orig_->linkLists_[oldId] = nullptr;
                orig_->element_levels_[newId] = orig_->element_levels_[oldId];
            }
        }

        // Free any remaining upper-level link lists beyond n0
        for (tableint i = (tableint)n0; i < (tableint)N; ++i) {
            if (orig_->element_levels_[i] > 0) {
                std::free(orig_->linkLists_[i]);
            }
            orig_->linkLists_[i] = nullptr;
        }

        // 3c: update metadata. the ghost CSR was already built at newId in pass 2
        orig_->cur_element_count = n0;

        // Rebuild label_lookup_
        orig_->label_lookup_.clear();
        orig_->label_lookup_.reserve(n0);
        int max_lvl = -1;
        tableint ep = (tableint)-1;
        for (tableint i = 0; i < (tableint)n0; ++i) {
            orig_->label_lookup_[orig_->getExternalLabel(i)] = i;
            if (orig_->element_levels_[i] > max_lvl) {
                max_lvl = orig_->element_levels_[i];
                ep = i;
            }
        }
        orig_->maxlevel_ = max_lvl;
        orig_->enterpoint_node_ = ep;

        auto t_pass3_end = std::chrono::steady_clock::now();
        double pass3_ms = std::chrono::duration<double, std::milli>(t_pass3_end - t_pass3_start).count();

        std::cout << "[HNSWSplitter] Split: nodes=" << N << ", children=" << num_children
                  << ", keep=" << n0
                  << ", remap internal ids=" << pass1_ms << "ms"
                  << ", size ghost CSR=" << pass2a_ms << "ms"
                  << ", rewrite correct neighbours in all child indexes=" << pass2_ms << "ms"
                  << ", compact the resident index=" << pass3_ms << "ms" << std::endl;

        // ---- Repair (child 0 only) ----
        // Pass collected L0 zero-degree ids; repairL0ZeroDegreeParallel rescans the whole index if empty.
        // Transfer children: saveIndex + loadIndex on worker, then repairL0ZeroDegreeParallel there.
        if (enable_repair) {
            auto t_repair_start = std::chrono::steady_clock::now();
            orig_->repairL0ZeroDegreeParallel(&zero_deg_child0);
            auto t_repair_end = std::chrono::steady_clock::now();
            double repair_ms = std::chrono::duration<double, std::milli>(t_repair_end - t_repair_start).count();
            std::cout << "[HNSWSplitter] Repair (resident child0): l0_zero_collected="
                      << zero_deg_child0.size() << ", time=" << repair_ms << "ms"
                      << " (transfer children deferred to post-load)" << std::endl;
        }

        // logGhostStats(*orig_, "original");
        // for (size_t c = 1; c < num_children; ++c) {
        //     logGhostStats(*children[c - 1], "child_" + std::to_string(c));
        // }

        return children;
    }

    // Count zero-degree nodes (nodes with no connections at any level)
    static int countZeroDegreeNodes(hnswlib::HierarchicalNSW<dist_t>& idx) {
        using tableint = hnswlib::tableint;
        using linklistsizeint = hnswlib::linklistsizeint;
        
        int count = 0;
        const size_t N = idx.getCurrentElementCount();
        
        for (tableint i = 0; i < (tableint)N; ++i) {
            if (idx.isMarkedDeleted(i)) continue;
            
            bool hasConnections = false;
            int elemLevel = idx.element_levels_[i];
            
            for (int level = 0; level <= elemLevel; ++level) {
                linklistsizeint* ll = (level == 0) ? idx.get_linklist0(i) : idx.get_linklist(i, level);
                if (ll && idx.getListCount(ll) > 0) {
                    hasConnections = true;
                    break;
                }
            }
            
            if (!hasConnections) {
                count++;
            }
        }
        
        return count;
    }

    static void repairZeroDegreeLevel0Node(hnswlib::HierarchicalNSW<dist_t>& idx,
                                           hnswlib::tableint internalId) {
        using tableint = hnswlib::tableint;

        if (idx.cur_element_count == 1) return;
        if (idx.isMarkedDeleted(internalId)) return;

        const void* dataPoint = idx.getDataByInternalId(internalId);
        idx.updatePoint(dataPoint, internalId, 1.0f);
    }

private:
    hnswlib::HierarchicalNSW<dist_t>* orig_;

    // one ghost edge while it is being gathered, before it lands in the CSR arrays
    struct GhostCand {
        hnswlib::labeltype label;
        dist_t distance;
    };

    // Ephemeral split transfer index: no link_list_locks_ / pool / label_lookup_ until load or merge.
    std::unique_ptr<hnswlib::HierarchicalNSW<dist_t>> cloneSaveSkeleton(
        hnswlib::SpaceInterface<dist_t>* space,
        size_t cap,
        bool dont_create_structures_unused_in_save) {

        auto dst = std::make_unique<hnswlib::HierarchicalNSW<dist_t>>(space, cap, orig_->M_, orig_->ef_construction_,
            /*random_seed=*/100, /*allow_replace_deleted=*/false, dont_create_structures_unused_in_save);
        dst->setEf(orig_->ef_);
        if (!dst->data_level0_memory_) throw std::bad_alloc();
        if (!dst->linkLists_) throw std::bad_alloc();
        return dst;
    }

    static inline hnswlib::linklistsizeint*
    get_linklist_at_level(const hnswlib::HierarchicalNSW<dist_t>& idx,
                          hnswlib::tableint internal_id, int level) {
        return (level == 0) ? idx.get_linklist0(internal_id)
                            : idx.get_linklist(internal_id, level);
    }

    static inline unsigned short
    get_list_count(const hnswlib::HierarchicalNSW<dist_t>& idx,
                   hnswlib::linklistsizeint* ll) {
        return idx.getListCount(ll);
    }

    static inline void
    set_list_count(hnswlib::HierarchicalNSW<dist_t>& idx,
                   hnswlib::linklistsizeint* ll, unsigned short count) {
        idx.setListCount(ll, count);
    }

    static inline size_t maxGhostDegree(const hnswlib::HierarchicalNSW<dist_t>& idx,
                                        int level) {
        return (level == 0) ? idx.maxM0_ : idx.maxM_;
    }

    static void capGhostCands(std::vector<GhostCand>& cands, size_t max_degree) {
        if (cands.size() <= max_degree) return;

        auto by_distance = [](const GhostCand& lhs, const GhostCand& rhs) {
            if (lhs.distance == rhs.distance) return lhs.label < rhs.label;
            return lhs.distance < rhs.distance;
        };

        std::nth_element(cands.begin(),
                         cands.begin() + static_cast<std::ptrdiff_t>(max_degree),
                         cands.end(),
                         by_distance);
        cands.resize(max_degree);
        std::sort(cands.begin(), cands.end(), by_distance);
    }

    struct GhostEdgeStats {
        size_t total_edges = 0;
        size_t level0_edges = 0;
        size_t max_level0_edges = 0;
    };

    static GhostEdgeStats countGhostEdges(const hnswlib::HierarchicalNSW<dist_t>& idx) {
        GhostEdgeStats stats;
        const size_t node_count = idx.node_level_offsets_.empty() ? 0 : idx.node_level_offsets_.size() - 1;

        for (size_t id = 0; id < node_count; ++id) {
            const uint32_t lo = idx.node_level_offsets_[id];
            const uint32_t hi = idx.node_level_offsets_[id + 1];
            size_t node_level0_edges = 0;

            for (uint32_t slot = lo; slot < hi; ++slot) {
                const size_t edge_count = (size_t)(idx.edge_offsets_[slot + 1] - idx.edge_offsets_[slot]);
                stats.total_edges += edge_count;
                if (slot == lo) {
                    stats.level0_edges += edge_count;
                    node_level0_edges = edge_count;
                }
            }

            stats.max_level0_edges = std::max(stats.max_level0_edges, node_level0_edges);
        }

        return stats;
    }

    static void logGhostStats(const hnswlib::HierarchicalNSW<dist_t>& idx,
                              const std::string& name) {
        const size_t node_count = idx.cur_element_count;
        const GhostEdgeStats stats = countGhostEdges(idx);
        const double avg_per_node = node_count == 0
            ? 0.0
            : static_cast<double>(stats.total_edges) / static_cast<double>(node_count);
        const double level0_avg_per_node = node_count == 0
            ? 0.0
            : static_cast<double>(stats.level0_edges) / static_cast<double>(node_count);

        std::cout << "[HNSWSplitter] Ghost stats [" << name << "]: "
                  << "nodes=" << node_count
                  << ", total_edges=" << stats.total_edges
                  << ", avg_per_node=" << avg_per_node
                  << ", level0_total_edges=" << stats.level0_edges
                  << ", level0_avg_per_node=" << level0_avg_per_node
                  << ", level0_max_per_node=" << stats.max_level0_edges
                  << std::endl;
    }

    static void fixZeroDegreeNodes(hnswlib::HierarchicalNSW<dist_t>& idx,
                                   const std::vector<hnswlib::tableint>& zeroDegreeNodes) {
        using tableint = hnswlib::tableint;
        using linklistsizeint = hnswlib::linklistsizeint;

        for (tableint internalId : zeroDegreeNodes) {
            if (idx.isMarkedDeleted(internalId)) continue;
            linklistsizeint* ll0 = idx.get_linklist0(internalId);
            if (!ll0 || idx.getListCount(ll0) != 0) continue;
            try {
                repairZeroDegreeLevel0Node(idx, internalId);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to repair zero-degree node " << internalId
                          << ": " << e.what() << std::endl;
            }
        }
    }
};

} // namespace hnsw_splitter
