#include <gtest/gtest.h>
#include "../include/hnswlib.h"
#include "../include/hnsw_splitter.h"
#include <vector>
#include <random>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <rocksdb/db.h>

namespace ann {

class GhostLinksTest : public ::testing::Test {
protected:
    static constexpr int DIM = 16;
    static constexpr int M = 16;
    static constexpr int EF_CONSTRUCTION = 100;
    static constexpr int EF_SEARCH = 50;

    using Index = hnswlib::HierarchicalNSW<float>;

    std::unique_ptr<hnswlib::L2Space> space_;

    void SetUp() override {
        space_ = std::make_unique<hnswlib::L2Space>(DIM);
    }

    std::unique_ptr<Index> buildIndex(size_t n, size_t max_elements, int seed = 42) {
        auto idx = std::make_unique<Index>(space_.get(), max_elements, M, EF_CONSTRUCTION);
        idx->setEf(EF_SEARCH);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 100.0f);
        for (size_t i = 0; i < n; i++) {
            std::vector<float> vec(DIM);
            for (int d = 0; d < DIM; d++) vec[d] = dist(rng);
            idx->addPoint(vec.data(), i);
        }
        return idx;
    }

    // Build two indexes from two spatially separated clusters so
    // the splitter assigns them cleanly to different partitions.
    struct SplitResult {
        std::unique_ptr<Index> A, B;
        std::unique_ptr<hnswlib::L2Space> spaceA, spaceB;
        std::vector<std::vector<float>> all_vectors;
        std::vector<hnswlib::labeltype> all_labels;
    };

    SplitResult buildAndSplit(size_t n_per_cluster, int seed = 42) {
        size_t n = n_per_cluster * 2;
        auto original = std::make_unique<Index>(space_.get(), n, M, EF_CONSTRUCTION);
        original->setEf(EF_SEARCH);

        std::mt19937 rng(seed);
        std::normal_distribution<float> noise(0.0f, 1.0f);

        SplitResult result;
        result.all_vectors.resize(n);
        result.all_labels.resize(n);

        for (size_t i = 0; i < n; i++) {
            std::vector<float> vec(DIM);
            float offset = (i < n_per_cluster) ? 0.0f : 50.0f;
            for (int d = 0; d < DIM; d++) vec[d] = offset + noise(rng);
            original->addPoint(vec.data(), i);
            result.all_vectors[i] = vec;
            result.all_labels[i] = i;
        }

        result.spaceA = std::make_unique<hnswlib::L2Space>(DIM);
        result.spaceB = std::make_unique<hnswlib::L2Space>(DIM);

        std::vector<uint32_t> assignments(n, 0);
        for (size_t i = 0; i < n; ++i) {
            assignments[i] = (i < n_per_cluster) ? 0U : 1U;
        }

        hnsw_splitter::HNSWSplitter<float> splitter(original.get());
        std::vector<hnswlib::SpaceInterface<float>*> spaces = {result.spaceB.get()};
        auto children = splitter.splitInPlace(assignments, spaces, true, false);
        result.A = std::move(original);
        result.B = std::move(children[0]);
        return result;
    }

    float computeRecall(Index& idx,
                        const std::vector<std::vector<float>>& vectors,
                        int k, int n_queries, int seed = 123) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<size_t> pick(0, vectors.size() - 1);

        size_t total_correct = 0;
        size_t total_expected = 0;

        auto brute_force_knn = [&](const float* query) {
            std::vector<std::pair<float, size_t>> dists;
            auto dist_func = space_->get_dist_func();
            auto dist_param = space_->get_dist_func_param();
            for (size_t i = 0; i < vectors.size(); i++) {
                float d = dist_func(query, vectors[i].data(), dist_param);
                dists.emplace_back(d, i);
            }
            std::sort(dists.begin(), dists.end());
            std::unordered_set<size_t> gt;
            for (int i = 0; i < k && i < (int)dists.size(); i++)
                gt.insert(dists[i].second);
            return gt;
        };

        for (int q = 0; q < n_queries; q++) {
            size_t qi = pick(rng);
            auto gt = brute_force_knn(vectors[qi].data());
            auto results = idx.searchKnn(vectors[qi].data(), k);
            while (!results.empty()) {
                if (gt.count(results.top().second)) total_correct++;
                results.pop();
            }
            total_expected += gt.size();
        }
        return (float)total_correct / (float)total_expected;
    }

    // Count how many level-0 neighbors of `id` are present in the index
    int getLevel0Degree(Index& idx, hnswlib::tableint id) {
        auto* ll = idx.get_linklist0(id);
        return idx.getListCount(ll);
    }

    // ---- ghost CSR helpers -------------------------------------------------
    // Ghost edges live in a flat CSR now, so tests read them through the
    // accessors instead of indexing a nested vector. The accessors hand back
    // the live prefix of a slot; the dead tail activateGhostLinks leaves behind
    // is never visible here.

    // Every live (label, distance) ghost edge of `id`, across all its levels.
    static std::vector<std::pair<hnswlib::labeltype, float>>
    ghostEdgesOf(const Index& idx, size_t id) {
        std::vector<std::pair<hnswlib::labeltype, float>> out;
        for (int l = 0; l <= idx.element_levels_[id]; ++l) {
            unsigned short len = 0;
            const hnswlib::labeltype* labels = idx.getGhostEdgesLabel(id, l, len);
            const float* distances = idx.getGhostEdgesDistance(id, l, len);
            for (unsigned short k = 0; k < len; ++k)
                out.emplace_back(labels[k], distances[k]);
        }
        return out;
    }

    // Total capacity of every slot, live or dead. capacity - getGhostLinkCount() is the slack.
    static uint64_t ghostCapacity(const Index& idx) {
        if (idx.node_level_offsets_.empty()) return 0;
        const uint32_t slots = idx.node_level_offsets_[idx.cur_element_count];
        return idx.edge_offsets_[slots];
    }

    // Structural invariants the CSR must hold at all times.
    static void expectCsrConsistent(const Index& idx) {
        const size_t n = idx.cur_element_count;
        ASSERT_EQ(idx.node_level_offsets_.size(), n + 1);

        const uint32_t slots = idx.node_level_offsets_[n];
        ASSERT_EQ(idx.edge_offsets_.size(), static_cast<size_t>(slots) + 1);
        ASSERT_EQ(idx.edge_counts_.size(), slots);
        EXPECT_EQ(idx.ghost_links_label_.size(), idx.edge_offsets_[slots]);
        EXPECT_EQ(idx.ghost_links_distance_.size(), idx.edge_offsets_[slots]);

        for (size_t i = 0; i < n; ++i) {
            const uint32_t span = idx.node_level_offsets_[i + 1] - idx.node_level_offsets_[i];
            EXPECT_TRUE(span == 0 || span == static_cast<uint32_t>(idx.element_levels_[i]) + 1)
                << "node " << i << " owns " << span << " slots but sits at level "
                << idx.element_levels_[i];
        }

        for (uint32_t s = 0; s < slots; ++s) {
            EXPECT_LE(idx.edge_offsets_[s], idx.edge_offsets_[s + 1]) << "offsets not monotonic at slot " << s;
            EXPECT_LE(idx.edge_counts_[s], idx.edge_offsets_[s + 1] - idx.edge_offsets_[s])
                << "live count exceeds capacity at slot " << s;
        }
    }

    // Repoint one existing ghost edge at a label that is not in either half.
    // Slots are sized exactly, so a new edge cannot simply be pushed in; rewriting
    // a live label in place gives the same thing an unresolvable link would.
    // Returns the internal id whose edge was rewritten, or -1 if there were none.
    static long long injectUnresolvableGhostTarget(Index& idx, hnswlib::labeltype fake_label) {
        for (size_t i = 0; i < idx.cur_element_count; ++i) {
            for (int l = 0; l <= idx.element_levels_[i]; ++l) {
                const size_t slot = idx.ghostSlot(i, l);
                if (slot == Index::GHOST_NO_SLOT) continue;
                if (idx.edge_counts_[slot] == 0) continue;

                idx.ghost_links_label_[idx.edge_offsets_[slot]] = fake_label;
                return static_cast<long long>(i);
            }
        }
        return -1;
    }
};


// After splitting, both sub-indexes should have ghost links
TEST_F(GhostLinksTest, SplitProducesGhostLinks) {
    auto sr = buildAndSplit(500);

    EXPECT_TRUE(sr.A->ghost_links_in_memory_);
    EXPECT_TRUE(sr.B->ghost_links_in_memory_);
    EXPECT_GT(sr.A->getGhostLinkCount(), 0u);
    EXPECT_GT(sr.B->getGhostLinkCount(), 0u);

    expectCsrConsistent(*sr.A);
    expectCsrConsistent(*sr.B);
}


// Straight out of a split every slot is filled to capacity: the splitter sizes
// each one from the exact live count, so there is no slack until a merge resolves
// something.
TEST_F(GhostLinksTest, SplitLeavesNoSlack) {
    auto sr = buildAndSplit(500);

    EXPECT_EQ(sr.A->getGhostLinkCount(), ghostCapacity(*sr.A));
    EXPECT_EQ(sr.B->getGhostLinkCount(), ghostCapacity(*sr.B));
}


// After merging split halves, ghost links should be fully resolved
TEST_F(GhostLinksTest, MergeResolvesAllGhostLinks) {
    auto sr = buildAndSplit(500);

    size_t ghosts_A = sr.A->getGhostLinkCount();
    size_t ghosts_B = sr.B->getGhostLinkCount();
    EXPECT_GT(ghosts_A + ghosts_B, 0u);

    sr.A->mergeIndex(*sr.B);

    EXPECT_EQ(sr.A->getCurrentElementCount(), 1000u);

    size_t remaining = sr.A->getGhostLinkCount();
    EXPECT_EQ(remaining, 0u) << "Remaining ghost links after merge: " << remaining;
    EXPECT_TRUE(sr.A->ghost_links_in_memory_);
    expectCsrConsistent(*sr.A);
}


// Resolving shrinks slots in place: the live count drops to zero but the
// capacity stays put, because no offset is allowed to move.
TEST_F(GhostLinksTest, MergeLeavesSlackBehind) {
    auto sr = buildAndSplit(500);
    sr.A->mergeIndex(*sr.B);

    EXPECT_EQ(sr.A->getGhostLinkCount(), 0u);
    EXPECT_GT(ghostCapacity(*sr.A), 0u) << "capacity should survive the shrink";
    expectCsrConsistent(*sr.A);
}


// Merged index element count equals sum of parts
TEST_F(GhostLinksTest, MergePreservesElementCount) {
    auto sr = buildAndSplit(300);

    size_t nA = sr.A->getCurrentElementCount();
    size_t nB = sr.B->getCurrentElementCount();

    sr.A->mergeIndex(*sr.B);

    EXPECT_EQ(sr.A->getCurrentElementCount(), nA + nB);
    EXPECT_EQ(sr.A->getCurrentElementCount(), 600u);
}


// All original labels are searchable in the merged index
TEST_F(GhostLinksTest, MergedIndexContainsAllLabels) {
    auto sr = buildAndSplit(500);
    sr.A->mergeIndex(*sr.B);

    for (size_t label = 0; label < 1000; label++) {
        auto results = sr.A->searchKnn(sr.all_vectors[label].data(), 1);
        ASSERT_FALSE(results.empty());
        EXPECT_EQ(results.top().second, label);
    }
}


// Recall of merged index should be close to a single monolithic index
TEST_F(GhostLinksTest, MergedIndexRecallIsHigh) {
    const size_t N = 2000;
    auto sr = buildAndSplit(N / 2);
    sr.A->mergeIndex(*sr.B);
    sr.A->setEf(EF_SEARCH);

    float recall = computeRecall(*sr.A, sr.all_vectors, 10, 200);
    EXPECT_GE(recall, 0.85f) << "Merged index recall@10 = " << recall;
}


// activateGhostLinks only resolves ghost links whose targets exist.
// A link pointing at a label that is in neither half must survive the merge.
TEST_F(GhostLinksTest, UnresolvedGhostLinksAreKept) {
    auto sr = buildAndSplit(500);

    const hnswlib::labeltype fake_label = 999999;
    const long long node = injectUnresolvableGhostTarget(*sr.A, fake_label);
    ASSERT_GE(node, 0) << "no ghost edge to repoint";

    sr.A->mergeIndex(*sr.B);

    EXPECT_TRUE(sr.A->ghost_links_in_memory_);
    expectCsrConsistent(*sr.A);

    bool found_fake = false;
    for (const auto& [label, distance] : ghostEdgesOf(*sr.A, static_cast<size_t>(node))) {
        (void)distance;
        if (label == fake_label) { found_fake = true; break; }
    }
    EXPECT_TRUE(found_fake) << "unresolvable ghost link was dropped by the merge";

    // everything else resolved, so the surviving link is the only live edge left
    EXPECT_EQ(sr.A->getGhostLinkCount(), 1u);
}


// Merging an empty index into a non-empty one is a no-op
TEST_F(GhostLinksTest, MergeEmptyIndexIsNoop) {
    auto idx = buildIndex(500, 1000);
    auto empty = std::make_unique<Index>(space_.get(), 100, M, EF_CONSTRUCTION);

    size_t count_before = idx->getCurrentElementCount();
    idx->mergeIndex(*empty);
    EXPECT_EQ(idx->getCurrentElementCount(), count_before);
}


// Merging indexes with mismatched M should throw
TEST_F(GhostLinksTest, MergeMismatchedMThrows) {
    auto idx1 = std::make_unique<Index>(space_.get(), 100, 16, EF_CONSTRUCTION);
    auto idx2 = std::make_unique<Index>(space_.get(), 100, 8, EF_CONSTRUCTION);
    idx1->addPoint(std::vector<float>(DIM, 1.0f).data(), 0);
    idx2->addPoint(std::vector<float>(DIM, 2.0f).data(), 1);

    EXPECT_THROW(idx1->mergeIndex(*idx2), std::runtime_error);
}


// Ghost links are bidirectional: both split halves store ghost links
// for every cross-partition edge
TEST_F(GhostLinksTest, GhostLinksAreBidirectional) {
    auto sr = buildAndSplit(500);

    auto collectTargets = [](const Index& idx) {
        std::unordered_set<hnswlib::labeltype> targets;
        for (size_t i = 0; i < idx.cur_element_count; i++)
            for (const auto& [label, distance] : ghostEdgesOf(idx, i)) {
                (void)distance;
                targets.insert(label);
            }
        return targets;
    };

    std::unordered_set<hnswlib::labeltype> A_targets = collectTargets(*sr.A);
    std::unordered_set<hnswlib::labeltype> B_targets = collectTargets(*sr.B);

    // Every label that A points to via ghost links should be a node in B,
    // and that node in B should have a ghost link back to some node in A
    std::unordered_set<hnswlib::labeltype> A_labels, B_labels;
    for (size_t i = 0; i < sr.A->getCurrentElementCount(); i++)
        A_labels.insert(sr.A->getExternalLabel(i));
    for (size_t i = 0; i < sr.B->getCurrentElementCount(); i++)
        B_labels.insert(sr.B->getExternalLabel(i));

    // A's ghost targets should all be labels in B
    for (auto lbl : A_targets) {
        EXPECT_TRUE(B_labels.count(lbl)) << "A ghost target " << lbl << " not in B";
    }
    // B's ghost targets should all be labels in A
    for (auto lbl : B_targets) {
        EXPECT_TRUE(A_labels.count(lbl)) << "B ghost target " << lbl << " not in A";
    }

    // Both sides should have ghost links (non-empty)
    EXPECT_FALSE(A_targets.empty());
    EXPECT_FALSE(B_targets.empty());
}


// After split+merge, no node should have zero degree at level 0
TEST_F(GhostLinksTest, MergedIndexHasNoZeroDegreeNodes) {
    auto sr = buildAndSplit(500);
    sr.A->mergeIndex(*sr.B);

    for (size_t i = 0; i < sr.A->getCurrentElementCount(); i++) {
        EXPECT_GT(getLevel0Degree(*sr.A, (hnswlib::tableint)i), 0)
            << "Node " << i << " has zero degree after merge";
    }
}


// activateGhostLinks should produce valid neighbor lists (no self-loops,
// no duplicates, count <= Mcurmax)
TEST_F(GhostLinksTest, ActivatedLinksAreValid) {
    auto sr = buildAndSplit(500);
    sr.A->mergeIndex(*sr.B);

    size_t N = sr.A->getCurrentElementCount();
    for (size_t i = 0; i < N; i++) {
        auto* ll = sr.A->get_linklist0((hnswlib::tableint)i);
        unsigned short sz = sr.A->getListCount(ll);
        auto* data = (hnswlib::tableint*)(ll + 1);

        EXPECT_LE(sz, sr.A->maxM0_);

        std::unordered_set<hnswlib::tableint> seen;
        for (unsigned short j = 0; j < sz; j++) {
            EXPECT_NE(data[j], (hnswlib::tableint)i) << "Self-loop at node " << i;
            EXPECT_LT(data[j], (hnswlib::tableint)N) << "Out-of-range neighbor at node " << i;
            EXPECT_TRUE(seen.insert(data[j]).second) << "Duplicate neighbor at node " << i;
        }
    }
}


// Recall comparison: merged index vs monolithic index built from same data
TEST_F(GhostLinksTest, MergedRecallVsMonolithic) {
    const size_t N = 2000;
    auto sr = buildAndSplit(N / 2);

    // Build a monolithic index from the same vectors
    auto monolithic = std::make_unique<Index>(space_.get(), N, M, EF_CONSTRUCTION);
    monolithic->setEf(EF_SEARCH);
    for (size_t i = 0; i < N; i++) {
        monolithic->addPoint(sr.all_vectors[i].data(), sr.all_labels[i]);
    }

    sr.A->mergeIndex(*sr.B);
    sr.A->setEf(EF_SEARCH);

    float recall_mono = computeRecall(*monolithic, sr.all_vectors, 10, 200, 99);
    float recall_merged = computeRecall(*sr.A, sr.all_vectors, 10, 200, 99);

    EXPECT_GE(recall_merged, recall_mono * 0.90f)
        << "Merged recall=" << recall_merged << " vs mono=" << recall_mono;
}


// Save a split index (with unresolved ghost links), load it back,
// verify ghost links survive the round-trip
TEST_F(GhostLinksTest, SaveLoadPreservesGhostLinks) {
    auto sr = buildAndSplit(500);

    size_t ghost_count_before = sr.A->getGhostLinkCount();
    ASSERT_GT(ghost_count_before, 0u);

    std::string path = "/tmp/ghost_links_test_A.bin";
    sr.A->saveIndex(path);

    auto loaded = std::make_unique<Index>(sr.spaceA.get(), path, false, 0);

    EXPECT_TRUE(loaded->ghost_links_in_memory_);
    EXPECT_EQ(loaded->getGhostLinkCount(), ghost_count_before);
    EXPECT_EQ(loaded->getCurrentElementCount(), sr.A->getCurrentElementCount());
    expectCsrConsistent(*loaded);

    // Verify the actual ghost link content matches, per node and per level
    for (size_t i = 0; i < sr.A->getCurrentElementCount(); i++) {
        EXPECT_EQ(ghostEdgesOf(*sr.A, i), ghostEdgesOf(*loaded, i))
            << "ghost edges differ at node " << i;
    }

    std::remove(path.c_str());
}


// Loading an older ghost sidecar after appending new nodes should restore
// ghost links for the old prefix and leave newer nodes with empty ghost slots.
TEST_F(GhostLinksTest, LoadOlderGhostLinksAfterInsertKeepsNewNodesEmpty) {
    auto sr = buildAndSplit(500);

    const size_t old_count = sr.A->getCurrentElementCount();
    const size_t ghost_count_before = sr.A->getGhostLinkCount();
    ASSERT_GT(ghost_count_before, 0u);

    // save and load now share one handle, the way the worker drives them
    std::string path = "/tmp/ghost_links_sidecar_reload.db";
    std::filesystem::remove_all(path);

    rocksdb::Options opts;
    opts.create_if_missing = true;
    rocksdb::DB* raw_db = nullptr;
    ASSERT_TRUE(rocksdb::DB::Open(opts, path, &raw_db).ok());
    std::unique_ptr<rocksdb::DB> db(raw_db);

    ASSERT_TRUE(sr.A->saveGhostLinks(db.get()));
    sr.A->clearGhostLinksMemory();
    EXPECT_FALSE(sr.A->ghost_links_in_memory_);

    for (size_t i = 0; i < 10; ++i) {
        std::vector<float> vec(DIM, 200.0f + static_cast<float>(i));
        sr.A->addPoint(vec.data(), 100000 + i);
    }

    EXPECT_EQ(sr.A->getCurrentElementCount(), old_count + 10);
    sr.A->loadGhostLinks(db.get());
    EXPECT_TRUE(sr.A->ghost_links_in_memory_);
    EXPECT_EQ(sr.A->node_level_offsets_.size(), sr.A->getCurrentElementCount() + 1);
    EXPECT_EQ(sr.A->getGhostLinkCount(), ghost_count_before);
    expectCsrConsistent(*sr.A);

    for (size_t i = old_count; i < sr.A->getCurrentElementCount(); ++i) {
        EXPECT_TRUE(sr.A->isGhostEdgesEmpty(i)) << "node " << i << " should have no ghost edges";
    }

    db.reset();
    std::filesystem::remove_all(path);
}


// A save/load cycle through RocksDB writes only the live prefix, so it is what
// reclaims the slack that resolving leaves behind.
TEST_F(GhostLinksTest, RocksDbRoundTripReclaimsSlack) {
    auto sr = buildAndSplit(500);

    const hnswlib::labeltype fake_label = 999999;
    ASSERT_GE(injectUnresolvableGhostTarget(*sr.A, fake_label), 0);
    sr.A->mergeIndex(*sr.B);

    const size_t live = sr.A->getGhostLinkCount();
    ASSERT_GT(live, 0u);
    ASSERT_GT(ghostCapacity(*sr.A), live) << "merge should have left slack to reclaim";

    std::string path = "/tmp/ghost_links_compaction.db";
    std::filesystem::remove_all(path);

    rocksdb::Options opts;
    opts.create_if_missing = true;
    rocksdb::DB* raw_db = nullptr;
    ASSERT_TRUE(rocksdb::DB::Open(opts, path, &raw_db).ok());
    std::unique_ptr<rocksdb::DB> db(raw_db);

    ASSERT_TRUE(sr.A->saveGhostLinks(db.get()));
    sr.A->clearGhostLinksMemory();
    sr.A->loadGhostLinks(db.get());

    expectCsrConsistent(*sr.A);
    EXPECT_EQ(sr.A->getGhostLinkCount(), live) << "round-trip must not lose live edges";
    EXPECT_EQ(ghostCapacity(*sr.A), live) << "dead tail should be gone after a round-trip";

    db.reset();
    std::filesystem::remove_all(path);
}


// Save a split index with ghost links, load it, merge, and verify search works
TEST_F(GhostLinksTest, SaveLoadThenMerge) {
    auto sr = buildAndSplit(500);

    std::string pathA = "/tmp/ghost_links_test_merge_A.bin";
    std::string pathB = "/tmp/ghost_links_test_merge_B.bin";
    sr.A->saveIndex(pathA);
    sr.B->saveIndex(pathB);

    auto loadedA = std::make_unique<Index>(sr.spaceA.get(), pathA, false, 0);
    auto loadedB = std::make_unique<Index>(sr.spaceB.get(), pathB, false, 0);

    loadedA->mergeIndex(*loadedB);
    loadedA->setEf(EF_SEARCH);

    EXPECT_EQ(loadedA->getCurrentElementCount(), 1000u);
    EXPECT_TRUE(loadedA->ghost_links_in_memory_);
    EXPECT_EQ(loadedA->getGhostLinkCount(), 0u);

    // All original labels should be searchable
    for (size_t label = 0; label < 1000; label++) {
        auto results = loadedA->searchKnn(sr.all_vectors[label].data(), 1);
        ASSERT_FALSE(results.empty());
        EXPECT_EQ(results.top().second, label);
    }

    std::remove(pathA.c_str());
    std::remove(pathB.c_str());
}


// Save a merged index (no ghost links), load it, verify search works
TEST_F(GhostLinksTest, SaveLoadMergedIndex) {
    auto sr = buildAndSplit(500);
    sr.A->mergeIndex(*sr.B);
    sr.A->setEf(EF_SEARCH);

    std::string path = "/tmp/ghost_links_test_merged.bin";
    sr.A->saveIndex(path);

    auto loaded = std::make_unique<Index>(sr.spaceA.get(), path, false, 0);
    loaded->setEf(EF_SEARCH);

    EXPECT_FALSE(loaded->ghost_links_in_memory_);
    EXPECT_EQ(loaded->getCurrentElementCount(), 1000u);

    float recall = computeRecall(*loaded, sr.all_vectors, 10, 200);
    EXPECT_GE(recall, 0.85f) << "Loaded merged index recall@10 = " << recall;

    std::remove(path.c_str());
}


// An index saved without ghost links should still load correctly.
TEST_F(GhostLinksTest, SaveLoadWithoutGhostLinks) {
    auto idx = buildIndex(500, 1000);
    EXPECT_FALSE(idx->ghost_links_in_memory_);

    std::string path = "/tmp/ghost_links_test_no_ghosts.bin";
    idx->saveIndex(path);

    auto loaded = std::make_unique<Index>(space_.get(), path, false, 0);

    EXPECT_FALSE(loaded->ghost_links_in_memory_);
    EXPECT_EQ(loaded->getCurrentElementCount(), 500u);

    auto results = loaded->searchKnn(
        loaded->getDataByLabel<float>(0).data(), 5);
    EXPECT_EQ(results.size(), 5u);

    std::remove(path.c_str());
}

} // namespace ann
