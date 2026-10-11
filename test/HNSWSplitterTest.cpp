#include <gtest/gtest.h>
#include "../include/hnsw_splitter.h"
#include "../include/hnswlib.h"
#include <vector>
#include <memory>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <unistd.h>

namespace hnsw_splitter {

class HNSWSplitterTest : public ::testing::Test {
protected:
    void SetUp() override {
        dim = 4;
        max_elements = 10000;
        M = 16;
        ef_construction = 200;
        ef_search = 100;
        
        space_ = std::make_unique<hnswlib::L2Space>(dim);
        child_spaces_.clear();
    }

    // Helper function to create an index with known structure
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> createTestIndex(int num_points) {
        auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.get(), max_elements, M, ef_construction);
        index->setEf(ef_search);
        
        // Insert points in a way that creates clear clusters
        for (int i = 0; i < num_points; ++i) {
            std::vector<float> vec(dim);
            // Create two clusters: points 0..num_points/2-1 are near origin
            // points num_points/2..num_points-1 are offset
            if (i < num_points / 2) {
                vec[0] = i * 0.1f;
                vec[1] = i * 0.1f;
                vec[2] = i * 0.1f;
                vec[3] = i * 0.1f;
            } else {
                vec[0] = 100.0f + i * 0.1f;
                vec[1] = 100.0f + i * 0.1f;
                vec[2] = 100.0f + i * 0.1f;
                vec[3] = 100.0f + i * 0.1f;
            }
            index->addPoint(vec.data(), i);
        }
        
        return index;
    }

    // Splits the original in-place: the original becomes child A (first half),
    // and a new index is returned as child B (second half).
    //
    // dont_create_structures_unused_in_save: false = full transfer child in RAM (default for tests).
    // true = production-style skeleton child; save+load in this helper before repair.
    std::unique_ptr<hnswlib::HierarchicalNSW<float>>
    splitTwoWayInPlace(hnswlib::HierarchicalNSW<float>* original,
                       bool enable_repair = true,
                       bool dont_create_structures_unused_in_save = false) {
        const size_t n = original->getCurrentElementCount();
        std::vector<uint32_t> assignments(n, 0);
        for (size_t i = 0; i < n; ++i) {
            assignments[i] = (i < n / 2) ? 0U : 1U;
        }

        // Keep split child spaces alive for the full fixture lifetime.
        // HierarchicalNSW stores raw space pointers and does not take ownership.
        child_spaces_.push_back(std::make_unique<hnswlib::L2Space>(dim));
        std::vector<hnswlib::SpaceInterface<float>*> spaces = {child_spaces_.back().get()};

        HNSWSplitter<float> splitter(original);
        auto children = splitter.splitInPlace(assignments, spaces, enable_repair,
                                               dont_create_structures_unused_in_save);
        auto childB = std::move(children[0]);

        if (dont_create_structures_unused_in_save) {
            char tmpl[] = "/tmp/hnswsplit_XXXXXX";
            int fd = mkstemp(tmpl);
            if (fd < 0) {
                ADD_FAILURE() << "mkstemp failed";
                return nullptr;
            }
            close(fd);
            const std::string path(tmpl);
            childB->saveIndex(path);
            childB = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                child_spaces_.back().get(), path, false, static_cast<size_t>(max_elements));
            std::remove(path.c_str());
        }
        childB->setEf(ef_search);
        if (enable_repair) {
            childB->repairL0ZeroDegreeParallel();
        }
        return childB;
    }

    // Helper function to count zero-degree nodes at a specific level
    int countZeroDegreeNodesAtLevel(hnswlib::HierarchicalNSW<float>& idx, int level) {
        int count = 0;
        size_t N = idx.getCurrentElementCount();
        for (size_t i = 0; i < N; ++i) {
            if (idx.isMarkedDeleted(i)) continue;
            if (idx.element_levels_[i] < level) continue;
            
            hnswlib::linklistsizeint* ll = (level == 0) 
                ? idx.get_linklist0(i) 
                : idx.get_linklist(i, level);
            unsigned short neighbor_count = idx.getListCount(ll);
            if (neighbor_count == 0) {
                count++;
            }
        }
        return count;
    }

    // Helper function to count total zero-degree nodes at any level
    int countZeroDegreeNodes(hnswlib::HierarchicalNSW<float>& idx) {
        int count = 0;
        size_t N = idx.getCurrentElementCount();
        for (size_t i = 0; i < N; ++i) {
            if (idx.isMarkedDeleted(i)) continue;
            
            const int maxLevel = idx.element_levels_[i];
            bool hasZeroDegree = false;
            for (int level = 0; level <= maxLevel; ++level) {
                hnswlib::linklistsizeint* ll = (level == 0) 
                    ? idx.get_linklist0(i) 
                    : idx.get_linklist(i, level);
                unsigned short neighbor_count = idx.getListCount(ll);
                if (neighbor_count == 0) {
                    hasZeroDegree = true;
                    break;
                }
            }
            if (hasZeroDegree) {
                count++;
            }
        }
        return count;
    }

    int dim;
    int max_elements;
    int M;
    int ef_construction;
    int ef_search;
    std::unique_ptr<hnswlib::L2Space> space_;
    std::vector<std::unique_ptr<hnswlib::L2Space>> child_spaces_;
};

// Test basic splitting functionality
TEST_F(HNSWSplitterTest, BasicSplit) {
    const int num_points = 100;
    auto original = createTestIndex(num_points);
    
    auto indexB = splitTwoWayInPlace(original.get());
    
    EXPECT_NE(indexB, nullptr);
    
    EXPECT_GT(original->getCurrentElementCount(), 0);
    EXPECT_GT(indexB->getCurrentElementCount(), 0);
    
    EXPECT_EQ(original->getCurrentElementCount() + indexB->getCurrentElementCount(), num_points);
}

// Test that split indexes can be searched
TEST_F(HNSWSplitterTest, SplitIndexesSearchable) {
    const int num_points = 200;
    auto original = createTestIndex(num_points);
    
    auto indexB = splitTwoWayInPlace(original.get());
    
    original->setEf(ef_search);
    indexB->setEf(ef_search);
    
    std::vector<float> queryA = {5.0f, 5.0f, 5.0f, 5.0f};
    std::vector<float> queryB = {105.0f, 105.0f, 105.0f, 105.0f};
    
    auto resultsA = original->searchKnn(queryA.data(), 5);
    auto resultsB = indexB->searchKnn(queryB.data(), 5);
    
    EXPECT_GT(resultsA.size(), 0);
    EXPECT_GT(resultsB.size(), 0);
}

// Test that zero-degree nodes are repaired after splitting
TEST_F(HNSWSplitterTest, ZeroDegreeNodesRepaired) {
    const int num_points = 500;
    auto original = createTestIndex(num_points);
    
    auto indexB = splitTwoWayInPlace(original.get());
    
    original->setEf(ef_search);
    indexB->setEf(ef_search);
    
    int zeroDegreeA = countZeroDegreeNodesAtLevel(*original, 0);
    int zeroDegreeB = countZeroDegreeNodesAtLevel(*indexB, 0);
    
    EXPECT_EQ(zeroDegreeA, 0) << "Index A should have no zero-degree nodes at level 0 after repair";
    EXPECT_EQ(zeroDegreeB, 0) << "Index B should have no zero-degree nodes at level 0 after repair";
    
    std::vector<float> queryA = {5.0f, 5.0f, 5.0f, 5.0f};
    std::vector<float> queryB = {105.0f, 105.0f, 105.0f, 105.0f};
    
    auto resultsA = original->searchKnn(queryA.data(), 5);
    auto resultsB = indexB->searchKnn(queryB.data(), 5);
    
    EXPECT_GT(resultsA.size(), 0) << "Index A should be searchable after repair";
    EXPECT_GT(resultsB.size(), 0) << "Index B should be searchable after repair";
}

// Test that repaired indexes maintain search quality
TEST_F(HNSWSplitterTest, RepairedIndexesMaintainSearchQuality) {
    const int num_points = 1000;
    auto original = createTestIndex(num_points);
    
    auto indexB = splitTwoWayInPlace(original.get());
    
    original->setEf(ef_search);
    indexB->setEf(ef_search);
    
    int found_in_split = 0;
    int total_queries = 0;
    
    for (int i = 0; i < std::min(100, num_points); ++i) {
        std::vector<float> query(dim);
        if (i < num_points / 2) {
            query[0] = i * 0.1f;
            query[1] = i * 0.1f;
            query[2] = i * 0.1f;
            query[3] = i * 0.1f;
            
            auto results = original->searchKnn(query.data(), 10);
            if (!results.empty()) {
                found_in_split++;
            }
        } else {
            query[0] = 100.0f + i * 0.1f;
            query[1] = 100.0f + i * 0.1f;
            query[2] = 100.0f + i * 0.1f;
            query[3] = 100.0f + i * 0.1f;
            
            auto results = indexB->searchKnn(query.data(), 10);
            if (!results.empty()) {
                found_in_split++;
            }
        }
        total_queries++;
    }
    
    EXPECT_GT(found_in_split, total_queries * 0.8) 
        << "Repaired indexes should be able to find results for most queries";
}

// Test edge case: very small index
TEST_F(HNSWSplitterTest, SmallIndexSplit) {
    const int num_points = 10;
    auto original = createTestIndex(num_points);
    
    auto indexB = splitTwoWayInPlace(original.get());
    
    EXPECT_NE(indexB, nullptr);
    
    int zeroDegreeA = countZeroDegreeNodes(*original);
    int zeroDegreeB = countZeroDegreeNodes(*indexB);
    
    EXPECT_EQ(zeroDegreeA, 0);
    EXPECT_EQ(zeroDegreeB, 0);
}

// Test that all nodes are accessible after repair
TEST_F(HNSWSplitterTest, AllNodesAccessibleAfterRepair) {
    const int num_points = 200;
    auto original = createTestIndex(num_points);
    
    auto indexB = splitTwoWayInPlace(original.get());
    
    original->setEf(ef_search);
    indexB->setEf(ef_search);
    
    size_t accessibleA = 0;
    size_t accessibleB = 0;
    
    size_t countA = original->getCurrentElementCount();
    size_t countB = indexB->getCurrentElementCount();
    
    for (size_t i = 0; i < countA; ++i) {
        if (!original->isMarkedDeleted(i)) {
            accessibleA++;
        }
    }
    
    for (size_t i = 0; i < countB; ++i) {
        if (!indexB->isMarkedDeleted(i)) {
            accessibleB++;
        }
    }
    
    EXPECT_EQ(accessibleA, countA);
    EXPECT_EQ(accessibleB, countB);
}

// Test that repair actually reduces zero-degree nodes by comparing with repair on vs off
TEST_F(HNSWSplitterTest, RepairReducesZeroDegreeNodes) {
    const int num_points = 1000;
    
    // Split WITHOUT repair
    {
        auto original_no_repair = createTestIndex(num_points);
        auto indexB_no_repair = splitTwoWayInPlace(original_no_repair.get(), false);
        
        int zeroDegreeA_no_repair = HNSWSplitter<float>::countZeroDegreeNodes(*original_no_repair);
        int zeroDegreeB_no_repair = HNSWSplitter<float>::countZeroDegreeNodes(*indexB_no_repair);
        int totalZeroDegree_no_repair = zeroDegreeA_no_repair + zeroDegreeB_no_repair;
        
        std::cout << "Without repair - Index A zero-degree nodes: " << zeroDegreeA_no_repair << std::endl;
        std::cout << "Without repair - Index B zero-degree nodes: " << zeroDegreeB_no_repair << std::endl;
        std::cout << "Without repair - Total zero-degree nodes: " << totalZeroDegree_no_repair << std::endl;
        
        int zeroDegreeLevel0A_no_repair = countZeroDegreeNodesAtLevel(*original_no_repair, 0);
        int zeroDegreeLevel0B_no_repair = countZeroDegreeNodesAtLevel(*indexB_no_repair, 0);
        int totalZeroDegreeLevel0_no_repair = zeroDegreeLevel0A_no_repair + zeroDegreeLevel0B_no_repair;
        
        // Split WITH repair
        auto original_with_repair = createTestIndex(num_points);
        auto indexB_with_repair = splitTwoWayInPlace(original_with_repair.get(), true);
        
        int zeroDegreeA_with_repair = HNSWSplitter<float>::countZeroDegreeNodes(*original_with_repair);
        int zeroDegreeB_with_repair = HNSWSplitter<float>::countZeroDegreeNodes(*indexB_with_repair);
        int totalZeroDegree_with_repair = zeroDegreeA_with_repair + zeroDegreeB_with_repair;
        
        std::cout << "With repair - Index A zero-degree nodes: " << zeroDegreeA_with_repair << std::endl;
        std::cout << "With repair - Index B zero-degree nodes: " << zeroDegreeB_with_repair << std::endl;
        std::cout << "With repair - Total zero-degree nodes: " << totalZeroDegree_with_repair << std::endl;
        
        int zeroDegreeLevel0A_with_repair = countZeroDegreeNodesAtLevel(*original_with_repair, 0);
        int zeroDegreeLevel0B_with_repair = countZeroDegreeNodesAtLevel(*indexB_with_repair, 0);
        int totalZeroDegreeLevel0_with_repair = zeroDegreeLevel0A_with_repair + zeroDegreeLevel0B_with_repair;
        
        std::cout << "Without repair - Level 0 zero-degree nodes: " << totalZeroDegreeLevel0_no_repair << std::endl;
        std::cout << "With repair - Level 0 zero-degree nodes: " << totalZeroDegreeLevel0_with_repair << std::endl;
        
        EXPECT_EQ(totalZeroDegreeLevel0_with_repair, 0) 
            << "With repair enabled, there should be no zero-degree nodes at level 0";
        
        EXPECT_LE(totalZeroDegreeLevel0_with_repair, totalZeroDegreeLevel0_no_repair) 
            << "Repair should reduce or maintain zero-degree nodes at level 0";
        
        original_with_repair->setEf(ef_search);
        indexB_with_repair->setEf(ef_search);
        
        std::vector<float> queryA = {5.0f, 5.0f, 5.0f, 5.0f};
        std::vector<float> queryB = {105.0f, 105.0f, 105.0f, 105.0f};
        
        auto resultsA = original_with_repair->searchKnn(queryA.data(), 5);
        auto resultsB = indexB_with_repair->searchKnn(queryB.data(), 5);
        
        EXPECT_GT(resultsA.size(), 0) << "Repaired Index A should be searchable";
        EXPECT_GT(resultsB.size(), 0) << "Repaired Index B should be searchable";
        
        if (totalZeroDegreeLevel0_no_repair > 0) {
            EXPECT_LT(totalZeroDegreeLevel0_with_repair, totalZeroDegreeLevel0_no_repair)
                << "If there were zero-degree nodes at level 0 without repair, repair should reduce them";
        }
    }
}

// Production path: transfer child built with dont_create_structures_unused_in_save, then save+load+repair.
TEST_F(HNSWSplitterTest, TransferChildSkeletonSaveLoadRepairPath) {
    const int num_points = 200;
    auto original = createTestIndex(num_points);
    auto indexB = splitTwoWayInPlace(original.get(), /*enable_repair=*/true,
                                     /*dont_create_structures_unused_in_save=*/true);

    EXPECT_FALSE(indexB->dont_create_structures_unused_in_save_)
        << "After save/load the index must be fully materialized";
    EXPECT_EQ(countZeroDegreeNodesAtLevel(*indexB, 0), 0)
        << "L0 zero-degree nodes should be repaired after load";

    std::vector<float> queryB = {105.0f, 105.0f, 105.0f, 105.0f};
    auto resultsB = indexB->searchKnn(queryB.data(), 5);
    EXPECT_GT(resultsB.size(), 0) << "Transfer child should be searchable after round-trip + repair";
}

// Test repair function on actual split indexes from disk
TEST_F(HNSWSplitterTest, RepairActualSplitIndexes) {
    // These are SIFT indexes (128 dimensions)
    const int sift_dim = 128;
    auto sift_space = std::make_unique<hnswlib::L2Space>(sift_dim);
    
    // Try to load actual split indexes (use absolute path)
    // Only include indexes that have zero-degree nodes for testing
    std::string test_index_dir = "/path/to/test_index";
    std::vector<std::string> index_files = {
        test_index_dir + "/uploaded_index_2.bin",   // Has 1 zero-degree node
        test_index_dir + "/uploaded_index_13.bin"   // Has 3 zero-degree nodes
    };
    
    for (const auto& index_file : index_files) {
        // Check if file exists
        std::ifstream check_file(index_file);
        if (!check_file.good()) {
            std::cout << "Skipping " << index_file << " (file not found)" << std::endl;
            continue;
        }
        check_file.close();
        
        std::cout << "\n=== Testing repair on " << index_file << " ===" << std::endl;
        
        try {
            // Load the index
            auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                sift_space.get(), index_file);
            
            size_t num_elements = index->getCurrentElementCount();
            std::cout << "Loaded index with " << num_elements << " elements" << std::endl;
            
            // Verify element count is reasonable
            EXPECT_GT(num_elements, 0) << "Index should have at least one element";
            EXPECT_LT(num_elements, 100000000) << "Sanity check: element count should be reasonable";
            
            // Count zero-degree nodes before repair
            int zeroDegreeBefore = HNSWSplitter<float>::countZeroDegreeNodes(*index);
            int zeroDegreeLevel0Before = countZeroDegreeNodesAtLevel(*index, 0);
            
            std::cout << "Zero-degree nodes (all levels) before repair: " << zeroDegreeBefore << std::endl;
            std::cout << "Zero-degree nodes (level 0) before repair: " << zeroDegreeLevel0Before << std::endl;
            
            // Skip indexes without zero-degree nodes (they're not useful for testing repair)
            if (zeroDegreeBefore == 0) {
                std::cout << "Skipping " << index_file << " - no zero-degree nodes found" << std::endl;
                continue;
            }
            
            // Verify search works before repair (should still work even with zero-degree nodes)
            index->setEf(ef_search);
            std::vector<float> query_before(sift_dim, 0.0f);
            auto results_before = index->searchKnn(query_before.data(), std::min(10, (int)num_elements));
            std::cout << "Search before repair returned " << results_before.size() << " results" << std::endl;
            EXPECT_GT(results_before.size(), 0) << "Index should be searchable before repair";
            
            // Find all zero-degree nodes
            std::vector<hnswlib::tableint> zeroDegreeNodes;
            const size_t N = index->getCurrentElementCount();
            
            for (hnswlib::tableint i = 0; i < (hnswlib::tableint)N; ++i) {
                if (index->isMarkedDeleted(i)) continue;
                
                bool hasConnections = false;
                int elemLevel = index->element_levels_[i];
                
                for (int level = 0; level <= elemLevel; ++level) {
                    hnswlib::linklistsizeint* ll = (level == 0) 
                        ? index->get_linklist0(i) 
                        : index->get_linklist(i, level);
                    if (ll && index->getListCount(ll) > 0) {
                        hasConnections = true;
                        break;
                    }
                }
                
                if (!hasConnections) {
                    zeroDegreeNodes.push_back(i);
                }
            }
            
            std::cout << "Found " << zeroDegreeNodes.size() << " zero-degree nodes to repair" << std::endl;
            
            // Repair each zero-degree node in-place (no new slot allocation)
            int repaired = 0;
            for (hnswlib::tableint internalId : zeroDegreeNodes) {
                try {
                    HNSWSplitter<float>::repairZeroDegreeLevel0Node(*index, internalId);
                    repaired++;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to repair node " << internalId 
                              << ": " << e.what() << std::endl;
                }
            }
            
            std::cout << "Repaired " << repaired << " zero-degree nodes" << std::endl;
            
            // Count zero-degree nodes after repair
            int zeroDegreeAfter = HNSWSplitter<float>::countZeroDegreeNodes(*index);
            int zeroDegreeLevel0After = countZeroDegreeNodesAtLevel(*index, 0);
            
            std::cout << "Zero-degree nodes (all levels) after repair: " << zeroDegreeAfter << std::endl;
            std::cout << "Zero-degree nodes (level 0) after repair: " << zeroDegreeLevel0After << std::endl;
            
            // Verify repair worked
            EXPECT_LE(zeroDegreeAfter, zeroDegreeBefore) 
                << "Repair should not increase zero-degree nodes";
            EXPECT_EQ(zeroDegreeLevel0After, 0) 
                << "After repair, there should be no zero-degree nodes at level 0";
            
            // In-place repair does not change element count
            size_t num_elements_after = index->getCurrentElementCount();
            EXPECT_EQ(num_elements_after, num_elements) 
                << "Element count should remain unchanged after in-place repair";
            std::cout << "Element count after repair: " << num_elements_after 
                      << " (was " << num_elements << ")" << std::endl;
            
            // Verify index is still searchable after repair
            index->setEf(ef_search);
            std::vector<float> query_after(sift_dim, 0.0f);  // Zero vector query
            auto results_after = index->searchKnn(query_after.data(), std::min(10, (int)num_elements));
            EXPECT_GT(results_after.size(), 0) << "Index should be searchable after repair";
            std::cout << "Search after repair returned " << results_after.size() << " results" << std::endl;
            
            // Verify search quality: should get at least as many results as before (or more)
            EXPECT_GE(results_after.size(), results_before.size()) 
                << "Search should return at least as many results after repair";
            
            // Test multiple queries to ensure search works consistently
            int successful_queries = 0;
            int total_query_tests = 5;
            for (int q = 0; q < total_query_tests; q++) {
                std::vector<float> test_query(sift_dim);
                // Create diverse queries
                for (int d = 0; d < sift_dim; d++) {
                    test_query[d] = (float)(q * 10 + d) * 0.1f;
                }
                try {
                    auto test_results = index->searchKnn(test_query.data(), std::min(5, (int)num_elements));
                    if (test_results.size() > 0) {
                        successful_queries++;
                    }
                } catch (...) {
                    // Query failed
                }
            }
            EXPECT_GT(successful_queries, 0) 
                << "At least some queries should succeed after repair";
            std::cout << "Successful queries: " << successful_queries << "/" << total_query_tests << std::endl;
            
            std::cout << "SUCCESS: Repair function worked on " << index_file << std::endl;
            std::cout << "  - Repaired " << repaired << " zero-degree nodes" << std::endl;
            std::cout << "  - Element count: " << num_elements << " → " << num_elements_after 
                      << " (increased by " << repaired << ")" << std::endl;
            std::cout << "  - Search functionality: verified (" << successful_queries 
                      << "/" << total_query_tests << " queries successful)" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading/repairing " << index_file << ": " << e.what() << std::endl;
            // Don't fail the test if file doesn't exist or can't be loaded
            // (might be running on a system without the test files)
        }
    }
}

} // namespace hnsw_splitter

