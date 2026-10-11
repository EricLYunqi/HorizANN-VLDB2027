#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include "src/master/query_router.h"

namespace ann {


TEST(QueryRouterKmeansTest, ThreeClusters) {
    std::vector<std::vector<float>> expected = {
        {  0.f,   0.f},   
        {100.f, 100.f}, 
        { -80.f,  40.f}   
    };

    std::vector<std::vector<float>> data = {
        
        {0.f, 0.f}, {0.1f, 0.f}, {0.f, 0.1f}, {0.1f, 0.1f},
        
        {100.f, 100.f}, {100.1f, 100.f}, {99.9f, 100.1f},
        
        {-80.f, 40.f}, {-79.9f, 40.1f}, {-80.1f, 39.9f},
    };

    const std::string out_path = "temp_kmeans_deterministic.fvecs";
    sift_utils::write_fvecs(out_path, data);
    
    std::vector<std::string> workers = {"W0", "W1", "W2"}; // k = 3
    QueryRouter router(2, 10, 16, 200,out_path, workers);
    std::filesystem::remove(out_path);
    auto clusters = router.getAllClusters();
    ASSERT_EQ(clusters.size(), workers.size()); //we should get k = num workers
    
    const float tol = 1e-1f;
    std::set<int> used;      // indices of clusters already matched


    //Every point is represented by a centroid. (3)
    //No centroid is reused for multiple expected clusters. (since they are far apart)
    // for (const auto& exp : expected) {
    //     int match_idx = -1;
    //     float best = std::numeric_limits<float>::infinity();
    //     for (size_t j = 0; j < clusters.size(); ++j) {
    //         if (used.count(static_cast<int>(j))) continue;
    //         const auto& got = clusters[j].centroid;
    //         float d = sift_utils::l2(got, exp);
    //         if (d < best) {
    //             best = d;
    //             match_idx = static_cast<int>(j);
    //         }
    //     }
    //     ASSERT_NE(match_idx, -1) << "No unmatched centroid found for expected center.";
    //     ASSERT_LT(best, tol) << "Centroid too far from expected center.";
    //     used.insert(match_idx);
    // }

    // Router test - routing picks the right cluster 
    // for (const auto& exp : expected) {
    //     auto routed = router.routeQuery(exp.data(), exp.size(),1);
    //     ASSERT_EQ(routed.size(), 1u);
    //     // routed[0].centroid should be near the same expected center
    //     EXPECT_LT(sift_utils::l2(routed[0].centroid, exp), tol);
    // }
}

} // namespace ann
