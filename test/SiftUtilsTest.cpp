#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include "include/sift_utils.h"

namespace ann {

TEST(SiftUtilsTest, SiftUtilsTest_ReadFvecsFile_Test) {
    //replace input with a path to the data
    std::string input = "../../data/sift_1M/sift_base.fvecs";  
    
    if (!std::filesystem::exists(input)) {
        GTEST_SKIP() << "SIFT fvecs file not found at " << input
                     << " — skipping test. Download data first.";
    }


    

    // Read existing .fvecs
    auto vecs = sift_utils::read_fvecs<float>(input);
    ASSERT_FALSE(vecs.empty());

    
    std::string output = "./test_copy_fvecs.fvecs";
    sift_utils::write_fvecs(output, vecs);

    // Diff input vs output
    EXPECT_TRUE(sift_utils::files_equal(input, output));
    std::filesystem::remove(output);
}

TEST(SiftUtilsTest, SiftUtilsTest_ReadIvecsFile_Test) {
    std::string input = "../../data/sift_1M/sift_groundtruth.ivecs";   // sample input file
    if (!std::filesystem::exists(input)) {
        GTEST_SKIP() << "SIFT fvecs file not found at " << input
                     << " — skipping test. Download data first.";
    }


    // Read existing .ivecs
    auto vecs = sift_utils::read_ivecs(input);
    ASSERT_FALSE(vecs.empty());

    // Write back out
    std::string output = "./test_copy_ivecs.ivecs";
    sift_utils::write_ivecs(output, vecs);

    // Diff input vs output
    EXPECT_TRUE(sift_utils::files_equal(input, output));
    std::filesystem::remove(output);
}


} // namespace ann
