#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <memory>
#include <algorithm>
#include <cstdint>


namespace bin_utils {

// SPACEV1B is int8 (-128..127), but the uint8 path clamps to [0,255] and would
// zero every negative coordinate. Shifting by +128 maps onto [0,255] exactly;
// L2 is translation invariant, so distances and rankings are unchanged.
// Requires: queries shifted too (both readers below do), and the metric stays
// L2 — inner product and cosine would break. Ground truth stores IDs, so it
// needs no change.
inline constexpr float kInt8ToUint8Shift = 128.0f;

// Read int8 vectors from a single .bin file
// Format: [int32 count][int32 dim][count * dim int8 data]
// Converts int8 to float, shifted into the uint8 range.
inline std::vector<std::vector<float>> read_bin_vectors(const std::string& filename, int max_vectors = -1) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    int32_t count, dim;
    file.read(reinterpret_cast<char*>(&count), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (!file) throw std::runtime_error("Failed to read header from: " + filename);

    int32_t to_read = (max_vectors >= 0 && max_vectors < count) ? max_vectors : count;

    std::vector<std::vector<float>> vectors;
    vectors.reserve(to_read);

    std::vector<int8_t> buf(dim);
    for (int32_t i = 0; i < to_read; ++i) {
        file.read(reinterpret_cast<char*>(buf.data()), dim);
        if (!file) break;

        std::vector<float> vec(dim);
        for (int j = 0; j < dim; ++j) {
            vec[j] = static_cast<float>(buf[j]) + kInt8ToUint8Shift;
        }
        vectors.push_back(std::move(vec));
    }

    std::cout << "Read " << vectors.size() << " vectors (dim=" << dim << ") from " << filename
              << " (int8 +128 -> uint8)" << std::endl;
    return vectors;
}

// Read ground truth from truth.bin
// Format: [int32 count][int32 topk][count*topk int32 IDs][count*topk float32 distances]
// Returns only the IDs (distances are skipped)
inline std::vector<std::vector<int>> read_bin_groundtruth(const std::string& filename, int max_queries = -1) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    int32_t count, topk;
    file.read(reinterpret_cast<char*>(&count), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&topk), sizeof(int32_t));
    if (!file) throw std::runtime_error("Failed to read header from: " + filename);

    int32_t to_read = (max_queries >= 0 && max_queries < count) ? max_queries : count;

    std::vector<std::vector<int>> groundtruth;
    groundtruth.reserve(to_read);

    std::vector<int32_t> row(topk);
    for (int32_t i = 0; i < to_read; ++i) {
        file.read(reinterpret_cast<char*>(row.data()), topk * sizeof(int32_t));
        if (!file) break;
        groundtruth.push_back(std::vector<int>(row.begin(), row.end()));
    }

    std::cout << "Read " << groundtruth.size() << " ground truth entries (topk=" << topk
              << ") from " << filename << std::endl;
    return groundtruth;
}

// Read ground truth from truth.bin with int64 IDs
// Format: [int32 count][int32 topk][count*topk int64 IDs][count*topk float32 distances]
// Returns only the IDs (distances are skipped)
inline std::vector<std::vector<int64_t>> read_bin_groundtruth_int64(const std::string& filename, int max_queries = -1) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    int32_t count, topk;
    file.read(reinterpret_cast<char*>(&count), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&topk), sizeof(int32_t));
    if (!file) throw std::runtime_error("Failed to read header from: " + filename);

    int32_t to_read = (max_queries >= 0 && max_queries < count) ? max_queries : count;

    std::vector<std::vector<int64_t>> groundtruth;
    groundtruth.reserve(to_read);

    std::vector<int64_t> row(topk);
    for (int32_t i = 0; i < to_read; ++i) {
        file.read(reinterpret_cast<char*>(row.data()), topk * sizeof(int64_t));
        if (!file) break;
        groundtruth.push_back(std::vector<int64_t>(row.begin(), row.end()));
    }

    std::cout << "Read " << groundtruth.size() << " ground truth entries (topk=" << topk
              << ") from " << filename << std::endl;
    return groundtruth;
}

// Chunked reader for SPACEV1B base vectors. download_dataset.sh names both
// releases vectors.bin, so they differ only by layout:
//   - a single file: the BigANN release, one contiguous blob.
//   - a directory: the Microsoft release, split into vectors_1.bin, vectors_2.bin, ...
// Only the first file has an 8-byte header [int32 total_count][int32 dim];
// data then runs contiguously across any file boundaries.
class bin_chunks {
private:
    std::vector<std::string> part_files_;
    size_t current_file_idx_;
    std::unique_ptr<std::ifstream> file_;
    size_t chunk_size_;
    size_t total_read_;
    int32_t total_vectors_;
    int32_t dim_;
    bool initialized_;
    std::vector<int8_t> buf_;

    bool read_bytes(char* out, size_t nbytes) {
        size_t got = 0;
        while (got < nbytes) {
            if (!file_ || !file_->good()) {
                current_file_idx_++;
                if (current_file_idx_ >= part_files_.size()) return false;
                file_ = std::make_unique<std::ifstream>(part_files_[current_file_idx_], std::ios::binary);
                if (!file_->is_open()) return false;
            }
            file_->read(out + got, static_cast<std::streamsize>(nbytes - got));
            size_t n = static_cast<size_t>(file_->gcount());
            got += n;
            if (n == 0) {
                file_->close();
                file_.reset();
            }
        }
        return got == nbytes;
    }

    static int extract_part_number(const std::string& path) {
        auto fname = std::filesystem::path(path).stem().string();
        auto pos = fname.rfind('_');
        if (pos == std::string::npos) return 0;
        return std::stoi(fname.substr(pos + 1));
    }

public:
    bin_chunks(const std::string& vectors_path, size_t chunk_size)
        : current_file_idx_(0), chunk_size_(chunk_size), total_read_(0),
          total_vectors_(0), dim_(0), initialized_(false) {
        if (std::filesystem::is_regular_file(vectors_path)) {
            part_files_.push_back(vectors_path);
        } else {
            for (auto& entry : std::filesystem::directory_iterator(vectors_path)) {
                if (entry.is_regular_file() && entry.path().extension() == ".bin") {
                    part_files_.push_back(entry.path().string());
                }
            }
            std::sort(part_files_.begin(), part_files_.end(),
                [](const std::string& a, const std::string& b) {
                    return extract_part_number(a) < extract_part_number(b);
                });
        }

        if (part_files_.empty()) {
            throw std::runtime_error("No .bin part files found in: " + vectors_path);
        }

        // Open first file and read header
        file_ = std::make_unique<std::ifstream>(part_files_[0], std::ios::binary);
        if (!file_->is_open()) {
            throw std::runtime_error("Cannot open: " + part_files_[0]);
        }
        file_->read(reinterpret_cast<char*>(&total_vectors_), sizeof(int32_t));
        file_->read(reinterpret_cast<char*>(&dim_), sizeof(int32_t));
        if (!*file_) {
            throw std::runtime_error("Failed to read header from: " + part_files_[0]);
        }
        buf_.resize(dim_);
    }

    ~bin_chunks() {
        if (file_) file_->close();
    }

    bool done() const {
        return total_read_ >= static_cast<size_t>(total_vectors_);
    }

    int dim() const { return dim_; }
    int32_t total_vectors_count() const { return total_vectors_; }
    size_t total_count() const { return total_read_; }

    std::vector<std::vector<float>> next() {
        if (done()) return {};

        std::vector<std::vector<float>> chunk;
        chunk.reserve(chunk_size_);

        while (chunk.size() < chunk_size_ && total_read_ < static_cast<size_t>(total_vectors_)) {
            if (!read_bytes(reinterpret_cast<char*>(buf_.data()), dim_)) break;

            std::vector<float> vec(dim_);
            for (int i = 0; i < dim_; ++i) {
                vec[i] = static_cast<float>(buf_[i]) + kInt8ToUint8Shift;
            }
            chunk.push_back(std::move(vec));
            total_read_++;
        }

        if (!initialized_ && !chunk.empty()) {
            std::cout << "Reading SPACEV1B vectors: " << total_vectors_
                      << " vectors, dim=" << dim_ << ", parts=" << part_files_.size()
                      << ", chunks of " << chunk_size_ << " (int8 +128 -> uint8)" << std::endl;
            initialized_ = true;
        }

        return chunk;
    }
};

} // namespace bin_utils