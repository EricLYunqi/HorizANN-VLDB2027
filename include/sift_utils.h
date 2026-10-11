#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem> 
#include <cmath>
#include <memory>


namespace sift_utils {

template<typename T>
std::vector<std::vector<T>> read_fvecs(const std::string& filename, int max_vectors = -1) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::vector<std::vector<T>> vectors;
    int count = 0;

    while (file.good() && (max_vectors == -1 || count < max_vectors)) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!file) break;

        std::vector<T> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(T));
        if (!file) break;

        vectors.push_back(std::move(vec));
        count++;
    }

    file.close();
    std::cout << "Read " << vectors.size() << " vectors from " << filename << std::endl;
    return vectors;
}

inline std::vector<std::vector<int>> read_ivecs(const std::string& filename, int max_vectors = -1) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::vector<std::vector<int>> vectors;
    int count = 0;

    while (file.good() && (max_vectors == -1 || count < max_vectors)) {
        int num;
        file.read(reinterpret_cast<char*>(&num), sizeof(int));
        if (!file) break;

        std::vector<int> vec(num);
        file.read(reinterpret_cast<char*>(vec.data()), num * sizeof(int));
        if (!file) break;

        vectors.push_back(std::move(vec));
        count++;
    }

    file.close();
    std::cout << "Read " << vectors.size() << " vectors from " << filename << std::endl;
    return vectors;
}

inline void write_fvecs(const std::string& path, const std::vector<std::vector<float>>& vecs) { 
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("write_fvecs: cannot open " + path);
    int32_t d_ref = 0;
    for (const auto& v : vecs) {
        int32_t d = static_cast<int32_t>(v.size());
        if (d_ref == 0) d_ref = d;
        out.write(reinterpret_cast<const char*>(&d), sizeof(int32_t));
        if (d > 0) {
            out.write(reinterpret_cast<const char*>(v.data()),
                      sizeof(float) * static_cast<size_t>(d));
        }
    }
}

inline void write_ivecs(const std::string& path, const std::vector<std::vector<int>>& vecs) { 
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("write_ivecs: cannot open " + path);
    int32_t d_ref = 0;
    for (const auto& v : vecs) {
        int32_t d = static_cast<int32_t>(v.size());
        if (d_ref == 0) d_ref = d;
        out.write(reinterpret_cast<const char*>(&d), sizeof(int32_t));
        if (d > 0) {
            out.write(reinterpret_cast<const char*>(v.data()),
                      sizeof(int32_t) * static_cast<size_t>(d));
        }
    }
}


inline bool files_equal(const std::string& a, const std::string& b) {
    
    std::error_code ec;
    auto sa = std::filesystem::file_size(a, ec);
    if (ec) return false;
    auto sb = std::filesystem::file_size(b, ec);
    if (ec) return false;
    if (sa != sb) return false;

    std::ifstream f1(a, std::ios::binary);
    std::ifstream f2(b, std::ios::binary);
    if (!f1 || !f2) return false;

    std::istreambuf_iterator<char> i1(f1), e1;
    std::istreambuf_iterator<char> i2(f2), e2;
    return std::equal(i1, e1, i2, e2); 
}

inline float l2(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

// Chunked reader for .bvecs files (uint8 -> float conversion)
class bvecs_chunks {
private:
    std::unique_ptr<std::ifstream> file_;
    size_t chunk_size_;
    size_t total_count_;
    bool initialized_;
    std::vector<unsigned char> buf_;  

public:
    bvecs_chunks(const std::string& filename, size_t chunk_size)
        : chunk_size_(chunk_size), total_count_(0), initialized_(false) {
        file_ = std::make_unique<std::ifstream>(filename, std::ios::binary);
        if (!file_->is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        buf_.reserve(1024);  
    }

    ~bvecs_chunks() {
        if (file_) {
            file_->close();
        }
    }

    // Check if we've reached the end of the file
    bool done() const {
        return !file_ || !file_->good();
    }

    // Get next chunk. Returns empty vector when done.
    std::vector<std::vector<float>> next() {
        if (done()) {
            return {};
        }

        std::vector<std::vector<float>> chunk;
        chunk.reserve(chunk_size_);

        while (chunk.size() < chunk_size_ && file_->good()) {
            int dim;
            file_->read(reinterpret_cast<char*>(&dim), sizeof(int));
            if (!file_->good()) break;

            // Resize buffer only if needed (reuse existing allocation)
            if (buf_.size() < static_cast<size_t>(dim)) {
                buf_.resize(dim);
            }

            file_->read(reinterpret_cast<char*>(buf_.data()), dim * sizeof(unsigned char));
            if (!file_->good()) break;

            std::vector<float> vec(dim);
            for (int i = 0; i < dim; ++i) {
                vec[i] = static_cast<float>(buf_[i]);
            }

            chunk.push_back(std::move(vec));
            total_count_++;
        }

        if (!initialized_ && !chunk.empty()) {
            std::cout << "Reading bvecs file in chunks of " << chunk_size_ << " vectors..." << std::endl;
            initialized_ = true;
        }

        return chunk;
    }

    size_t total_count() const {
        return total_count_;
    }
};


// Chunked reader for .fvecs files (float32, self-describing per-vector dim prefix)
class fvecs_chunks {
private:
    std::unique_ptr<std::ifstream> file_;
    size_t chunk_size_;
    size_t total_count_;
    bool initialized_;

public:
    fvecs_chunks(const std::string& filename, size_t chunk_size)
        : chunk_size_(chunk_size), total_count_(0), initialized_(false) {
        file_ = std::make_unique<std::ifstream>(filename, std::ios::binary);
        if (!file_->is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
    }

    ~fvecs_chunks() {
        if (file_) file_->close();
    }

    bool done() const {
        return !file_ || !file_->good();
    }

    // Get next chunk. Returns empty vector when done.
    std::vector<std::vector<float>> next() {
        if (done()) return {};

        std::vector<std::vector<float>> chunk;
        chunk.reserve(chunk_size_);

        while (chunk.size() < chunk_size_ && file_->good()) {
            int dim;
            file_->read(reinterpret_cast<char*>(&dim), sizeof(int));
            if (!file_->good()) break;

            std::vector<float> vec(dim);
            file_->read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
            if (!file_->good()) break;

            chunk.push_back(std::move(vec));
            total_count_++;
        }

        if (!initialized_ && !chunk.empty()) {
            std::cout << "Reading fvecs file in chunks of " << chunk_size_ << " vectors..." << std::endl;
            initialized_ = true;
        }

        return chunk;
    }

    size_t total_count() const { return total_count_; }
};


} // namespace sift_utils
