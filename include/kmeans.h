#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>

namespace kmeans {

template<typename T>
class KMeans {
private:
    int k_;
    int max_iterations_;
    std::vector<std::vector<T>> centroids_;
    std::vector<int> assignments_;
    
public:
    KMeans(int k, int max_iterations = 100) 
        : k_(k), max_iterations_(max_iterations) {}
    
    // Fit K-means to the data
    void fit(const std::vector<std::vector<T>>& data) {
        if (data.empty() || k_ <= 0) return;
        
        int dim = data[0].size();
        int n_samples = data.size();
        
        // Initialize centroids randomly
        initialize_centroids(data);
        
        assignments_.resize(n_samples);
        
        for (int iter = 0; iter < max_iterations_; ++iter) {
            bool converged = true;
            
            // Assign points to nearest centroid
            for (int i = 0; i < n_samples; ++i) {
                int nearest_centroid = find_nearest_centroid(data[i]);
                if (assignments_[i] != nearest_centroid) {
                    assignments_[i] = nearest_centroid;
                    converged = false;
                }
            }
            
            if (converged) {
                std::cout << "K-means converged after " << iter + 1 << " iterations" << std::endl;
                break;
            }
            
            // Update centroids
            update_centroids(data);
        }
    }
    
    // Get cluster assignments
    const std::vector<int>& get_assignments() const {
        return assignments_;
    }
    
    // Get centroids
    const std::vector<std::vector<T>>& get_centroids() const {
        return centroids_;
    }
    
    // Get points belonging to a specific cluster
    std::vector<std::vector<T>> get_cluster_points(const std::vector<std::vector<T>>& data, int cluster_id) const {
        std::vector<std::vector<T>> cluster_points;
        for (size_t i = 0; i < assignments_.size(); ++i) {
            if (assignments_[i] == cluster_id) {
                cluster_points.push_back(data[i]);
            }
        }
        return cluster_points;
    }
    
private:
    void initialize_centroids(const std::vector<std::vector<T>>& data) {
        int dim = data[0].size();
        int n_samples = data.size();
        
        centroids_.clear();
        centroids_.resize(k_);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n_samples - 1);
        
        // K-means++ initialization
        std::vector<bool> used(n_samples, false);
        
        // First centroid: random point
        int first_centroid = dis(gen);
        centroids_[0] = data[first_centroid];
        used[first_centroid] = true;
        
        // Subsequent centroids: choose with probability proportional to distance^2
        for (int i = 1; i < k_; ++i) {
            std::vector<T> distances(n_samples, std::numeric_limits<T>::max());
            
            for (int j = 0; j < n_samples; ++j) {
                if (used[j]) continue;
                
                for (int c = 0; c < i; ++c) {
                    T dist = euclidean_distance_squared(data[j], centroids_[c]);
                    distances[j] = std::min(distances[j], dist);
                }
            }
            
            // Choose next centroid with probability proportional to distance^2
            T total_distance = 0;
            for (int j = 0; j < n_samples; ++j) {
                if (!used[j]) total_distance += distances[j];
            }
            
            std::uniform_real_distribution<T> dis_real(0, total_distance);
            T random_value = dis_real(gen);
            
            T cumulative = 0;
            int selected = -1;
            for (int j = 0; j < n_samples; ++j) {
                if (used[j]) continue;
                cumulative += distances[j];
                if (cumulative >= random_value) {
                    selected = j;
                    break;
                }
            }
            
            if (selected == -1) selected = n_samples - 1;
            
            centroids_[i] = data[selected];
            used[selected] = true;
        }
    }
    
    int find_nearest_centroid(const std::vector<T>& point) const {
        T min_distance = std::numeric_limits<T>::max();
        int nearest = 0;
        
        for (int i = 0; i < k_; ++i) {
            T distance = euclidean_distance_squared(point, centroids_[i]);
            if (distance < min_distance) {
                min_distance = distance;
                nearest = i;
            }
        }
        
        return nearest;
    }
    
    void update_centroids(const std::vector<std::vector<T>>& data) {
        int dim = data[0].size();
        
        // Reset centroids
        for (auto& centroid : centroids_) {
            std::fill(centroid.begin(), centroid.end(), T(0));
        }
        
        std::vector<int> counts(k_, 0);
        
        // Sum all points in each cluster
        for (size_t i = 0; i < data.size(); ++i) {
            int cluster = assignments_[i];
            counts[cluster]++;
            
            for (int j = 0; j < dim; ++j) {
                centroids_[cluster][j] += data[i][j];
            }
        }
        
        // Average to get new centroids
        for (int i = 0; i < k_; ++i) {
            if (counts[i] > 0) {
                for (int j = 0; j < dim; ++j) {
                    centroids_[i][j] /= counts[i];
                }
            }
        }
    }
    
    T euclidean_distance_squared(const std::vector<T>& a, const std::vector<T>& b) const {
        T sum = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            T diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
};

template<typename T>
class BalancedKMeans {
private:
    int k_;
    int max_iterations_;
    std::vector<std::vector<T>> centroids_;
    std::vector<int> assignments_;

public:
    BalancedKMeans(int k, int max_iterations = 100)
        : k_(k), max_iterations_(max_iterations) {}

    void fit(const std::vector<std::vector<T>>& data) {
        if (data.empty() || k_ <= 0) return;

        int n_samples = data.size();
        assignments_.resize(n_samples, -1);

        initialize_centroids(data);

        int max_cluster_size = (n_samples + k_ - 1) / k_;

        for (int iter = 0; iter < max_iterations_; ++iter) {
            std::vector<int> old_assignments = assignments_;

            balanced_assign(data, max_cluster_size);

            bool converged = (old_assignments == assignments_);
            if (converged) {
                std::cout << "Balanced K-means converged after " << iter + 1 << " iterations" << std::endl;
                break;
            }

            update_centroids(data);
        }
    }

    void fit(const T* data, int n, int dim) {
        std::vector<std::vector<T>> rows(static_cast<size_t>(n), std::vector<T>(static_cast<size_t>(dim)));
        for (int i = 0; i < n; ++i)
            rows[static_cast<size_t>(i)].assign(data + i * dim, data + i * dim + dim);
        fit(rows);
    }

    const std::vector<int>& get_assignments() const { return assignments_; }
    const std::vector<std::vector<T>>& get_centroids() const { return centroids_; }

    std::vector<int> get_cluster_sizes() const {
        std::vector<int> sizes(k_, 0);
        for (int a : assignments_) {
            if (a >= 0 && a < k_) sizes[a]++;
        }
        return sizes;
    }

private:
    void initialize_centroids(const std::vector<std::vector<T>>& data) {
        int n_samples = data.size();
        centroids_.clear();
        centroids_.resize(k_);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n_samples - 1);

        std::vector<bool> used(n_samples, false);

        int first = dis(gen);
        centroids_[0] = data[first];
        used[first] = true;

        for (int i = 1; i < k_; ++i) {
            std::vector<T> distances(n_samples, std::numeric_limits<T>::max());

            for (int j = 0; j < n_samples; ++j) {
                if (used[j]) continue;
                for (int c = 0; c < i; ++c) {
                    T dist = euclidean_distance_squared(data[j], centroids_[c]);
                    distances[j] = std::min(distances[j], dist);
                }
            }

            T total_distance = 0;
            for (int j = 0; j < n_samples; ++j) {
                if (!used[j]) total_distance += distances[j];
            }

            std::uniform_real_distribution<T> dis_real(0, total_distance);
            T random_value = dis_real(gen);

            T cumulative = 0;
            int selected = -1;
            for (int j = 0; j < n_samples; ++j) {
                if (used[j]) continue;
                cumulative += distances[j];
                if (cumulative >= random_value) {
                    selected = j;
                    break;
                }
            }
            if (selected == -1) selected = n_samples - 1;

            centroids_[i] = data[selected];
            used[selected] = true;
        }
    }

    void balanced_assign(const std::vector<std::vector<T>>& data, int max_cluster_size) {
        int n = data.size();

        struct Triple {
            T distance;
            int point_idx;
            int centroid_idx;
        };

        std::vector<Triple> triples;
        triples.reserve((size_t)n * k_);

        for (int i = 0; i < n; ++i) {
            for (int c = 0; c < k_; ++c) {
                T dist = euclidean_distance_squared(data[i], centroids_[c]);
                triples.push_back({dist, i, c});
            }
        }

        std::sort(triples.begin(), triples.end(), [](const Triple& a, const Triple& b) {
            return a.distance < b.distance;
        });

        std::vector<int> cluster_sizes(k_, 0);
        std::vector<bool> assigned(n, false);
        std::fill(assignments_.begin(), assignments_.end(), -1);

        for (const auto& t : triples) {
            if (assigned[t.point_idx]) continue;
            if (cluster_sizes[t.centroid_idx] >= max_cluster_size) continue;

            assignments_[t.point_idx] = t.centroid_idx;
            assigned[t.point_idx] = true;
            cluster_sizes[t.centroid_idx]++;
        }

        for (int i = 0; i < n; ++i) {
            if (!assigned[i]) {
                int min_c = 0;
                for (int c = 1; c < k_; ++c) {
                    if (cluster_sizes[c] < cluster_sizes[min_c]) min_c = c;
                }
                assignments_[i] = min_c;
                cluster_sizes[min_c]++;
            }
        }
    }

    void update_centroids(const std::vector<std::vector<T>>& data) {
        int dim = data[0].size();

        for (auto& centroid : centroids_) {
            std::fill(centroid.begin(), centroid.end(), T(0));
        }

        std::vector<int> counts(k_, 0);

        for (size_t i = 0; i < data.size(); ++i) {
            int cluster = assignments_[i];
            counts[cluster]++;
            for (int j = 0; j < dim; ++j) {
                centroids_[cluster][j] += data[i][j];
            }
        }

        for (int i = 0; i < k_; ++i) {
            if (counts[i] > 0) {
                for (int j = 0; j < dim; ++j) {
                    centroids_[i][j] /= counts[i];
                }
            }
        }
    }

    T euclidean_distance_squared(const std::vector<T>& a, const std::vector<T>& b) const {
        T sum = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            T diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
};

// K-means++ style: pick one new centroid given fixed existing centroids.
// Used when adding a new partition so existing centroids stay unchanged and
// only data moves to the new worker (minimal reshuffling).
// Returns new_centroid and assignments (each point -> cluster index 0..num_existing,
// where 0..num_existing-1 are existing clusters and num_existing is the new cluster).
template<typename T>
struct KMeansPlusPlusNewCentroidResult {
    std::vector<T> new_centroid;
    std::vector<int> assignments;  // assignments[i] = cluster index in [0, num_existing]
};

template<typename T>
KMeansPlusPlusNewCentroidResult<T> pick_new_centroid_kmeans_pp(
    const std::vector<std::vector<T>>& existing_centroids,
    const std::vector<std::vector<T>>& data,
    std::mt19937* rng = nullptr
) {
    KMeansPlusPlusNewCentroidResult<T> result;
    if (data.empty() || existing_centroids.empty()) return result;

    const int num_existing = static_cast<int>(existing_centroids.size());
    const int n_samples = static_cast<int>(data.size());
    const int dim = static_cast<int>(data[0].size());

    auto dist_sq = [&](const std::vector<T>& a, const std::vector<T>& b) -> T {
        T sum = 0;
        for (int d = 0; d < dim; ++d) {
            T diff = a[d] - b[d];
            sum += diff * diff;
        }
        return sum;
    };

    // Min distance^2 from each point to any existing centroid
    std::vector<T> d_squared(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        T min_d = std::numeric_limits<T>::max();
        for (int c = 0; c < num_existing; ++c) {
            T d = dist_sq(data[i], existing_centroids[c]);
            if (d < min_d) min_d = d;
        }
        d_squared[i] = min_d;
    }

    std::mt19937 local_gen(std::random_device{}());
    std::mt19937& gen = rng ? *rng : local_gen;

    // Total weight for sampling (sum of D^2). Can be 0 if every point lies on an existing centroid.
    T total = 0;
    for (int i = 0; i < n_samples; ++i) total += d_squared[i];
    if (total <= 0) {
        // Fallback: all points on existing centroids; pick uniformly at random
        std::uniform_int_distribution<int> dis(0, n_samples - 1);
        result.new_centroid = data[dis(gen)];
    } else {
        // Weighted sampling: pick one point with P(i) proportional to d_squared[i].
        // Draw r in [0, total), then walk cumulative sum until cumulative >= r.
        std::uniform_real_distribution<T> dis_real(0, total);
        T r = dis_real(gen);
        T cumulative = 0;
        int selected = n_samples - 1;
        for (int i = 0; i < n_samples; ++i) {
            cumulative += d_squared[i];
            if (cumulative >= r) { selected = i; break; }
        }
        result.new_centroid = data[selected];
    }

    // Assign each point to nearest of (existing centroids + new centroid)
    result.assignments.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        T best = std::numeric_limits<T>::max();
        int best_c = 0;
        for (int c = 0; c < num_existing; ++c) {
            T d = dist_sq(data[i], existing_centroids[c]);
            if (d < best) { best = d; best_c = c; }
        }
        T d_new = dist_sq(data[i], result.new_centroid);
        if (d_new < best) best_c = num_existing;
        result.assignments[i] = best_c;
    }

    return result;
}

//  pick a new centroid via K-means++ and then refine it
// with a few Lloyd-style iterations, where existing centroids remain fixed and
// only the new centroid is moved.
template<typename T>
KMeansPlusPlusNewCentroidResult<T> pick_and_refine_new_centroid_kmeans_pp(
    const std::vector<std::vector<T>>& existing_centroids,
    const std::vector<std::vector<T>>& data,
    int refine_iters = 3,
    T convergence_eps = static_cast<T>(1e-6),
    std::mt19937* rng = nullptr
) {
    auto result = pick_new_centroid_kmeans_pp<T>(existing_centroids, data, rng);
    if (data.empty() || existing_centroids.empty() || refine_iters <= 0) {
        return result;
    }

    const int num_existing = static_cast<int>(existing_centroids.size());
    const int n_samples = static_cast<int>(data.size());
    const int dim = static_cast<int>(data[0].size());

    auto dist_sq = [&](const std::vector<T>& a, const std::vector<T>& b) -> T {
        T sum = 0;
        for (int d = 0; d < dim; ++d) {
            T diff = a[d] - b[d];
            sum += diff * diff;
        }
        return sum;
    };

    for (int iter = 0; iter < refine_iters; ++iter) {
        // Reassign each point to the nearest of (existing centroids + current new centroid)
        for (int i = 0; i < n_samples; ++i) {
            const auto& point = data[i];
            T best_dist = std::numeric_limits<T>::max();
            int best_idx = 0;

            // Existing centroids remain fixed
            for (int c = 0; c < num_existing; ++c) {
                T d = dist_sq(point, existing_centroids[c]);
                if (d < best_dist) {
                    best_dist = d;
                    best_idx = c;
                }
            }

            // New centroid can move
            T d_new = dist_sq(point, result.new_centroid);
            if (d_new < best_dist) {
                best_idx = num_existing;
            }

            result.assignments[i] = best_idx;
        }

        // Recompute new centroid as mean of its assigned points
        std::vector<T> accum(dim, static_cast<T>(0));
        int new_count = 0;
        for (int i = 0; i < n_samples; ++i) {
            if (result.assignments[i] == num_existing) {
                const auto& point = data[i];
                for (int d = 0; d < dim; ++d) {
                    accum[d] += point[d];
                }
                ++new_count;
            }
        }

        if (new_count == 0) {
            // No points assigned to new centroid; stop refinement but keep current center.
            break;
        }

        for (int d = 0; d < dim; ++d) {
            accum[d] /= static_cast<T>(new_count);
        }

        T shift = static_cast<T>(0);
        for (int d = 0; d < dim; ++d) {
            T diff = accum[d] - result.new_centroid[d];
            shift += diff * diff;
        }

        result.new_centroid.swap(accum);

        if (shift < convergence_eps) {
            break;
        }
    }

    return result;
}

} // namespace kmeans

