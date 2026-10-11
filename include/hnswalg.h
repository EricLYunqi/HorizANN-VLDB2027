#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <fstream>
#include <limits>
#include <cstdio>
#include <vector>
#include <cstddef>
#include <omp.h>
#include <rocksdb/db.h>
#include <rocksdb/write_batch.h>
#include <rocksdb/options.h>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    // Split transfer children only: skip link_list_locks_, VisitedListPool, and label_lookup_
    // in ctor (not used by saveIndex). Production always saveIndex then loadIndex on the worker
    // before search/repair/resize; do not reuse a transfer child in RAM without that round-trip.
    bool dont_create_structures_unused_in_save_{false};

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    /* 
     * Ghost edges (cross-partition edges) design: replace the nested vectors by flat CSR storage.
     * node_level_offsets: offset array for edge_offsets
     * edge_offsets: offset array for ghost edges arrays
     * 
     * Old design:
     * ghost_links_[internal_id][level] = vector of external labels + distances.
     */
    std::vector<uint32_t>  node_level_offsets_; // calculate via uint64_t, throw overflow runtime error if exceeds 0xFFFFFFFF
    std::vector<uint64_t>  edge_offsets_;
    std::vector<uint16_t>  edge_counts_; // live edges in a slot, capacity is edge_offsets_[s + 1] - edge_offsets_[s]
    std::vector<labeltype> ghost_links_label_;
    std::vector<dist_t>    ghost_links_distance_;

    static constexpr size_t GHOST_NO_SLOT = (size_t)-1;

    // Old ghost edges storage
    // not storage any more, only the shape AddGhostEdges hands insertion-time edges over in
    struct GhostLink {
        labeltype label;
        dist_t distance;
    };

    // RocksDB persistence
    bool ghost_links_in_memory_ = false;
    static constexpr uint64_t GHOST_MAGIC = 0x47484F53544C4E32ULL;

    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }

    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }

    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false,
        bool dont_create_structures_unused_in_save = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(dont_create_structures_unused_in_save ? 0 : max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        dont_create_structures_unused_in_save_ = dont_create_structures_unused_in_save;
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        if (!dont_create_structures_unused_in_save_) {
            visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));
        }

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }

    ~HierarchicalNSW() {
        clear();
    }

    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    void setEf(size_t ef) {
        ef_ = ef;
    }

    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }

    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }

    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }

    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }

    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }

    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }

    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }

    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }

    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }

    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }

    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }

    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }

    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }

    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }

    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }

    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }

    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }

    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }
        // This fixes dataPoint's connections to the graph.
        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }

    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }

    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }

    /*
     * Modified functions (except constructor)
     * 
     * clear(): add ghost edges release
     * resizeIndex(): add ghost edges resizing
     * indexFileSize(): add ghost edges counting
     * saveIndex() & loadIndex(): add ghost edges persistence support
     */

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        clearGhostLinksMemory();
        visited_list_pool_.reset(nullptr);
        dont_create_structures_unused_in_save_ = false;
    }

    void resizeIndex(size_t new_max_elements) {
        if (dont_create_structures_unused_in_save_) {
            throw std::runtime_error(
                "resizeIndex: split transfer index must be saveIndex+loadIndex before resize");
        }
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);
        // the ghost CSR is sized by cur_element_count, not max_elements_, so a resize leaves it alone

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }

        return size + ghostFileSize();
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }

        writeGhostLinks(output);

        output.close();
    }

    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        // clear() resets dont_create_structures_unused_in_save_. File load always allocates
        // link_list_locks_, VisitedListPool, and label_lookup_ below.
        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        // Use > instead of != to allow ghost link data appended after normal data
        if (input.tellg() > total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 100;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        if (input.peek() != EOF && input.good()) {
            uint64_t magic = 0;
            readBinaryPOD(input, magic);
            if (input.good()) {
                readGhostLinks(input, magic);
            }
        }


        input.close();

        return;
    }

    /*
     * Added functions
     * 
     * mergeIndex(): for rebalancing
     * repairL0ZeroDegreeParallel(): for splitting
     * ghost edges utils ...
     */

    void initGhostEdges(size_t init_node_count) {
        node_level_offsets_.clear();
        edge_offsets_.clear();
        edge_counts_.clear();
        ghost_links_label_.clear();
        ghost_links_distance_.clear();

        // populate
        node_level_offsets_.resize(init_node_count + 1, 0);
        edge_offsets_.assign(1, 0);
    }

    // slot of (nodeId, level) in edge_offsets_ / edge_counts_
    size_t ghostSlot(tableint nodeId, int level) const {
        if (static_cast<size_t>(nodeId) + 1 >= node_level_offsets_.size())
            return GHOST_NO_SLOT;

        uint32_t level_off = node_level_offsets_[nodeId];
        if (static_cast<uint32_t>(level) >= node_level_offsets_[nodeId + 1] - level_off)
            return GHOST_NO_SLOT;

        size_t slot = static_cast<size_t>(level_off) + static_cast<size_t>(level);
        if (slot >= edge_counts_.size())
            return GHOST_NO_SLOT;

        return slot;
    }

    bool isGhostEdgesEmpty(tableint nodeId) const {
        if (static_cast<size_t>(nodeId) + 1 >= node_level_offsets_.size())
            return true;

        uint32_t level_off = node_level_offsets_[nodeId];
        uint32_t level_end = node_level_offsets_[nodeId + 1];
        if (level_end > edge_counts_.size())
            level_end = static_cast<uint32_t>(edge_counts_.size());

        for (uint32_t s = level_off; s < level_end; ++s)
            if (edge_counts_[s])
                return false;

        return true;
    }

    unsigned short getGhostEdgesCount(tableint nodeId, int level) const {
        size_t slot = ghostSlot(nodeId, level);
        if (slot == GHOST_NO_SLOT)
            return 0;

        return edge_counts_[slot];
    }

    // live prefix only, the dead tail past length is never handed out
    const labeltype* getGhostEdgesLabel(tableint nodeId, int level, unsigned short &length) const {
        size_t slot = ghostSlot(nodeId, level);
        length = slot == GHOST_NO_SLOT ? 0 : edge_counts_[slot];
        if (!length)
            return nullptr;

        return ghost_links_label_.data() + edge_offsets_[slot];
    }

    const dist_t* getGhostEdgesDistance(tableint nodeId, int level, unsigned short &length) const {
        size_t slot = ghostSlot(nodeId, level);
        length = slot == GHOST_NO_SLOT ? 0 : edge_counts_[slot];
        if (!length)
            return nullptr;

        return ghost_links_distance_.data() + edge_offsets_[slot];
    }

    // Parallel L0 zero-degree repair (OpenMP). Requires a fully materialized index.
    // l0_zero_candidates: if non-null and non-empty, only those internal ids are considered
    // (still re-checked for L0 degree 0 before updatePoint). If null or empty, recalculates
    // candidates by scanning the whole index.
    void repairL0ZeroDegreeParallel(const std::vector<tableint>* l0_zero_candidates = nullptr) {
        if (dont_create_structures_unused_in_save_) {
            throw std::runtime_error(
                "repairL0ZeroDegreeParallel: split transfer index must be saveIndex+loadIndex before repair");
        }
        const size_t N = cur_element_count;
        if (N <= 1) {
            return;
        }

        std::vector<tableint> scanned;
        const std::vector<tableint>* candidates_ptr = nullptr;

        if (l0_zero_candidates != nullptr && !l0_zero_candidates->empty()) {
            candidates_ptr = l0_zero_candidates;
        } else {
            const int max_threads = std::max(1, omp_get_max_threads());
            std::vector<std::vector<tableint>> buckets(static_cast<size_t>(max_threads));

            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                #pragma omp for schedule(static)
                for (tableint i = 0; i < (tableint)N; ++i) {
                    if (isMarkedDeleted(i)) {
                        continue;
                    }
                    linklistsizeint* ll0 = get_linklist0(i);
                    if (ll0 && getListCount(ll0) == 0) {
                        buckets[static_cast<size_t>(tid)].push_back(i);
                    }
                }
            }

            size_t total = 0;
            for (const auto& b : buckets) {
                total += b.size();
            }
            if (total == 0) {
                return;
            }

            scanned.reserve(total);
            for (auto& b : buckets) {
                scanned.insert(scanned.end(), b.begin(), b.end());
            }
            candidates_ptr = &scanned;
        }

        if (candidates_ptr->empty()) {
            return;
        }

        // Thread count is the OpenMP team size (default: all cores, overridable via OMP_NUM_THREADS).
        // The "32" is schedule(dynamic) chunk size (iterations grabbed per steal), not num threads.
        #pragma omp parallel for schedule(dynamic, 32)
        for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(candidates_ptr->size()); ++j) {
            const tableint internalId = (*candidates_ptr)[static_cast<size_t>(j)];
            if (isMarkedDeleted(internalId)) {
                continue;
            }
            linklistsizeint* ll0 = get_linklist0(internalId);
            if (!ll0 || getListCount(ll0) != 0) {
                continue;
            }
            try {
                const void* dataPoint = getDataByInternalId(internalId);
                updatePoint(dataPoint, internalId, 1.0f);
            } catch (const std::exception& e) {
                HNSWERR << "repairL0ZeroDegreeParallel: id=" << internalId << " " << e.what() << std::endl;
            }
        }
    }

    // swap rather than clear, the capacity is the whole point of releasing here
    void clearGhostLinksMemory() {
        std::vector<uint32_t>().swap(node_level_offsets_);
        std::vector<uint64_t>().swap(edge_offsets_);
        std::vector<uint16_t>().swap(edge_counts_);
        std::vector<labeltype>().swap(ghost_links_label_);
        std::vector<dist_t>().swap(ghost_links_distance_);
        ghost_links_in_memory_ = false;
    }

    size_t getGhostLinkedVectorCount() const {
        if (!ghost_links_in_memory_) return 0;

        const int N = static_cast<int>(cur_element_count);
        size_t count = 0;
        #pragma omp parallel for schedule(static) reduction(+:count)
        for (int i = 0; i < N; ++i) {
            count += isGhostEdgesEmpty(i) ? 0 : 1;
        }
        return count;
    }

    // incremental CSR build, shared by readGhostLinks and loadGhostLinks.
    // records must arrive in ascending internal id order
    void ghostBuildBegin(uint64_t edge_hint) {
        ghost_links_in_memory_ = true;
        node_level_offsets_.assign(cur_element_count + 1, 0);
        edge_offsets_.assign(1, 0);
        edge_counts_.clear();
        ghost_links_label_.clear();
        ghost_links_distance_.clear();

        // the hint only sizes the arrays up front, nothing depends on it being exact
        if (edge_hint) {
            ghost_links_label_.reserve(edge_hint);
            ghost_links_distance_.reserve(edge_hint);
        }
    }

    // nodes with no record own zero slots
    void ghostBuildGap(tableint& next, tableint id) {
        while (next < id) {
            node_level_offsets_[next + 1] = node_level_offsets_[next];
            ++ next;
        }
    }

    void ghostBuildNode(tableint& next, tableint id, uint32_t num_levels) {
        const uint64_t slots = static_cast<uint64_t>(node_level_offsets_[id]) + num_levels;
        if (slots > std::numeric_limits<uint32_t>::max()) {
            std::string err_msg = "Ghost link payload overflows the uint32_t level-slot index";
            throw std::runtime_error(err_msg);
        }

        node_level_offsets_[id + 1] = static_cast<uint32_t>(slots);
        next = id + 1;
    }

    size_t ghostFileSize() const {
        if (!ghost_links_in_memory_) return 0;

        size_t size = sizeof(uint64_t);  // magic
        size += sizeof(uint64_t);        // cur_element_count
        size += sizeof(uint64_t);        // nodes that have ghost links
        size += sizeof(uint64_t);        // total live edges, so the reader can reserve

        const size_t N = cur_element_count;
        for (size_t i = 0; i < N; ++i) {
            if (isGhostEdgesEmpty(i)) continue;

            const uint32_t base = node_level_offsets_[i];
            const uint32_t num_levels = node_level_offsets_[i + 1] - base;

            size += sizeof(tableint); // internal_id
            size += sizeof(uint32_t); // num_levels
            for (uint32_t l = 0; l < num_levels; ++l) {
                size += sizeof(uint32_t); // count
                size += edge_counts_[base + l] * (sizeof(labeltype) + sizeof(dist_t));
            }
        }
        return size;
    }

    bool writeGhostLinks(std::ostream& output, size_t* active_nodes_out = nullptr) const {
        const size_t active_nodes = getGhostLinkedVectorCount();
        if (active_nodes_out != nullptr) {
            *active_nodes_out = active_nodes;
        }
        if (active_nodes == 0) return false;

        writeBinaryPOD(output, GHOST_MAGIC);

        const uint64_t node_count = cur_element_count;
        const uint64_t active_node_count = active_nodes;
        const uint64_t edge_count = getGhostLinkCount();
        writeBinaryPOD(output, node_count);
        writeBinaryPOD(output, active_node_count);
        writeBinaryPOD(output, edge_count);

        // only the live prefix of every slot is written, so a reload comes back compacted
        const size_t N = cur_element_count;
        for (size_t i = 0; i < N; ++i) {
            if (isGhostEdgesEmpty(i)) continue;

            const tableint internal_id = static_cast<tableint>(i);
            const uint32_t base = node_level_offsets_[i];
            const uint32_t num_levels = node_level_offsets_[i + 1] - base;
            writeBinaryPOD(output, internal_id);
            writeBinaryPOD(output, num_levels);

            for (uint32_t l = 0; l < num_levels; ++l) {
                const uint32_t count = edge_counts_[base + l];
                const uint64_t lo = edge_offsets_[base + l];
                writeBinaryPOD(output, count);
                for (uint32_t k = 0; k < count; ++k) {
                    writeBinaryPOD(output, ghost_links_label_[lo + k]);
                    writeBinaryPOD(output, ghost_links_distance_[lo + k]);
                }
            }
        }
        return true;
    }

    void readGhostLinks(std::istream& input, uint64_t magic) {
        clearGhostLinksMemory();

        if (magic != GHOST_MAGIC) {
            throw std::runtime_error("Unsupported ghost link payload");
        }

        uint64_t stored_node_count = 0;
        uint64_t active_nodes = 0;
        uint64_t edge_count = 0;
        readBinaryPOD(input, stored_node_count);
        readBinaryPOD(input, active_nodes);
        readBinaryPOD(input, edge_count);

        if (stored_node_count > static_cast<uint64_t>(cur_element_count)) {
            throw std::runtime_error("Ghost link payload is larger than the current index size");
        }

        if (active_nodes == 0) {
            return;
        }

        ghostBuildBegin(edge_count);

        // records are written in ascending internal id order, so one pass builds the CSR
        tableint next = 0;
        uint64_t running = 0;
        for (uint64_t node_idx = 0; node_idx < active_nodes; ++node_idx) {
            tableint internal_id = 0;
            uint32_t num_levels = 0;
            readBinaryPOD(input, internal_id);
            readBinaryPOD(input, num_levels);

            if (internal_id >= stored_node_count || internal_id >= cur_element_count || internal_id < next) {
                throw std::runtime_error("Ghost link payload is corrupted");
            }

            ghostBuildGap(next, internal_id);

            for (uint32_t level = 0; level < num_levels; ++level) {
                uint32_t count = 0;
                readBinaryPOD(input, count);
                if (count > std::numeric_limits<uint16_t>::max()) {
                    throw std::runtime_error("Ghost link payload has an oversized level");
                }

                edge_counts_.push_back(static_cast<uint16_t>(count));
                running += count;
                edge_offsets_.push_back(running);

                for (uint32_t j = 0; j < count; ++j) {
                    labeltype label;
                    dist_t distance;
                    readBinaryPOD(input, label);
                    readBinaryPOD(input, distance);
                    ghost_links_label_.push_back(label);
                    ghost_links_distance_.push_back(distance);
                }
            }

            ghostBuildNode(next, internal_id, num_levels);
        }

        ghostBuildGap(next, static_cast<tableint>(cur_element_count));
    }

    // `db` is the worker's long-lived ghost handle, shared with appendInsertionGhostLinks and
    // loadGhostLinks. RocksDB allows one writer per directory, so opening a second handle here
    // would force the caller to close that one first, which costs a full reopen on the next append.
    bool saveGhostLinks(rocksdb::DB* db, size_t* active_nodes_out = nullptr) const {
        if (!ghost_links_in_memory_) {
            if (active_nodes_out != nullptr) {
                *active_nodes_out = 0;
            }
            return false;
        }

        const size_t active_nodes = getGhostLinkedVectorCount();
        if (active_nodes_out != nullptr) {
            *active_nodes_out = active_nodes;
        }
        if (active_nodes == 0) {
            return false;
        }
        if (!db) {
            std::string err_msg = "saveGhostLinks: null ghost DB handle";
            throw std::runtime_error(err_msg);
        }

        rocksdb::WriteBatch batch;

        // Delete all existing keys so stale entries from a previous save don't persist.
        // DeleteRange is O(1) — writes a single range tombstone covering all 4-byte node keys.
        batch.DeleteRange(std::string(4, '\x00'), std::string(4, '\xFF') + '\x01');
        batch.Delete("__meta__");

        // Metadata key
        {
            const uint64_t node_count = cur_element_count;
            const uint64_t active_node_count = active_nodes;
            const uint64_t edge_count = getGhostLinkCount();
            std::string meta_val;
            meta_val.resize(sizeof(uint64_t) * 3);
            memcpy(&meta_val[0], &node_count, sizeof(uint64_t));
            memcpy(&meta_val[sizeof(uint64_t)], &active_node_count, sizeof(uint64_t));
            memcpy(&meta_val[sizeof(uint64_t) * 2], &edge_count, sizeof(uint64_t));
            batch.Put("__meta__", meta_val);
        }

        // Per-node ghost links. Only the live prefix of each slot is written, so the dead tail
        // activateGhostLinks leaves behind is dropped here and a reload comes back compacted.
        std::string val;
        const size_t N = cur_element_count;
        for (size_t i = 0; i < N; ++i) {
            // Currently the key is the internal id (this will break on deletions!)
            if (isGhostEdgesEmpty(i)) continue;

            const tableint internal_id = static_cast<tableint>(i);
            const uint32_t base = node_level_offsets_[i];
            const uint32_t num_levels = node_level_offsets_[i + 1] - base;

            size_t val_size = sizeof(uint32_t); // num_levels
            for (uint32_t l = 0; l < num_levels; ++l) {
                val_size += sizeof(uint32_t); // count
                val_size += edge_counts_[base + l] * (sizeof(labeltype) + sizeof(dist_t));
            }

            val.clear();
            val.resize(val_size); // keeps capacity across nodes, so this does not realloc per node
            char* ptr = &val[0];

            memcpy(ptr, &num_levels, sizeof(uint32_t));
            ptr += sizeof(uint32_t);

            for (uint32_t l = 0; l < num_levels; ++l) {
                const uint32_t count = edge_counts_[base + l];
                const uint64_t lo = edge_offsets_[base + l];
                memcpy(ptr, &count, sizeof(uint32_t));
                ptr += sizeof(uint32_t);
                for (uint32_t k = 0; k < count; ++k) {
                    memcpy(ptr, &ghost_links_label_[lo + k], sizeof(labeltype));
                    ptr += sizeof(labeltype);
                    memcpy(ptr, &ghost_links_distance_[lo + k], sizeof(dist_t));
                    ptr += sizeof(dist_t);
                }
            }

            // Key: 4-byte big-endian internal_id for sorted ordering
            char key_buf[4];
            key_buf[0] = static_cast<char>((internal_id >> 24) & 0xFF);
            key_buf[1] = static_cast<char>((internal_id >> 16) & 0xFF);
            key_buf[2] = static_cast<char>((internal_id >> 8) & 0xFF);
            key_buf[3] = static_cast<char>(internal_id & 0xFF);
            batch.Put(rocksdb::Slice(key_buf, 4), val);
        }

        rocksdb::Status s = db->Write(rocksdb::WriteOptions(), &batch);
        if (!s.ok()) {
            throw std::runtime_error("Failed to write ghost links to RocksDB: " + s.ToString());
        }

        return true;
    }

    // Appends insertion-time ghost links to the RocksDB ghost database without
    // clearing existing entries (safe to call alongside split-produced ghost links).
    // new_links: pairs of (external_label, level-0 GhostLink vector).
    // Returns true if at least one entry was written.
    bool appendInsertionGhostLinks(
        rocksdb::DB* db,
        const std::vector<std::pair<labeltype, std::vector<GhostLink>>>& new_links)
    {
        if (!db || new_links.empty()) return false;

        rocksdb::WriteBatch batch;
        size_t written = 0;
        uint64_t written_edges = 0;

        for (const auto& [ext_label, links] : new_links) {
            if (links.empty()) continue;

            tableint internal_id;
            {
                std::unique_lock<std::mutex> lock_table(label_lookup_lock);
                auto it = label_lookup_.find(ext_label);
                if (it == label_lookup_.end()) continue;
                internal_id = it->second;
            }

            const uint32_t num_levels =
                static_cast<uint32_t>(element_levels_[internal_id]) + 1;
            const uint32_t count = static_cast<uint32_t>(links.size());

            const size_t per_level_size =
                sizeof(uint32_t) + count * (sizeof(labeltype) + sizeof(dist_t));
            const size_t val_size = sizeof(uint32_t) + num_levels * per_level_size;

            std::string val;
            val.resize(val_size);
            char* ptr = &val[0];

            memcpy(ptr, &num_levels, sizeof(uint32_t)); ptr += sizeof(uint32_t);
            for (uint32_t lvl = 0; lvl < num_levels; ++lvl) {
                memcpy(ptr, &count, sizeof(uint32_t)); ptr += sizeof(uint32_t);
                for (const auto& lnk : links) {
                    memcpy(ptr, &lnk.label,    sizeof(labeltype)); ptr += sizeof(labeltype);
                    memcpy(ptr, &lnk.distance, sizeof(dist_t));    ptr += sizeof(dist_t);
                }
            }

            char key_buf[4];
            key_buf[0] = static_cast<char>((internal_id >> 24) & 0xFF);
            key_buf[1] = static_cast<char>((internal_id >> 16) & 0xFF);
            key_buf[2] = static_cast<char>((internal_id >> 8)  & 0xFF);
            key_buf[3] = static_cast<char>( internal_id        & 0xFF);
            batch.Put(rocksdb::Slice(key_buf, 4), val);
            ++written;
            written_edges += static_cast<uint64_t>(num_levels) * count;
        }

        if (written == 0) return false;

        // Update __meta__: accumulate active count and the edge-count hint loadGhostLinks reserves on
        {
            std::string meta_val;
            uint64_t node_count   = static_cast<uint64_t>(cur_element_count);
            uint64_t active_count = written;
            uint64_t edge_count   = written_edges;

            rocksdb::Status s = db->Get(rocksdb::ReadOptions(), "__meta__", &meta_val);
            if (s.ok() && meta_val.size() >= 2 * sizeof(uint64_t)) {
                uint64_t existing_active = 0;
                memcpy(&existing_active,
                       meta_val.data() + sizeof(uint64_t),
                       sizeof(uint64_t));
                active_count += existing_active;

                if (meta_val.size() >= 3 * sizeof(uint64_t)) {
                    uint64_t existing_edges = 0;
                    memcpy(&existing_edges,
                           meta_val.data() + 2 * sizeof(uint64_t),
                           sizeof(uint64_t));
                    edge_count += existing_edges;
                }
            }

            meta_val.resize(3 * sizeof(uint64_t));
            memcpy(&meta_val[0],                   &node_count,   sizeof(uint64_t));
            memcpy(&meta_val[sizeof(uint64_t)],     &active_count, sizeof(uint64_t));
            memcpy(&meta_val[2 * sizeof(uint64_t)], &edge_count,   sizeof(uint64_t));
            batch.Put("__meta__", meta_val);
        }

        rocksdb::Status s = db->Write(rocksdb::WriteOptions(), &batch);
        if (!s.ok()) {
            throw std::runtime_error(
                "appendInsertionGhostLinks: write failed: " + s.ToString());
        }

        return true;
    }

    // reads through the worker's long-lived write handle rather than OpenForReadOnly, so it
    // sees every append made since the last save. read-only opens snapshot the MANIFEST and
    // would miss them
    void loadGhostLinks(rocksdb::DB* db) {
        clearGhostLinksMemory();

        if (!db) {
            std::string err_msg = "loadGhostLinks: null ghost DB handle";
            throw std::runtime_error(err_msg);
        }

        // Read metadata
        std::string meta_val;
        rocksdb::Status s = db->Get(rocksdb::ReadOptions(), "__meta__", &meta_val);
        if (!s.ok() || meta_val.size() < sizeof(uint64_t) * 2) {
            throw std::runtime_error("Ghost link RocksDB metadata is missing or corrupted");
        }

        uint64_t stored_node_count = 0;
        uint64_t active_nodes = 0;
        uint64_t edge_hint = 0;
        memcpy(&stored_node_count, meta_val.data(), sizeof(uint64_t));
        memcpy(&active_nodes, meta_val.data() + sizeof(uint64_t), sizeof(uint64_t));
        if (meta_val.size() >= sizeof(uint64_t) * 3)
            memcpy(&edge_hint, meta_val.data() + sizeof(uint64_t) * 2, sizeof(uint64_t));

        if (stored_node_count > static_cast<uint64_t>(cur_element_count)) {
            throw std::runtime_error("Ghost link payload is larger than the current index size");
        }

        if (active_nodes == 0) {
            return;
        }

        ghostBuildBegin(edge_hint);

        // 4-byte keys sort by internal id, so one pass builds the CSR. __meta__ lands somewhere
        // in the middle of that order and is simply skipped
        std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(rocksdb::ReadOptions()));
        tableint next = 0;
        uint64_t running = 0;
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            rocksdb::Slice key = it->key();
            if (key == "__meta__") continue;

            if (key.size() != 4) {
                throw std::runtime_error("Ghost link RocksDB has corrupted key");
            }

            const unsigned char* kb = reinterpret_cast<const unsigned char*>(key.data());
            tableint internal_id = (static_cast<tableint>(kb[0]) << 24) |
                                   (static_cast<tableint>(kb[1]) << 16) |
                                   (static_cast<tableint>(kb[2]) << 8) |
                                   static_cast<tableint>(kb[3]);

            if (internal_id >= stored_node_count || internal_id >= cur_element_count || internal_id < next) {
                throw std::runtime_error("Ghost link RocksDB payload is corrupted");
            }

            rocksdb::Slice val = it->value();
            const char* ptr = val.data();
            const char* end = ptr + val.size();

            if (ptr + sizeof(uint32_t) > end) throw std::runtime_error("Ghost link RocksDB payload truncated");
            uint32_t num_levels = 0;
            memcpy(&num_levels, ptr, sizeof(uint32_t));
            ptr += sizeof(uint32_t);

            ghostBuildGap(next, internal_id);

            for (uint32_t level = 0; level < num_levels; ++level) {
                if (ptr + sizeof(uint32_t) > end) throw std::runtime_error("Ghost link RocksDB payload truncated");
                uint32_t count = 0;
                memcpy(&count, ptr, sizeof(uint32_t));
                ptr += sizeof(uint32_t);

                if (count > std::numeric_limits<uint16_t>::max()) {
                    throw std::runtime_error("Ghost link RocksDB payload has an oversized level");
                }

                edge_counts_.push_back(static_cast<uint16_t>(count));
                running += count;
                edge_offsets_.push_back(running);

                for (uint32_t j = 0; j < count; ++j) {
                    if (ptr + sizeof(labeltype) + sizeof(dist_t) > end) {
                        throw std::runtime_error("Ghost link RocksDB payload truncated");
                    }

                    labeltype label;
                    dist_t distance;
                    memcpy(&label, ptr, sizeof(labeltype));
                    ptr += sizeof(labeltype);
                    memcpy(&distance, ptr, sizeof(dist_t));
                    ptr += sizeof(dist_t);

                    ghost_links_label_.push_back(label);
                    ghost_links_distance_.push_back(distance);
                }
            }

            ghostBuildNode(next, internal_id, num_levels);
        }
        if (!it->status().ok()) {
            throw std::runtime_error("Error iterating ghost link RocksDB: " + it->status().ToString());
        }

        ghostBuildGap(next, static_cast<tableint>(cur_element_count));
    }

    // live edges only, the dead tail activateGhostLinks leaves behind is not counted
    size_t getGhostLinkCount() const {
        size_t count = 0;
        for (uint16_t live : edge_counts_)
            count += live;

        return count;
    }

    // Total normal (non-ghost) HNSW edge count across all levels, for every
    // resident element. Mirrors getGhostLinkCount()'s style but walks the
    // regular link lists (data_level0_memory_ / linkLists_) instead.
    size_t getNormalLinkCount() const {
        size_t count = 0;
        const int N = static_cast<int>(cur_element_count);
        #pragma omp parallel for schedule(static) reduction(+:count)
        for (int i = 0; i < N; ++i) {
            for (int level = 0; level <= element_levels_[i]; ++level) {
                linklistsizeint* ll = get_linklist_at_level(static_cast<tableint>(i), level);
                count += getListCount(ll);
            }
        }
        return count;
    }

    // Merge another index into this one. Ghost links are used to establish
    // cross-partition edges without full graph rebuild. After merge, `other`
    // should not be used.
    void mergeIndex(HierarchicalNSW<dist_t>& other) {
        if (dont_create_structures_unused_in_save_)
            throw std::runtime_error("mergeIndex not supported on split transfer skeleton index (this)");
        
        // `other` may still be a split skeleton; merge only reads other's graph memory. Production
        // loads `other` from disk first (full index); tests may merge an unsaved split child.
        const size_t N_self = cur_element_count;
        const size_t N_other = other.cur_element_count;

        if (N_other == 0) 
            return;

        const size_t N_total = N_self + N_other;

        if (maxM_ != other.maxM_ || maxM0_ != other.maxM0_ || M_ != other.M_) {
            throw std::runtime_error("Cannot merge indexes with different M parameters");
        }
        if (size_data_per_element_ != other.size_data_per_element_) {
            throw std::runtime_error("Cannot merge indexes with different element sizes");
        }
        if (!ghost_links_in_memory_ && !other.ghost_links_in_memory_) {
            throw std::runtime_error("mergeIndex: both resident and incoming index have no ghost links");
        }

        if (N_total > max_elements_)
            resizeIndex(N_total);

        for (tableint i = 0; i < (tableint)N_other; i++) {
            tableint new_id = (tableint)N_self + i;

            memcpy(data_level0_memory_ + (size_t)new_id * size_data_per_element_,
                   other.data_level0_memory_ + (size_t)i * other.size_data_per_element_,
                   size_data_per_element_);

            linklistsizeint* ll = get_linklist0(new_id);
            unsigned short sz = getListCount(ll);
            tableint* data = (tableint*)(ll + 1);
            for (unsigned short j = 0; j < sz; j++) {
                data[j] += (tableint)N_self;
            }

            int lvl = other.element_levels_[i];
            element_levels_[new_id] = lvl;
            if (lvl > 0) {
                size_t bytes = (size_t)lvl * size_links_per_element_;
                linkLists_[new_id] = (char*)malloc(bytes);
                if (!linkLists_[new_id]) throw std::bad_alloc();
                memcpy(linkLists_[new_id], other.linkLists_[i], bytes);

                for (int l = 1; l <= lvl; l++) {
                    linklistsizeint* ll_upper = get_linklist(new_id, l);
                    unsigned short sz_upper = getListCount(ll_upper);
                    tableint* data_upper = (tableint*)(ll_upper + 1);
                    for (unsigned short j = 0; j < sz_upper; j++) {
                        data_upper[j] += (tableint)N_self;
                    }
                }
            } else {
                linkLists_[new_id] = nullptr;
            }

            labeltype label = getExternalLabel(new_id);
            label_lookup_[label] = new_id;

            if (lvl > maxlevel_) {
                maxlevel_ = lvl;
                enterpoint_node_ = new_id;
            }
        }
        cur_element_count = N_total;

        if (other.ghost_links_in_memory_ || ghost_links_in_memory_) {
            ghost_links_in_memory_ = true;

            // our side may predate any ghost data, a node with no record simply owns zero slots
            if (node_level_offsets_.size() < N_self + 1) {
                uint32_t last = node_level_offsets_.empty() ? 0 : node_level_offsets_.back();
                node_level_offsets_.resize(N_self + 1, last);
            }

            const uint32_t self_slots = node_level_offsets_[N_self];
            if (edge_offsets_.size() < static_cast<size_t>(self_slots) + 1) {
                uint64_t last = edge_offsets_.empty() ? 0 : edge_offsets_.back();
                edge_offsets_.resize(self_slots + 1, last);
            } else {
                edge_offsets_.resize(self_slots + 1);
            }

            const uint64_t self_edges = edge_offsets_[self_slots];
            node_level_offsets_.resize(N_self + 1);
            edge_counts_.resize(self_slots, 0);
            ghost_links_label_.resize(self_edges);
            ghost_links_distance_.resize(self_edges);

            const bool other_csr = other.node_level_offsets_.size() >= N_other + 1;
            const uint32_t other_slots = other_csr ? other.node_level_offsets_[N_other] : 0;
            const uint64_t other_edges = other_slots ? other.edge_offsets_[other_slots] : 0;

            if (other_slots &&
                (other.edge_offsets_.size() < static_cast<size_t>(other_slots) + 1 ||
                 other.edge_counts_.size() < other_slots ||
                 other.ghost_links_label_.size() < other_edges ||
                 other.ghost_links_distance_.size() < other_edges)) {
                std::string err_msg = "[mergeIndex] incoming ghost CSR is inconsistent";
                throw std::runtime_error(err_msg);
            }

            if (static_cast<uint64_t>(self_slots) + other_slots > std::numeric_limits<uint32_t>::max()) {
                std::string err_msg = "[mergeIndex] ghost CSR level-slot count overflows uint32_t";
                throw std::runtime_error(err_msg);
            }

            // both offset arrays are already cumulative, so appending is one rebase each.
            // slack rides along and is dropped at the next split or save.
            node_level_offsets_.reserve(N_total + 1);
            for (tableint i = 0; i < (tableint)N_other; ++i)
                node_level_offsets_.push_back(self_slots + (other_csr ? other.node_level_offsets_[i + 1] : 0));

            edge_offsets_.reserve(static_cast<size_t>(self_slots) + other_slots + 1);
            for (uint32_t s = 1; s <= other_slots; ++s)
                edge_offsets_.push_back(self_edges + other.edge_offsets_[s]);

            edge_counts_.insert(edge_counts_.end(),
                                other.edge_counts_.begin(), other.edge_counts_.begin() + other_slots);
            ghost_links_label_.insert(ghost_links_label_.end(),
                                      other.ghost_links_label_.begin(), other.ghost_links_label_.begin() + other_edges);
            ghost_links_distance_.insert(ghost_links_distance_.end(),
                                         other.ghost_links_distance_.begin(), other.ghost_links_distance_.begin() + other_edges);
        }

        if (ghost_links_in_memory_) {
            activateGhostLinks();
        }

        visited_list_pool_.reset(new VisitedListPool(1, max_elements_));
    }

    // For each node, for each level: resolve ghost links whose target labels
    // exist in the current index. Combine resolved ghost neighbors with
    // existing neighbors and re-select via getNeighborsByHeuristic2.
    // Unresolved ghost links (target not in index) are kept for future merges.
    // Ghost links are bidirectional by construction (both sides of a split
    // store ghost links to each other), so no explicit reverse-edge pass needed.
    void activateGhostLinks() {
        const int N = (int)cur_element_count;

        // reused across nodes and levels so the resolve loop never allocates.
        // getNeighborsByHeuristic2 drains candidates, so it comes back empty every time.
        static thread_local std::vector<std::pair<dist_t, tableint>> new_nbs;
        static thread_local std::vector<tableint> seen;
        static thread_local std::priority_queue<std::pair<dist_t, tableint>,
                                                std::vector<std::pair<dist_t, tableint>>,
                                                CompareByFirst> candidates;

        #pragma omp parallel for schedule(dynamic, 256)
        for (int id = 0; id < N; ++id) {
            if (isGhostEdgesEmpty(id))
                continue;

            int level = element_levels_[id];
            for (int lvl = 0; lvl <= level; ++lvl) {
                size_t slot = ghostSlot(id, lvl);
                if (slot == GHOST_NO_SLOT || !edge_counts_[slot])
                    continue;

                const uint64_t begin = edge_offsets_[slot];
                const uint64_t end = begin + edge_counts_[slot];

                new_nbs.clear();

                // survivors are compacted to the front, edge_counts_ shrinks, offsets stay put
                uint64_t keep = begin;
                for (uint64_t k = begin; k < end; ++k) {
                    labeltype label = ghost_links_label_[k];
                    dist_t distance = ghost_links_distance_[k];

                    auto it = label_lookup_.find(label);
                    if (it == label_lookup_.end()) {
                        ghost_links_label_[keep] = label;
                        ghost_links_distance_[keep] = distance;
                        ++ keep;
                        continue;
                    }

                    tableint target_id = it->second;
                    // deleted or self, drop it instead of keeping it
                    if (isMarkedDeleted(target_id) || target_id == (tableint)id)
                        continue;

                    // target sits too low to carry an edge at this level, keep waiting
                    if (lvl > element_levels_[target_id]) {
                        ghost_links_label_[keep] = label;
                        ghost_links_distance_[keep] = distance;
                        ++ keep;
                        continue;
                    }

                    new_nbs.emplace_back(distance, target_id);
                }

                edge_counts_[slot] = static_cast<uint16_t>(keep - begin);

                if (new_nbs.empty())
                    continue;

                size_t Mcurmax = (lvl == 0) ? maxM0_ : maxM_;
                linklistsizeint* ll = (lvl == 0) ? get_linklist0(id) : get_linklist(id, lvl);
                unsigned short cur_sz = getListCount(ll);
                tableint* cur_data = (tableint*)(ll + 1);

                // linear scan beats a hash set here, the candidate set is at most maxM0_ + cap
                seen.clear();
                seen.push_back((tableint)id);
                auto first_seen = [&](tableint nb) {
                    if (std::find(seen.begin(), seen.end(), nb) != seen.end())
                        return false;

                    seen.push_back(nb);
                    return true;
                };

                for (auto& [d, nb] : new_nbs) {
                    if (first_seen(nb))
                        candidates.emplace(d, nb);
                }
                for (unsigned short j = 0; j < cur_sz; j++) {
                    tableint nb = cur_data[j];
                    if (!first_seen(nb)) continue;
                    dist_t d = fstdistfunc_(getDataByInternalId(id), getDataByInternalId(nb), dist_func_param_);
                    candidates.emplace(d, nb);
                }

                getNeighborsByHeuristic2(candidates, Mcurmax);

                setListCount(ll, (unsigned short)candidates.size());
                tableint* out = (tableint*)(ll + 1);
                int idx = 0;
                while (!candidates.empty()) {
                    out[idx++] = candidates.top().second;
                    candidates.pop();
                }
            }
        }

    }

};

}  // namespace hnswlib
