#pragma once

#include "space_l2.h"

namespace hnswlib {

template<typename dist_t>
struct SpaceFor;

template<>
struct SpaceFor<float> {
    using type = L2Space;
};

template<>
struct SpaceFor<int> {
    using type = L2SpaceI;
};

} // namespace hnswlib
