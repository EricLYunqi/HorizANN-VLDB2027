#pragma once
#include <string>

namespace ann {

inline const float* as_floats(const std::string& s) {
    return reinterpret_cast<const float*>(s.data());
}

inline void set_float_bytes(std::string* s, const float* data, int n) {
    s->assign(reinterpret_cast<const char*>(data), static_cast<size_t>(n) * sizeof(float));
}

inline void append_float_bytes(std::string* s, const float* data, int n) {
    s->append(reinterpret_cast<const char*>(data), static_cast<size_t>(n) * sizeof(float));
}

} // namespace ann
