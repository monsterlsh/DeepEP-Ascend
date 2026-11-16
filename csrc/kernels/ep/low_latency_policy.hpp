#pragma once

#include <charconv>
#include <cstdlib>
#include <cstring>

namespace deep_ep {

constexpr int kDecodeNumExperts = 128;
constexpr int kMaxDecodeCapacity = 256;

// M is the common per-rank slot capacity used to address each peer's communication buffer.
constexpr bool is_decode_capacity(int num_ranks, int capacity) {
    return num_ranks > 1 and kDecodeNumExperts % num_ranks == 0 and
           capacity >= 1 and capacity <= kMaxDecodeCapacity;
}

constexpr bool is_decode_batch(int batch, int capacity) {
    return batch > 0 and batch <= capacity;
}

constexpr bool is_decode_geometry(int num_ranks, int capacity, int hidden, int num_topk, int num_experts) {
    return is_decode_capacity(num_ranks, capacity) and hidden == 7168 and
           num_topk == 8 and num_experts == kDecodeNumExperts;
}

constexpr bool is_decode_shape(int num_ranks, int capacity, int hidden, int num_topk, int num_experts,
                               int num_vec_cores, int num_ub_bytes, bool use_fp8) {
    return is_decode_geometry(num_ranks, capacity, hidden, num_topk, num_experts) and
           num_vec_cores == 64 and num_ub_bytes == 216 * 1024 and not use_fp8;
}

// An explicit combine limit can narrow the decode capacity range.
inline int mte_bs_threshold_override() {
    const auto value = std::getenv("EP_MTE_BS_THRESHOLD");
    if (value == nullptr or *value == '\0')
        return -1;
    const auto end = value + std::strlen(value);
    int threshold = -1;
    const auto result = std::from_chars(value, end, threshold);
    if (result.ec != std::errc{} or result.ptr != end or threshold < -1)
        return -1;
    return threshold;
}

// Diagnostic escape hatch for comparing the same public API against its original
// transport. No public constructor or dispatch/combine option is required.
inline bool low_latency_transport_enabled() {
    const auto disabled = std::getenv("EP_DISABLE_LOW_LATENCY_TRANSPORT");
    return disabled == nullptr or std::strcmp(disabled, "1") != 0;
}

}  // namespace deep_ep
