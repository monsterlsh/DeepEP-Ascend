#pragma once

#include <charconv>
#include <cstdlib>
#include <cstring>

namespace deep_ep {

// The measured grid uses a shared per-rank capacity, not the number of tokens on this rank.
constexpr bool is_decode_capacity(int num_ranks, int capacity) {
    return (num_ranks == 8 or num_ranks == 16 or num_ranks == 32) and
           capacity >= 1 and capacity <= 256;
}

constexpr bool is_decode_shape(int num_ranks, int capacity, int hidden, int num_topk, int num_experts,
                               int num_vec_cores, int num_ub_bytes, bool use_fp8) {
    return is_decode_capacity(num_ranks, capacity) and hidden == 7168 and num_topk == 8 and
           num_experts == 128 and num_vec_cores == 64 and num_ub_bytes == 216 * 1024 and not use_fp8;
}

// An override may narrow a measured route. A larger threshold cannot enable unmeasured traffic.
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

constexpr bool combine_shmem_capacity(int num_ranks, int capacity, int threshold_override) {
    return is_decode_capacity(num_ranks, capacity) and
           ((num_ranks == 8 and capacity <= 32) or (num_ranks == 16 and capacity <= 16)) and
           (threshold_override < 0 or capacity <= threshold_override);
}

constexpr bool combine_native_mte_rank(int num_ranks, int capacity, int rank, int tokens, int threshold_override) {
    return tokens == capacity - rank and rank >= 0 and rank < num_ranks and
           ((num_ranks == 8 and capacity == 64 and rank >= 4) or
            (num_ranks == 16 and capacity == 32 and rank == 15)) and
           (threshold_override < 0 or tokens <= threshold_override);
}

// Diagnostic escape hatch for comparing the same public API against its original
// transport. No public constructor or dispatch/combine option is required.
inline bool low_latency_transport_enabled() {
    const auto disabled = std::getenv("EP_DISABLE_LOW_LATENCY_TRANSPORT");
    return disabled == nullptr or std::strcmp(disabled, "1") != 0;
}

}  // namespace deep_ep
