#pragma once

#include <charconv>
#include <cstdlib>
#include <cstring>

namespace deep_ep {

// Per-operation batch thresholds for native MTE transport.
// Zero keeps URMA; eligible shapes must also pass the guards below.

constexpr int dispatch_mte_threshold_bs(int num_ranks) {
    switch (num_ranks) {
        case 8:   return 100;
        case 16:  return 77;
        case 32:  return 70;
        case 64:  return 67;
        case 128: return 32;
        default:  return 0;
    }
}

constexpr int combine_mte_threshold_bs(int num_ranks) {
    switch (num_ranks) {
        case 8:   return 60;
        case 16:  return 17;
        case 128: return 32;
        default:  return 0;
    }
}

inline int mte_threshold_bs(bool is_combine, int num_ranks) {
    return is_combine ? combine_mte_threshold_bs(num_ranks)
                      : dispatch_mte_threshold_bs(num_ranks);
}

// Override the BS threshold only for supported operation/EP combinations.
// Unset, empty, -1 or invalid values use the default policy; 0 disables MTE.
// All ranks must use the same value.
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

inline bool mte_workload_fits(bool is_combine, int num_ranks, int num_tokens, int capacity,
                              int hidden, int num_topk, int num_experts, int bs_threshold_override) {
    const auto threshold = mte_threshold_bs(is_combine, num_ranks);
    return threshold > 0 and hidden == 7168 and num_topk == 8 and num_experts == 128 and
           num_tokens > 0 and num_tokens <= capacity and capacity <= 256 and
           num_tokens <= (bs_threshold_override >= 0 ? bs_threshold_override : threshold);
}

// Host policy: hard hardware/capacity guards + fixed per-operation threshold.
inline bool prefer_low_latency_transport(bool is_combine, int arch, int num_ranks,
                                         int num_tokens, int num_max_tokens_per_rank,
                                         int hidden, int num_topk, int num_experts,
                                         int num_vec_cores, bool use_fp8,
                                         int num_ub_bytes = 190 * 1024,
                                         int bs_threshold_override = -1) {
    return arch == 3510 and not use_fp8 and num_vec_cores == 64 and num_ub_bytes >= 190 * 1024 and
           mte_workload_fits(is_combine, num_ranks, num_tokens, num_max_tokens_per_rank,
                             hidden, num_topk, num_experts, bs_threshold_override);
}

// Runtime decision entry: wraps the policy with the environment override.
// ep.hpp dispatch/combine call this.
inline bool prefer_low_latency_transport_runtime(bool is_combine, int arch, int num_ranks,
                                                 int num_tokens, int num_max_tokens_per_rank,
                                                 int hidden, int num_topk, int num_experts,
                                                 int num_vec_cores, bool use_fp8,
                                                 int num_ub_bytes = 190 * 1024) {
    return prefer_low_latency_transport(is_combine, arch, num_ranks, num_tokens,
                                        num_max_tokens_per_rank, hidden, num_topk, num_experts,
                                        num_vec_cores, use_fp8, num_ub_bytes, mte_bs_threshold_override());
}

// Diagnostic escape hatch for comparing the same public API against its original
// transport. No public constructor or dispatch/combine option is required.
inline bool low_latency_transport_enabled() {
    const auto disabled = std::getenv("EP_DISABLE_LOW_LATENCY_TRANSPORT");
    return disabled == nullptr or std::strcmp(disabled, "1") != 0;
}

}  // namespace deep_ep
