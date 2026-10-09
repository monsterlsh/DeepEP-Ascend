#pragma once

#include <cstdlib>
#include <cstring>

namespace deep_ep {

// Native MTE transport selection policy (simple fixed-threshold form).
//
// Per-operation batch-size thresholds, fixed at compile time. Values are the
// conservative crossing points measured in G-p4 (bias 0.40 over the nominal
// break-even batch):
//   dispatch (point-to-point, MTE advantage decays slowly with EP):
//     EP8 = 100, EP16 = 77, EP32 = 70, EP64 = 67, EP128 = 32 (historical)
//   combine (all-to-all, MTE engines saturate across machines):
//     EP8 = 60, EP16 = 17, EP128 = 32 (historical), others 0
// A threshold of 0 means "never use MTE" for that (op, EP). B <= threshold uses
// MTE, otherwise the official URMA path.

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

// EP_MTE_BS_THRESHOLD=<n> overrides the threshold for every EP and both ops.
// Returns -1 when unset. Must be identical across all ranks.
inline int mte_bs_threshold_override() {
    const auto v = std::getenv("EP_MTE_BS_THRESHOLD");
    if (v == nullptr or *v == '\0')
        return -1;
    return std::atoi(v);
}

// Host policy: hard hardware/capacity guards + fixed per-operation threshold.
inline bool prefer_low_latency_transport(bool is_combine, int arch, int num_ranks,
                                         int num_tokens, int num_max_tokens_per_rank,
                                         int hidden, int num_topk, int num_experts,
                                         int num_vec_cores, bool use_fp8,
                                         int num_ub_bytes = 190 * 1024) {
    if (arch != 3510 or use_fp8 or hidden != 7168 or num_topk != 8 or num_experts != 128 or
        num_vec_cores != 64 or num_ub_bytes < 190 * 1024 or num_tokens <= 0 or
        num_tokens > num_max_tokens_per_rank or num_max_tokens_per_rank > 256)
        return false;
    return num_tokens <= mte_threshold_bs(is_combine, num_ranks);
}

// Runtime decision entry: wraps the policy with the environment override.
// ep.hpp dispatch/combine call this.
inline bool prefer_low_latency_transport_runtime(bool is_combine, int arch, int num_ranks,
                                                 int num_tokens, int num_max_tokens_per_rank,
                                                 int hidden, int num_topk, int num_experts,
                                                 int num_vec_cores, bool use_fp8,
                                                 int num_ub_bytes = 190 * 1024) {
    const int override = mte_bs_threshold_override();
    if (override >= 0)
        return num_tokens <= override;
    return prefer_low_latency_transport(is_combine, arch, num_ranks, num_tokens,
                                        num_max_tokens_per_rank, hidden, num_topk, num_experts,
                                        num_vec_cores, use_fp8, num_ub_bytes);
}

// Diagnostic escape hatch for comparing the same public API against its original
// transport. No public constructor or dispatch/combine option is required.
inline bool low_latency_transport_enabled() {
    const auto disabled = std::getenv("EP_DISABLE_LOW_LATENCY_TRANSPORT");
    return disabled == nullptr or std::strcmp(disabled, "1") != 0;
}

}  // namespace deep_ep
