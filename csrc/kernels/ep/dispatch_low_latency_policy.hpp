#pragma once

#include "low_latency_policy.hpp"

#include <cstdint>
#include <optional>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/impls/ep/dispatch_direct_layout.hpp>

namespace deep_ep {

struct DispatchLowLatencyPlan {
    int64_t buffer_bytes;
};

constexpr int get_dispatch_low_latency_threshold(int num_ranks, bool cached_mode, bool barrier_in_prologue) {
    if (barrier_in_prologue) {
        switch (num_ranks) {
            case 8:  return 256;
            case 16: return cached_mode ? 16 : 32;
            case 32: return cached_mode ? 32 : 64;
            default: return 0;
        }
    }
    switch (num_ranks) {
        case 8:  return cached_mode ? 8 : 16;
        case 16:
        case 32: return cached_mode ? 0 : 16;
        default: return 0;
    }
}

constexpr bool dispatch_direct_ub_fits(int64_t workspace_bytes, int64_t tokens_per_core,
                                       int64_t hidden_bytes, int64_t num_topk, int64_t ub_bytes,
                                       bool cached_mode) {
    const auto metadata_bytes = get_dispatch_token_bytes(0, 0, static_cast<int>(num_topk), cached_mode);
    const auto hidden_stage_bytes = math::align<int64_t>(hidden_bytes, kNumUbAlignmentBytes);
    const auto stages = get_dispatch_buffer_stages(
        ub_bytes - workspace_bytes, hidden_stage_bytes, metadata_bytes, tokens_per_core);
    return stages.num_metadata_stages > 0 and stages.num_hidden_stages > 0;
}

inline std::optional<DispatchLowLatencyPlan> get_dispatch_low_latency_plan(
        int arch, int num_ranks, int capacity, int hidden, int num_topk, int num_experts,
        int num_vec_cores, int num_ub_bytes, bool use_fp8) {
    if (arch != 3510 or not is_decode_shape(num_ranks, capacity, hidden, num_topk, num_experts,
                                            num_vec_cores, num_ub_bytes, use_fp8))
        return std::nullopt;

    static_assert(32 <= kNumMaxRanks and 32 <= kDispatchReductionMaxArrivals);
    static_assert(32 * 256 <= kDispatchReductionMaxValue);
    const auto total_capacity = static_cast<int64_t>(capacity) * num_ranks;
    const auto hidden_bytes = static_cast<int64_t>(hidden) * 2;
    const auto metadata_bytes = get_dispatch_token_bytes(0, 0, num_topk, false);
    const auto tokens_per_core = math::ceil_div<int64_t>(capacity, num_vec_cores);
    const auto workspace_bytes = get_dispatch_routing_workspace_bytes(
        num_ranks, num_experts, tokens_per_core, num_topk);
    if (not dispatch_direct_ub_fits(workspace_bytes, tokens_per_core, hidden_bytes, num_topk, num_ub_bytes, false) or
        not dispatch_direct_ub_fits(workspace_bytes, tokens_per_core, hidden_bytes, num_topk, num_ub_bytes, true))
        return std::nullopt;
    return DispatchLowLatencyPlan{total_capacity * (hidden_bytes + metadata_bytes)};
}

}  // namespace deep_ep
