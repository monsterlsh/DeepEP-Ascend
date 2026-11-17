#pragma once

#include "low_latency_policy.hpp"

#include <cstdint>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/impls/ep/dispatch_direct_layout.hpp>

namespace deep_ep {

constexpr bool dispatch_direct_ub_fits(int64_t workspace_bytes, int64_t tokens_per_core,
                                       int64_t hidden_bytes, int sf_bytes, int num_topk, int64_t ub_bytes,
                                       bool cached_mode) {
    const auto metadata_bytes = get_dispatch_token_bytes(0, sf_bytes, num_topk, cached_mode);
    const auto hidden_stage_bytes = math::align<int64_t>(hidden_bytes, kNumUbAlignmentBytes);
    const auto stages = get_dispatch_buffer_stages(
        ub_bytes - workspace_bytes, hidden_stage_bytes, metadata_bytes, tokens_per_core);
    return stages.num_metadata_stages > 0 and stages.num_hidden_stages > 0;
}

inline bool supports_dispatch_mte(
        int arch, int num_ranks, int capacity, int hidden, int num_topk, int num_experts,
        int num_vec_cores, int num_ub_bytes, int num_sf_packs) {
    const bool use_fp8 = num_sf_packs > 0;
    if (arch != 3510 or num_sf_packs < 0 or num_sf_packs > num_ub_bytes / static_cast<int>(sizeof(sf_pack_t)) or
        not is_decode_shape(num_ranks, capacity, hidden, num_topk, num_experts, num_vec_cores, num_ub_bytes, use_fp8))
        return false;

    static_assert(kMaxDecodeRanks <= kNumMaxRanks and kMaxDecodeRanks <= kDispatchReductionMaxArrivals);
    static_assert(get_dispatch_max_expert_rows(kMaxDecodeRanks, kMaxDecodeCapacity) <= kDispatchReductionMaxValue);
    const auto hidden_bytes = static_cast<int64_t>(hidden) * (use_fp8 ? 1 : 2);
    const int sf_bytes = num_sf_packs * static_cast<int>(sizeof(sf_pack_t));
    const auto tokens_per_core = math::ceil_div<int64_t>(capacity, num_vec_cores);
    const auto workspace_bytes = get_dispatch_routing_workspace_bytes(
        num_ranks, num_experts, tokens_per_core, num_topk);
    return dispatch_direct_ub_fits(workspace_bytes, tokens_per_core, hidden_bytes, sf_bytes, num_topk, num_ub_bytes, false) and
           dispatch_direct_ub_fits(workspace_bytes, tokens_per_core, hidden_bytes, sf_bytes, num_topk, num_ub_bytes, true);
}

}  // namespace deep_ep
