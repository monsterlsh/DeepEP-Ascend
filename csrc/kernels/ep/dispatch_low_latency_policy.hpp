#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/common/reduction.hpp>
#include <deep_ep/impls/ep/dispatch_direct_layout.hpp>
#include <deep_ep/layout/ep/token.hpp>

#include "low_latency_policy.hpp"

namespace deep_ep {

constexpr DispatchBufferStages get_dispatch_mte_stages(int num_ranks, int num_experts, int64_t tokens_per_core,
                                                       int64_t hidden_bytes, int sf_bytes, int num_topk,
                                                       int64_t ub_bytes, bool cached_mode) {
    const auto workspace_bytes = get_dispatch_routing_workspace_bytes(num_ranks, num_experts, tokens_per_core, num_topk);
    const auto metadata_bytes = layout::TokenLayout(0, sf_bytes, num_topk, not cached_mode).get_num_bytes();
    const auto hidden_stage_bytes = math::align<int64_t>(hidden_bytes, kNumUbAlignmentBytes);
    return get_dispatch_buffer_stages(ub_bytes - workspace_bytes, hidden_stage_bytes, metadata_bytes, tokens_per_core);
}

inline bool supports_dispatch_mte(
        int arch, int num_ranks, int capacity, int hidden, int num_topk, int num_experts,
        int num_vec_cores, int num_ub_bytes, int num_sf_packs) {
    const bool use_fp8 = num_sf_packs > 0;
    if (arch != 3510 or num_sf_packs < 0 or num_sf_packs > num_ub_bytes / static_cast<int>(sizeof(sf_pack_t)) or
        not is_decode_geometry(num_ranks, capacity, hidden, num_topk, num_experts, num_vec_cores, num_ub_bytes) or
        (use_fp8 and hidden % kNumVectorRegisterBytes != 0))
        return false;

    static_assert(kMaxDecodeRanks <= kNumMaxRanks and kMaxDecodeRanks <= reduction::kReductionMaxHighCount);
    static_assert(get_dispatch_max_expert_rows(kMaxDecodeRanks, kMaxDecodeCapacity) <= reduction::kReductionMaxLowCount);
    const auto hidden_bytes = static_cast<int64_t>(hidden) * (use_fp8 ? 1 : 2);
    const int sf_bytes = num_sf_packs * static_cast<int>(sizeof(sf_pack_t));
    const auto tokens_per_core = math::ceil_div<int64_t>(capacity, num_vec_cores);
    return get_dispatch_mte_stages(num_ranks, num_experts, tokens_per_core, hidden_bytes, sf_bytes, num_topk, num_ub_bytes, false).fits() and
           get_dispatch_mte_stages(num_ranks, num_experts, tokens_per_core, hidden_bytes, sf_bytes, num_topk, num_ub_bytes, true).fits();
}

}  // namespace deep_ep
