#pragma once

#include <deep_ep/common/reduction.hpp>
#include <deep_ep/impls/ep/combine_layout.hpp>
#include <deep_ep/impls/ep/dispatch_direct_layout.hpp>
#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>

namespace deep_ep {

// Inclusive per-rank capacities from the BF16 H=7168, K=6, E=384 benchmark.
constexpr int get_dispatch_mte_threshold(int num_ranks, bool cached_mode) {
    switch (num_ranks) {
        case 8:
        case 16: return 128;
        case 32: return 32;
        case 64: return cached_mode ? 16 : 32;
        default: return 0;
    }
}

constexpr int get_combine_mte_threshold(int num_ranks) {
    switch (num_ranks) {
        case 8: return 96;
        case 16: return 16;
        case 32: return 32;
        case 64: return 8;
        default: return 0;
    }
}

constexpr bool is_decode_rank_count(int num_ranks) {
    return get_dispatch_mte_threshold(num_ranks, false) > 0;
}

constexpr bool is_decode_geometry(int hidden, int num_topk, int num_experts, int num_vec_cores, int num_ub_bytes) {
    return hidden == 7168 and num_topk == 6 and num_experts == 384 and
           num_vec_cores == 64 and num_ub_bytes == 216 * 1024;
}

constexpr DispatchBufferStages get_dispatch_mte_stages(int num_ranks, int num_experts, int64_t tokens_per_core,
                                                       int64_t hidden_bytes, int sf_bytes, int num_topk,
                                                       int64_t ub_bytes, bool cached_mode) {
    const auto workspace_bytes = get_dispatch_routing_workspace_bytes(num_ranks, num_experts, tokens_per_core, num_topk);
    const auto metadata_bytes = layout::TokenLayout(0, sf_bytes, num_topk, not cached_mode).get_num_bytes();
    const auto hidden_stage_bytes = math::align<int64_t>(hidden_bytes, kNumUbAlignmentBytes);
    return get_dispatch_buffer_stages(ub_bytes - workspace_bytes, hidden_stage_bytes, metadata_bytes, tokens_per_core);
}

inline bool supports_dispatch_mte(int num_ranks, int capacity, int hidden, int num_topk, int num_experts,
                                  int num_vec_cores, int num_ub_bytes, bool cached_mode) {
    if (capacity <= 0 or capacity > get_dispatch_mte_threshold(num_ranks, cached_mode) or
        not is_decode_geometry(hidden, num_topk, num_experts, num_vec_cores, num_ub_bytes) or
        get_dispatch_max_expert_rows(num_ranks, capacity) > reduction::kReductionMaxLowCount)
        return false;

    const int64_t hidden_bytes = int64_t{2} * hidden;
    const auto tokens_per_core = math::ceil_div<int64_t>(capacity, num_vec_cores);
    return get_dispatch_mte_stages(num_ranks, num_experts, tokens_per_core, hidden_bytes, 0, num_topk,
                                   num_ub_bytes, cached_mode).fits();
}

constexpr bool supports_combine_mte(int num_ranks, int capacity, int hidden, int num_topk, int num_experts,
                                    int num_vec_cores, int num_ub_bytes) {
    if (capacity <= 0 or capacity > get_combine_mte_threshold(num_ranks) or
        not is_decode_geometry(hidden, num_topk, num_experts, num_vec_cores, num_ub_bytes))
        return false;
    const int64_t hidden_bytes = int64_t{2} * hidden;
    const int issue_bytes = num_ub_bytes - get_combine_fixed_bytes(num_ranks, num_topk, true);
    const auto reduce_slots = get_combine_hidden_slots(num_ub_bytes, hidden_bytes + get_combine_weight_bytes(num_topk));
    return num_topk <= get_combine_hidden_slots(issue_bytes, hidden_bytes) and
           math::min(num_ranks, num_topk) + 2 <= reduce_slots;
}

}  // namespace deep_ep
