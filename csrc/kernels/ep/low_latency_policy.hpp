#pragma once

#include <deep_ep/common/reduction.hpp>
#include <deep_ep/impls/ep/combine_layout.hpp>
#include <deep_ep/impls/ep/dispatch_direct_layout.hpp>
#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>

namespace deep_ep {

constexpr int kMaxDecodeRanks = 128;

// M is the common per-rank slot capacity used to address each peer's communication buffer.
constexpr bool use_decode_implementation(int capacity) {
    return capacity >= 1 and capacity <= layout::kDecodeCapacity;
}

constexpr bool is_decode_capacity(int num_ranks, int capacity) {
    return num_ranks > 1 and num_ranks <= kMaxDecodeRanks and math::is_power_of_2(num_ranks) and
           use_decode_implementation(capacity);
}

constexpr bool is_decode_geometry(int num_ranks, int capacity, int hidden, int num_topk, int num_experts,
                                  int num_vec_cores, int num_ub_bytes) {
    return is_decode_capacity(num_ranks, capacity) and num_vec_cores == 64 and num_ub_bytes == 216 * 1024 and
           hidden > 0 and hidden % (kNumVectorRegisterBytes / 2) == 0 and
           num_experts > 0 and num_experts <= kDispatchWorkerThreads and num_experts % num_ranks == 0 and
           num_topk > 0 and num_topk <= 32 and num_topk <= num_experts;
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
                                  int num_vec_cores, int num_ub_bytes, int num_sf_packs) {
    const bool use_fp8 = num_sf_packs > 0;
    if (num_sf_packs < 0 or num_sf_packs > num_ub_bytes / static_cast<int>(sizeof(sf_pack_t)) or
        not is_decode_geometry(num_ranks, capacity, hidden, num_topk, num_experts, num_vec_cores, num_ub_bytes) or
        (use_fp8 and hidden % kNumVectorRegisterBytes != 0))
        return false;

    static_assert(kMaxDecodeRanks <= kNumMaxRanks and kMaxDecodeRanks <= reduction::kReductionMaxHighCount);
    static_assert(get_dispatch_max_expert_rows(kMaxDecodeRanks, layout::kDecodeCapacity) <= reduction::kReductionMaxLowCount);
    const auto hidden_bytes = static_cast<int64_t>(hidden) * (use_fp8 ? 1 : 2);
    const int sf_bytes = num_sf_packs * static_cast<int>(sizeof(sf_pack_t));
    const auto tokens_per_core = math::ceil_div<int64_t>(capacity, num_vec_cores);
    return get_dispatch_mte_stages(num_ranks, num_experts, tokens_per_core, hidden_bytes, sf_bytes, num_topk, num_ub_bytes, false).fits() and
           get_dispatch_mte_stages(num_ranks, num_experts, tokens_per_core, hidden_bytes, sf_bytes, num_topk, num_ub_bytes, true).fits();
}

constexpr bool supports_combine_mte(int num_ranks, int capacity, int hidden, int num_topk, int num_experts,
                                    int num_vec_cores, int num_ub_bytes) {
    if (not is_decode_geometry(num_ranks, capacity, hidden, num_topk, num_experts, num_vec_cores, num_ub_bytes))
        return false;
    // BF16 and FP16 expert outputs use two bytes per element after any dispatch dtype.
    const int64_t hidden_bytes = int64_t{2} * hidden;
    const int issue_bytes = num_ub_bytes - get_combine_fixed_bytes(num_ranks, num_topk, true);
    const auto reduce_slots = get_combine_hidden_slots(num_ub_bytes, hidden_bytes + get_combine_weight_bytes(num_topk));
    return num_topk <= get_combine_hidden_slots(issue_bytes, hidden_bytes) and
           math::min(num_ranks, num_topk) + 2 <= reduce_slots;
}

}  // namespace deep_ep
