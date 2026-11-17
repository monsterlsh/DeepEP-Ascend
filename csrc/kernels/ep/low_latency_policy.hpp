#pragma once

#include <deep_ep/impls/ep/combine_layout.hpp>
#include <deep_ep/impls/ep/dispatch_direct_layout.hpp>
#include <deep_ep/layout/ep/workspace.hpp>

namespace deep_ep {

constexpr int kMaxDecodeRanks = 128;
constexpr int kMaxDecodeCapacity = layout::kDecodeCapacity;

// M is the common per-rank slot capacity used to address each peer's communication buffer.
constexpr bool use_decode_implementation(int capacity) {
    return capacity >= 1 and capacity <= kMaxDecodeCapacity;
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
