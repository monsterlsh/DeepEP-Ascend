#pragma once

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>

namespace deep_ep {

constexpr int kCombineTokensPerTask = 8;
constexpr int kCombineStages = 2;

constexpr __aicore__ __forceinline__ int get_combine_weight_bytes(int num_topk) {
    return math::align(num_topk * static_cast<int>(sizeof(float)), kNumUbAlignmentBytes);
}

// Jetty, peer table, two rank counters and the staged per-token metadata precede the hidden ring.
// combine_impl checks the computed offset against the compiled CANN structure.
constexpr __aicore__ __forceinline__ int get_combine_fixed_bytes(int num_ranks, int num_topk, bool direct) {
    const int base_elems = math::max(32, num_topk + 3);
    const int base_bytes = base_elems * static_cast<int>(sizeof(int));
    const int metadata_bytes = direct ? math::align(base_bytes, kNumUbAlignmentBytes) +
        get_combine_weight_bytes(num_topk) : base_bytes;
    return math::align(88, kNumUbAlignmentBytes) +
           math::align(48 * num_ranks, kNumUbAlignmentBytes) +
           2 * math::align(static_cast<int>(sizeof(int)) * num_ranks, kNumUbAlignmentBytes) +
           math::align(kCombineStages * kCombineTokensPerTask * metadata_bytes, kNumUbAlignmentBytes);
}

constexpr __aicore__ __forceinline__ int get_combine_hidden_slots(int ub_bytes, int64_t row_bytes) {
    return static_cast<int>(math::min<int64_t>(32, math::max(0, ub_bytes) / row_bytes));
}

}  // namespace deep_ep
