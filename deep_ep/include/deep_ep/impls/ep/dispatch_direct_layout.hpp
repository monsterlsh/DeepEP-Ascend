#pragma once

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>

namespace deep_ep {

// Portable host/device bounds. The device implementation asserts these against
// its actual reduction encoding and DispatchUBLayout before instantiation.
static constexpr int kDispatchReductionMaxArrivals = 255;
static constexpr int kDispatchReductionMaxValue = (1 << 24) - 1;

constexpr __aicore__ __forceinline__ int64_t get_dispatch_routing_workspace_bytes(
    const int num_ranks, const int num_experts, const int tokens_per_core, const int num_topk) {
    const auto integer_arrays = 4LL * num_experts + 36LL * num_ranks +
                                16LL * tokens_per_core * num_topk + 4LL * tokens_per_core + 16;
    return math::align<int64_t>(math::align<int64_t>(integer_arrays, 8) + 88 + 48LL * num_ranks,
                               kNumUbAlignmentBytes);
}

constexpr __aicore__ __forceinline__ int64_t get_dispatch_token_bytes(
    const int hidden_bytes, const int sf_bytes, const int num_topk, const bool cached) {
    const auto metadata = 4LL * num_topk +
        (cached ? 0 : math::align<int64_t>(8LL * num_topk, kNumUbAlignmentBytes) + 4);
    return static_cast<int64_t>(hidden_bytes) +
        (sf_bytes > 0 ? math::align<int64_t>(sf_bytes, kNumUbAlignmentBytes) : 0) +
        math::align<int64_t>(metadata, kNumUbAlignmentBytes);
}

// Native receive slots are partitioned by source rank, not destination expert.
// Use the current fresh/cached token stride and the SHMEM peer's buffer base;
// no old low-latency packet-bank or per-rank address bias belongs here.
constexpr __aicore__ __forceinline__ uint64_t dispatch_direct_token_address(
    const uint64_t peer_base, const int source_rank, const int num_max_tokens_per_rank,
    const int dst_slot, const int64_t token_bytes) {
    return peer_base + (static_cast<uint64_t>(source_rank) * num_max_tokens_per_rank +
                        static_cast<uint64_t>(dst_slot)) * static_cast<uint64_t>(token_bytes);
}

}  // namespace deep_ep
