#pragma once

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>

namespace deep_ep {

// dispatch_impl checks these host-side bounds against the device reduction encoding.
static constexpr int kDispatchReductionMaxArrivals = 255;
static constexpr int kDispatchReductionMaxValue = (1 << 24) - 1;

constexpr __aicore__ __forceinline__ int64_t get_dispatch_routing_workspace_bytes(
    const int num_ranks, const int num_experts, const int tokens_per_core, const int num_topk) {
    // DispatchUBLayout stores four ints per route and one local copy slot per token.
    // Each rank has a histogram count, a prefix sum, one SQE base and six spare SGE entries.
    const auto num_routes = static_cast<int64_t>(tokens_per_core) * num_topk;
    const auto num_ints = num_experts + 9LL * num_ranks + 4 * num_routes + tokens_per_core + 4;
    const auto jetty_offset = math::align<int64_t>(num_ints * sizeof(int), 8);
    constexpr int64_t kNumJettyInfoBytes = 88, kNumPeerInfoBytes = 48;
    return math::align<int64_t>(jetty_offset + kNumJettyInfoBytes + num_ranks * kNumPeerInfoBytes, kNumUbAlignmentBytes);
}

constexpr __aicore__ __forceinline__ int64_t get_dispatch_token_bytes(
    const int hidden_bytes, const int sf_bytes, const int num_topk, const bool cached) {
    const int64_t weight_bytes = static_cast<int64_t>(num_topk) * sizeof(float);
    const auto index_bytes = math::align<int64_t>(static_cast<int64_t>(num_topk) * sizeof(int64_t), kNumUbAlignmentBytes);
    const auto metadata_bytes = weight_bytes + (cached ? 0 : index_bytes + static_cast<int>(sizeof(int)));
    const auto aligned_sf_bytes = sf_bytes > 0 ? math::align<int64_t>(sf_bytes, kNumUbAlignmentBytes) : 0;
    return hidden_bytes + aligned_sf_bytes + math::align<int64_t>(metadata_bytes, kNumUbAlignmentBytes);
}

struct DispatchBufferStages {
    int64_t tokens_per_metadata_stage;
    int64_t num_metadata_stages;
    int64_t num_hidden_stages;
};

constexpr __aicore__ __forceinline__ DispatchBufferStages get_dispatch_buffer_stages(
    const int64_t buffer_bytes, const int64_t hidden_stage_bytes, const int64_t metadata_token_bytes,
    const int64_t tokens_per_core) {
    // Reserve one hidden stage, allocate metadata batches, then use the remaining space for hidden stages.
    const auto available_metadata_bytes = buffer_bytes - hidden_stage_bytes;
    if (hidden_stage_bytes <= 0 or metadata_token_bytes <= 0 or available_metadata_bytes < 4 * metadata_token_bytes)
        return {};
    const int64_t tokens_per_stage = available_metadata_bytes / metadata_token_bytes >= 64 ? 64 : 4;
    const auto metadata_stage_bytes = tokens_per_stage * metadata_token_bytes;
    const auto num_metadata_stages = math::min<int64_t>(8, math::min(
        math::ceil_div(tokens_per_core, tokens_per_stage), available_metadata_bytes / metadata_stage_bytes));
    const auto num_hidden_stages = math::min<int64_t>(8,
        (buffer_bytes - num_metadata_stages * metadata_stage_bytes) / hidden_stage_bytes);
    return {tokens_per_stage, num_metadata_stages, num_hidden_stages};
}

// Each source rank owns a contiguous range of receive slots in the peer's SHMEM buffer.
constexpr __aicore__ __forceinline__ uint64_t dispatch_direct_token_address(
    const uint64_t peer_base, const int source_rank, const int num_max_tokens_per_rank,
    const int dst_slot, const int64_t token_bytes) {
    const auto slot = static_cast<uint64_t>(source_rank) * num_max_tokens_per_rank + dst_slot;
    return peer_base + slot * token_bytes;
}

}  // namespace deep_ep
