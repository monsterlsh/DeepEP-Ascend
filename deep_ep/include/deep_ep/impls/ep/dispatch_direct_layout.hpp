#pragma once

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>

namespace deep_ep {

static constexpr int kDispatchWorkerThreads = 512;

// Top-k selects distinct experts per token, so each expert can receive one
// row per source token, and a rank can receive at most min(K, local experts).
constexpr __aicore__ __forceinline__ int64_t get_dispatch_max_expert_rows(
        int num_ranks, int capacity) {
    return static_cast<int64_t>(num_ranks) * capacity;
}

constexpr __aicore__ __forceinline__ int64_t get_dispatch_max_expanded_rows(
        int64_t max_expert_rows, int num_topk, int num_local_experts, int expert_alignment) {
    return math::align<int64_t>(max_expert_rows * math::min(num_topk, num_local_experts) +
        static_cast<int64_t>(expert_alignment - 1) * num_local_experts, expert_alignment);
}

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

struct DispatchBufferStages {
    int64_t tokens_per_metadata_stage;
    int64_t num_metadata_stages;
    int64_t num_hidden_stages;

    constexpr __aicore__ __forceinline__ bool fits() const {
        return num_metadata_stages > 0 and num_hidden_stages > 0;
    }

    constexpr __aicore__ __forceinline__ bool keeps_metadata_in_ub(int num_sf_packs, int64_t tokens_per_core) const {
        return num_sf_packs == 0 and num_metadata_stages * tokens_per_metadata_stage >= tokens_per_core;
    }
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

}  // namespace deep_ep
