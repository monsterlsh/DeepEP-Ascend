#pragma once

// Copyright (c) 2026, Lu Lu

#include <deep_ep/impls/low_latency/combine/low_latency/ep_memory_server_dedup_combine_config.h>
#include <deep_ep/impls/low_latency/combine/low_latency/ep_memory_server_dedup_buffer_layout.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h>

namespace CombineDedup {

struct Layout {
    uint64_t capacityRows = 0, requiredWindowBytes = 0;
};

inline bool BuildLayout(int64_t num_ranks, int64_t num_tokens, int64_t hidden, int64_t num_topk, int64_t num_experts,
                        Layout& out, uint32_t ub_bytes = kEpServerDedupCombineRecvUbBytes) {
    out = {};
    if (num_ranks < 8 or num_ranks > DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE or num_ranks % 8 or
        num_tokens <= 0 or num_tokens > 65536 or hidden <= 0 or hidden > kEpServerDedupCombineRecvUbBytes / 12 or
        num_topk < 1 or num_topk > 32 or num_experts < num_ranks or num_experts > 1024 or
        num_experts % num_ranks or num_topk > num_experts)
        return false;

    const auto num_local_experts = num_experts / num_ranks;
    const uint64_t capacity = num_ranks * num_tokens * (num_topk < num_local_experts ? num_topk : num_local_experts);
    const uint32_t peak = capacity < kEpServerDedupCombineMaxTokenNum ? capacity : kEpServerDedupCombineMaxTokenNum;
    if (not EpCombinePayloadLaneCount(peak, hidden) or not EpCombineM45Layout(peak, num_tokens, hidden, num_topk, ub_bytes).inputLanes)
        return false;

    const uint64_t packet_bytes = EpCombinePayloadBlockCount(hidden) * 512U;
    const uint64_t stage_stride = EpCombineAlign(hidden * 2U, 512U);
    const uint64_t bytes = COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET +
                           num_tokens * num_topk * packet_bytes + num_experts * num_tokens * stage_stride;
    if (bytes > DispatchDedup::kEpServerDedupWindowUsableBytes or
        DispatchDedup::DispatchServerDedupCountLayout(num_experts).stateBytes > COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET)
        return false;
    out.capacityRows = capacity;
    out.requiredWindowBytes = bytes;
    return true;
}

}  // namespace CombineDedup
