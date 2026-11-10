#pragma once

#include <kernel_operator.h>

namespace deep_ep {

template <int kHidden, int kNumTopk, int kCapacity, int kNumExperts>
__global__ __vector__ void dispatch_low_latency_prepare(
    GM_ADDR x, GM_ADDR topk_idx, GM_ADDR padded_x, GM_ADDR wire_idx, GM_ADDR wire_weights, int num_tokens) {
    using namespace AscendC;
    InitSocState();
    TPipe pipe;
    TBuf<> hidden_buffer, indices_buffer, routes_buffer, weights_buffer;
    constexpr uint32_t hidden_bytes = (kHidden * 2U + 31U) / 32U * 32U;
    constexpr uint32_t indices_bytes = (kNumTopk * 8U + 31U) / 32U * 32U;
    constexpr uint32_t routes_bytes = (kNumTopk * 4U + 31U) / 32U * 32U;
    pipe.InitBuffer(hidden_buffer, hidden_bytes);
    pipe.InitBuffer(indices_buffer, indices_bytes);
    pipe.InitBuffer(routes_buffer, routes_bytes);
    pipe.InitBuffer(weights_buffer, routes_bytes);
    auto hidden = hidden_buffer.Get<uint16_t>();
    auto indices = indices_buffer.Get<int64_t>();
    auto routes = routes_buffer.Get<int32_t>();
    auto weights = weights_buffer.Get<float>();
    GlobalTensor<uint16_t> input, output;
    GlobalTensor<int64_t> source_indices;
    GlobalTensor<int32_t> output_indices;
    GlobalTensor<float> output_weights;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(padded_x));
    source_indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(topk_idx));
    output_indices.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(wire_idx));
    output_weights.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wire_weights));
    for (uint32_t token = GetBlockIdx(); token < kCapacity; token += GetBlockNum()) {
        if (token < num_tokens) {
            DataCopyPad(hidden, input[uint64_t(token) * kHidden], {1U, kHidden * 2U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            DataCopyPad(indices, source_indices[uint64_t(token) * kNumTopk],
                        {1U, kNumTopk * 8U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        } else {
            Duplicate(hidden, uint16_t(0), hidden_bytes / 2U);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
        for (uint32_t slot = 0; slot < kNumTopk; ++slot) {
            int64_t expert = token < num_tokens ? indices.GetValue(slot) : -1;
            for (uint32_t previous = 0; previous < slot && expert >= 0; ++previous)
                if (indices.GetValue(previous) == expert)
                    expert = -1;
            routes.SetValue(slot, expert >= 0 && expert < kNumExperts ? static_cast<int32_t>(expert) : -1);
            weights.SetValue(slot, 0.0f);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(output[uint64_t(token) * kHidden], hidden, {1U, kHidden * 2U, 0U, 0U, 0U});
        DataCopyPad(output_indices[uint64_t(token) * kNumTopk], routes, {1U, kNumTopk * 4U, 0U, 0U, 0U});
        DataCopyPad(output_weights[uint64_t(token) * kNumTopk], weights, {1U, kNumTopk * 4U, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
}

}  // namespace deep_ep
