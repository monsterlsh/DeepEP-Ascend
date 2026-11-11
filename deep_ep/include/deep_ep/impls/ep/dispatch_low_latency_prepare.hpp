#pragma once

#include <kernel_operator.h>

#include <deep_ep/common/math.hpp>

namespace deep_ep {

template <int kHidden, int kNumTopk, int kCapacity, int kNumExperts>
__global__ __vector__ void dispatch_low_latency_prepare(
    GM_ADDR x, GM_ADDR topk_idx, GM_ADDR padded_x, GM_ADDR wire_idx, GM_ADDR wire_weights, const int num_tokens) {
    using namespace AscendC;
    InitSocState();

    constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(uint16_t);
    constexpr uint32_t kNumIndexBytes = kNumTopk * sizeof(int64_t);
    constexpr uint32_t kNumRouteBytes = kNumTopk * sizeof(int32_t);
    constexpr uint32_t kNumHiddenBufferBytes = math::align<uint32_t>(kNumHiddenBytes, kNumUbAlignmentBytes);
    constexpr uint32_t kNumHiddenBufferElems = kNumHiddenBufferBytes / sizeof(uint16_t);
    constexpr uint32_t kNumRouteBufferBytes = math::align<uint32_t>(kNumRouteBytes, kNumUbAlignmentBytes);
    TPipe pipe;
    TBuf<> hidden_buffer, indices_buffer, routes_buffer, weights_buffer;
    pipe.InitBuffer(hidden_buffer, kNumHiddenBufferBytes);
    pipe.InitBuffer(indices_buffer, math::align<uint32_t>(kNumIndexBytes, kNumUbAlignmentBytes));
    pipe.InitBuffer(routes_buffer, kNumRouteBufferBytes);
    pipe.InitBuffer(weights_buffer, kNumRouteBufferBytes);
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

    for (uint32_t token_idx = GetBlockIdx(); token_idx < kCapacity; token_idx += GetBlockNum()) {
        const auto hidden_offset = static_cast<uint64_t>(token_idx) * kHidden;
        const auto topk_offset = static_cast<uint64_t>(token_idx) * kNumTopk;
        const auto valid_token = token_idx < static_cast<uint32_t>(num_tokens);
        if (valid_token) {
            DataCopyPad(hidden, input[hidden_offset], {1U, kNumHiddenBytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            DataCopyPad(indices, source_indices[topk_offset], {1U, kNumIndexBytes, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        } else {
            Duplicate(hidden, uint16_t(0), kNumHiddenBufferElems);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
        for (uint32_t slot = 0; slot < kNumTopk; ++ slot) {
            int64_t expert = valid_token ? indices.GetValue(slot) : -1;
            for (uint32_t previous = 0; previous < slot and expert >= 0; ++ previous) {
                if (indices.GetValue(previous) == expert)
                    expert = -1;
            }
            routes.SetValue(slot, 0 <= expert and expert < kNumExperts ? static_cast<int32_t>(expert) : -1);
            weights.SetValue(slot, 0.0f);
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(output[hidden_offset], hidden, {1U, kNumHiddenBytes, 0U, 0U, 0U});
        DataCopyPad(output_indices[topk_offset], routes, {1U, kNumRouteBytes, 0U, 0U, 0U});
        DataCopyPad(output_weights[topk_offset], weights, {1U, kNumRouteBytes, 0U, 0U, 0U});
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
