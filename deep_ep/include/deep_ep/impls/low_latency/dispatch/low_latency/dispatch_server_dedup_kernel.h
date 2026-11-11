#pragma once

// Copyright (c) 2026, Lu Lu

#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_server_dedup_magic.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_server_dedup_counts.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_server_dedup_send.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_server_dedup_recv.h>

namespace DispatchDedup {

template <class XType, bool WireFp8, bool E4M3, bool Prequantized = false>
class DispatchMemoryServerDedupKernel {
public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expertScales, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tokenTypeOut,
                                GM_ADDR destinationIndexOut, GM_ADDR relayReadIndexOut, GM_ADDR sourceMaskOut, GM_ADDR expertScalesOut,
                                uint32_t bs, uint32_t h, uint32_t topK, uint32_t experts, uint32_t expertTokenNumsType, TPipe* pipe,
                                GM_ADDR inputScales = nullptr, bool enableDedup = true, GM_ADDR nativeSourceIds = nullptr) {
        const auto* args = reinterpret_cast<const __gm__ CommArgs*>(commArgs);
        core_ = GetBlockIdx();
        const uint32_t magic = RotateDispatchServerDedupMagic(args, core_, pipe);
        counts_.Init(commArgs, expertIds, expertTokenNumsOut, sendCountsOut, bs, topK, experts, expertTokenNumsType, magic, pipe);
        if (core_ < kEpServerDedupSendCoreNum)
            sender_.Init(args, x, expertIds, expertScales, sourceMaskOut, expertScalesOut, bs, h, topK, experts, pipe, magic, inputScales,
                         enableDedup);
        else
            receiver_.Init(args, expandXOut, dynamicScalesOut, tokenTypeOut, destinationIndexOut, relayReadIndexOut, bs, h, topK, experts,
                           pipe, magic, enableDedup, nativeSourceIds);
    }

    __aicore__ inline void Process() {
        counts_.SendCnt();

        if (core_ < kEpServerDedupSendCoreNum) {
            sender_.Process(counts_.ScratchPool(), counts_.ReservedBytes());
        } else {
            counts_.RecvCnt();
            counts_.CalcCumsum();
            counts_.CompleteCumsum();
            // CompleteCumsum may still be writing expert counts from its UB.
            // Retire that stage before receiver resets the pipe and reloads GM.
            PipeBarrier<PIPE_ALL>();
            receiver_.Process();
        }

        counts_.Finish();
    }

private:
    ServerDedupCounts counts_;
    ServerDedupSender<XType, WireFp8, E4M3, Prequantized> sender_;
    ServerDedupReceiver<XType, WireFp8, E4M3, Prequantized> receiver_;
    uint32_t core_ = 0;
};

}  // namespace DispatchDedup
