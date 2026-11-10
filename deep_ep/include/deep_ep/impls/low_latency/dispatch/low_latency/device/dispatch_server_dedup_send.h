#pragma once

// Copyright (c) 2026, Lu Lu


#include <kernel_operator.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_dedup_mxfp8_quant.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_data_layout.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h>
#include <deep_ep/impls/low_latency/common/ep_server_order.h>
#include <deep_ep/impls/low_latency/common/ep_packet_receive_layout.h>

namespace DispatchDedup {
using namespace AscendC;

// R02-01/R05-04 source path: first hops precede the cached DIRECT tasks on
// each core, sharing one packet sequence and ring of in-flight send buffers.
template <class XType, bool WireFp8, bool E4M3, bool Prequantized = false>
class ServerDedupSender {
    static_assert(!Prequantized || WireFp8, "prequantized input requires FP8 wire");
    static constexpr bool NeedQuantize = WireFp8 && !Prequantized;
    static constexpr EpServerDedupInputMode InputMode =
        Prequantized ? EpServerDedupInputMode::PrequantizedFp8Packs
                     : (NeedQuantize ? EpServerDedupInputMode::Quantize16 : EpServerDedupInputMode::Plain16);

public:
    __aicore__ inline void Init(const __gm__ DispatchDedup::CommArgs* args, GM_ADDR x, GM_ADDR ids, GM_ADDR weights, GM_ADDR sourceMask,
                                GM_ADDR sourceWeights, uint32_t bs, uint32_t h, uint32_t k, uint32_t experts, TPipe* pipe,
                                int64_t magic = 0, GM_ADDR inputScales = nullptr, bool enableDedup = true) {
        args_ = args;
        pipe_ = pipe;
        enableDedup_ = enableDedup;
        bs_ = bs;
        h_ = h;
        k_ = k;
        localExperts_ = experts / args->rankSize;
        const uint32_t core = GetBlockIdx();
        const uint32_t base = bs / kEpServerDedupSendCoreNum, remainder = bs % kEpServerDedupSendCoreNum;
        count_ = base + (core < remainder ? 1U : 0U);
        start_ = core * base + (core < remainder ? core : remainder);
        data_ = DispatchServerDedupDataLayout(bs, h, k, experts, InputMode);
        data_.dispatchOffset += DispatchServerDedupWindowOffset(magic);
        inputScales_.SetGlobalBuffer(inputScales);
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(x));
        ids_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ids));
        weights_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
        maskOut_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sourceMask));
        weightOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sourceWeights));
    }

    __aicore__ inline void Process(TBufPool<TPosition::VECCALC, 16>& scratch, uint32_t reservedBytes) {
        // Empty token partitions return immediately. Other senders reuse count
        // scratch, excluding reservedBytes that count DMA may still read.
        if (enableDedup_ && count_ == 0U)
            return;
        scratch_ = &scratch;
        TQue<QuePosition::VECIN, 1> input;
        TBufPool<TPosition::VECIN, 1> cleanPool;
        TBuf<> cleanInput;
        // MTE2 feeds a double-buffered input queue. Quantization reads padding
        // to a 256-element boundary, so zero that tail in both buffers first.
        const uint32_t inputBytes = NeedQuantize ? uint32_t(data_.quantCount * sizeof(XType)) : uint32_t(data_.packedPayloadBytes);
        scratch_->InitBuffer(input, 2U, inputBytes);
        usedBytes_ = reservedBytes + 2U * inputBytes;
        if constexpr (NeedQuantize) {
            // Alias the two input buffers exactly as Memory does. Quantization
            // reads the compute padding, so initialize it once before copying input.
            pipe_->InitBufPool(cleanPool, 2U * inputBytes, scratch);
            cleanPool.InitBuffer(cleanInput, 2U * inputBytes);
            Duplicate<uint8_t>(cleanInput.Get<uint8_t>(), 0U, 2U * inputBytes);
            Sync<HardEvent::V_MTE2>();
        }

        // Load all bs * k routes to compute q in original routing order.
        // Load weights only for this core's token partition.
        const uint32_t routeBytes = uint32_t(DispatchDedupAlign(uint64_t(bs_) * k_ * 4U, 256U));
        const uint32_t qBytes = uint32_t(DispatchDedupAlign(uint64_t(bs_) * k_ * 4U, 32U));
        const uint32_t scaleCount = uint32_t(data_.quantCount / 32U);
        const uint32_t quantWorkBytes =
            uint32_t(DispatchDedupAlign(DispatchDedupAlign(scaleCount, 32U) * 4U + uint64_t(scaleCount) * 2U, 32U));
        const uint32_t scratchBytes = NeedQuantize && quantWorkBytes > qBytes ? quantWorkBytes : qBytes;
        Allocate(idsBuf_, routeBytes);
        Allocate(qDstBuf_, scratchBytes);
        Allocate(qSubBuf_, scratchBytes);
        Allocate(qWorkBuf_, qBytes);
        if constexpr (NeedQuantize)
            Allocate(quantBuf_, uint32_t(data_.packedPayloadBytes));
        Allocate(weightInBuf_, uint32_t(DispatchDedupAlign(uint64_t(count_) * k_ * 4U, 32U)));
        Allocate(weightOutBuf_, uint32_t(DispatchDedupAlign(uint64_t(count_) * k_ * 4U, 32U)));
        Allocate(maskBuf_, uint32_t(DispatchDedupAlign(uint64_t(count_) * 4U, 32U)));
        Allocate(directBuf_, uint32_t(DispatchDedupAlign(uint64_t(count_) * k_ * 16U, 32U)));
        // One uint32 state per server, including EP256's 32 servers.
        const uint32_t serverBytes = uint32_t(DispatchDedupAlign(uint64_t(args_->rankSize / 8U) * 4U, 32U));
        Allocate(serverBuf_, serverBytes < 64U ? 64U : serverBytes);
        const uint32_t remoteServers = args_->rankSize / 8U - 1U;
        const uint32_t maxGroups = remoteServers < k_ / 2U ? remoteServers : k_ / 2U;
        if (maxGroups != 0U)
            Allocate(firstHopBuf_, uint32_t(DispatchDedupAlign(maxGroups * 4U, 32U)));

        // The design's safe input contract guarantees room for at least one lane.
        // Use the remaining UB for up to eight full-packet lanes. Shape validation
        // must guarantee space for at least one lane.
        const uint32_t available = (kEpServerDedupCountUbBytes - usedBytes_) / uint32_t(data_.tokenStride);
        lanes_ = available < 8U ? available : 8U;
        Allocate(outBuf_, lanes_ * uint32_t(data_.tokenStride));
        routeIds_ = idsBuf_.Get<int32_t>();
        if (count_ != 0U)
            routeWeights_ = weightInBuf_.Get<float>();
        if (count_ != 0U) {
            sourceWeights_ = weightOutBuf_.Get<float>();
            sourceMasks_ = maskBuf_.Get<uint32_t>();
            directTasks_ = directBuf_.Get<uint32_t>();
        }
        serverState_ = serverBuf_.Get<uint32_t>();
        if (maxGroups != 0U)
            firstHopMasks_ = firstHopBuf_.Get<uint32_t>();
        DataCopyPad(routeIds_, ids_, {1U, bs_ * k_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        if (count_ != 0U)
            DataCopyPad(routeWeights_, weights_[uint64_t(start_) * k_], {1U, count_ * k_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0.0f});
        Sync<HardEvent::MTE2_S>();
        // Each 512-byte block carries 32 bytes of ready flags initialized to one.
        // MTE3_V permits the first write and protects reuse after each packet send.
        uint64_t flagMask[2] = {0xffU, 0U};
        for (uint32_t lane = 0; lane < lanes_; ++lane) {
            // Initialize only the 32B flags. Padding and unused entries carry no semantics.
            Duplicate<float>(outBuf_.Get<float>()[lane * data_.tokenStride / 4U + 120U], 1.0f, flagMask, uint8_t(data_.blockCount), 1U,
                             16U);
            SetFlag<HardEvent::MTE3_V>(lane);
        }
        PipeBarrier<PIPE_V>();

        if (enableDedup_)
            SendWithDedup(input);
        else
            SendWithoutDedup(input);

        // Retire all send lanes before writing sourceMask and sourceWeights.
        // Only weights selected by sourceMask are valid. MTE3_S waits for both outputs.
        for (uint32_t lane = 0; lane < lanes_; ++lane)
            WaitFlag<HardEvent::MTE3_V>(lane);
        Sync<HardEvent::S_MTE3>();
        if (enableDedup_)
            DataCopyPad(maskOut_[start_], sourceMasks_, {1U, count_ * 4U, 0U, 0U, 0U});
        if (count_ != 0U)
            DataCopyPad(weightOut_[uint64_t(start_) * k_], enableDedup_ ? sourceWeights_ : routeWeights_,
                        {1U, count_ * k_ * 4U, 0U, 0U, 0U});
        Sync<HardEvent::MTE3_S>();
    }

    __aicore__ inline void SendWithDedup(TQue<QuePosition::VECIN, 1>& input) {
        // Group routes to one remote server into a FIRST_HOP packet. Its first
        // expert is selected by the smallest k; the tail carries the other members.
        // Send all first hops first to start remote relays early.

        for (uint32_t token = 0; token < count_; ++token) {
            uint32_t directMask = 0, firstHopCount = 0;
            Classify(token, directMask, firstHopCount);
            sourceMasks_.SetValue(token, 0U);
            for (uint32_t group = 0; group < firstHopCount; ++group) {
                const uint32_t mask = firstHopMasks_.GetValue(group);
                uint32_t firstK = 0;
                while ((mask & (uint32_t(1) << firstK)) == 0U)
                    ++firstK;
                const uint32_t index = (start_ + token) * k_ + firstK;
                const uint32_t expert = uint32_t(routeIds_.GetValue(index));
                SendPacket(input, start_ + token, firstK, expert, CalTokenSendExpertCnt(expert, index), mask);
                sourceMasks_.SetValue(token, sourceMasks_.GetValue(token) | (uint32_t(1) << firstK));
                // Mark only the emitted first-hop route with weight 1; retain original weights in the tail.
                sourceWeights_.SetValue(token * k_ + firstK, 1.0f);
            }
            // Cache DIRECT token/k/expert/q tuples. Keep q from the original route
            // order when later reordering transmission by server.
            for (uint32_t k = 0; k < k_; ++k)
                if (directMask & (uint32_t(1) << k)) {
                    const uint32_t index = (start_ + token) * k_ + k;
                    const uint32_t expert = uint32_t(routeIds_.GetValue(index));
                    const uint32_t q = CalTokenSendExpertCnt(expert, index);
                    directTasks_.SetValue(directCount_ * 4U, token);
                    directTasks_.SetValue(directCount_ * 4U + 1U, k);
                    directTasks_.SetValue(directCount_ * 4U + 2U, expert);
                    directTasks_.SetValue(directCount_ * 4U + 3U, q);
                    ++directCount_;
                }
        }

        // Send DIRECT routes in server order: alternate halves at 128+ ranks,
        // use ring order below that. Share the packet counter and lanes with first
        // hops without inserting a drain between phases.

        // Stable per-server scans retain the original q and token/k order.
        // Each core advances independently; the packet/lane sequence continues
        // across first hops and server boundaries without an added drain.
        const uint32_t serverNum = args_->rankSize / 8U;
        const uint32_t expertsPerServer = localExperts_ * 8U;
        const uint32_t ownServer = args_->rank / 8U;
        for (uint32_t step = 0; step < serverNum; ++step) {
            const uint32_t server = EpServerOrder::ServerDedupDispatchDirectOrder(serverNum, ownServer, step);
            const uint32_t firstExpert = server * expertsPerServer;
            for (uint32_t task = 0; task < directCount_; ++task) {
                const uint32_t expert = directTasks_.GetValue(task * 4U + 2U);
                if (expert < firstExpert || expert >= firstExpert + expertsPerServer)
                    continue;
                const uint32_t token = directTasks_.GetValue(task * 4U);
                const uint32_t k = directTasks_.GetValue(task * 4U + 1U);
                const uint32_t q = directTasks_.GetValue(task * 4U + 3U);
                SendPacket(input, start_ + token, k, expert, q, 0U);
                sourceMasks_.SetValue(token, sourceMasks_.GetValue(token) | (uint32_t(1) << k));
                sourceWeights_.SetValue(token * k_ + k, routeWeights_.GetValue(token * k_ + k));
            }
        }
    }
    // Without deduplication, partition the flattened original routes across cores.
    __aicore__ inline void SendWithoutDedup(TQue<QuePosition::VECIN, 1>& input) {
        // Route ownership is independent of the token partition used to copy weights.
        const uint32_t total = bs_ * k_, core = GetBlockIdx();
        const uint32_t base = total / kEpServerDedupSendCoreNum;
        const uint32_t remainder = total % kEpServerDedupSendCoreNum;
        const uint32_t begin = core * base + (core < remainder ? core : remainder);
        const uint32_t count = base + (core < remainder ? 1U : 0U);
        const uint32_t serverNum = args_->rankSize / 8U;
        if (serverNum < 16U) {
            for (uint32_t i = begin; i < begin + count; ++i) {
                const uint32_t expert = uint32_t(routeIds_.GetValue(i));
                if (expert >= localExperts_ * uint32_t(args_->rankSize))
                    continue;
                SendPacket(input, i / k_, i % k_, expert, CalTokenSendExpertCnt(expert, i), 0U);
            }
            return;
        }
        // Keep the original route ownership and q slots. Only reorder this
        // core's sends by the same 128P+ server order used by dedup DIRECT.
        const uint32_t expertsPerServer = localExperts_ * 8U;
        const uint32_t ownServer = args_->rank / 8U;
        for (uint32_t step = 0; step < serverNum; ++step) {
            const uint32_t server = EpServerOrder::ServerDedupDispatchDirectOrder(serverNum, ownServer, step);
            const uint32_t firstExpert = server * expertsPerServer;
            for (uint32_t i = begin; i < begin + count; ++i) {
                const uint32_t expert = uint32_t(routeIds_.GetValue(i));
                if (expert < firstExpert || expert >= firstExpert + expertsPerServer)
                    continue;
                SendPacket(input, i / k_, i % k_, expert, CalTokenSendExpertCnt(expert, i), 0U);
            }
        }
    }

private:
    template <HardEvent event>
    __aicore__ inline void Sync() {
        const TEventID id = pipe_->FetchEventID(event);
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }
    __aicore__ inline void Allocate(TBuf<>& buffer, uint32_t bytes) {
        // Auto keeps the dedup layout even for cores owning routes but no tokens.
        if (bytes == 0U)
            return;
        scratch_->InitBuffer(buffer, bytes);
        usedBytes_ += bytes;
    }
    // Group multiple routes to one remote server as FIRST_HOP; other routes remain DIRECT.
    __aicore__ inline void Classify(uint32_t token, uint32_t& directMask, uint32_t& groups) {
        uint32_t seen = 0, grouped = 0;
        for (uint32_t k = 0; k < k_; ++k) {
            const uint32_t expert = uint32_t(routeIds_.GetValue((start_ + token) * k_ + k));
            if (expert >= localExperts_ * uint32_t(args_->rankSize))
                continue;
            const uint32_t server = expert / localExperts_ / 8U;
            const uint32_t bit = uint32_t(1) << k, serverBit = uint32_t(1) << server;
            if (server == uint32_t(args_->rank) / 8U) {
                directMask |= bit;
            } else if ((seen & serverBit) == 0U) {
                seen |= serverBit;
                serverState_.SetValue(server, k);
                directMask |= bit;
            } else if ((grouped & serverBit) == 0U) {
                const uint32_t firstBit = uint32_t(1) << serverState_.GetValue(server);
                directMask &= ~firstBit;
                firstHopMasks_.SetValue(groups, firstBit | bit);
                serverState_.SetValue(server, groups++);
                grouped |= serverBit;
            } else {
                const uint32_t index = serverState_.GetValue(server);
                firstHopMasks_.SetValue(index, firstHopMasks_.GetValue(index) | bit);
            }
        }
    }
    // q counts preceding routes to this expert in the original flattened input;
    // the first token uses zero under the routing contract.
    __aicore__ inline uint32_t CalTokenSendExpertCnt(uint32_t expert, uint32_t index) {
        if (index < k_)
            return 0U;
        // Memory's prefix reduction over the complete original route sequence.
        auto dst = qDstBuf_.Get<int32_t>();
        auto sub = qSubBuf_.Get<int32_t>();
        Duplicate<int32_t>(dst, int32_t(expert), index);
        PipeBarrier<PIPE_V>();
        Sub(sub, routeIds_, dst, index);
        PipeBarrier<PIPE_V>();
        auto input = sub.template ReinterpretCast<float>();
        auto output = dst.template ReinterpretCast<float>();
        Abs(output, input, index);
        PipeBarrier<PIPE_V>();
        Mins(sub, dst, 1, index);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(output, input, qWorkBuf_.Get<float>(), index);
        Sync<HardEvent::V_S>();
        return index - uint32_t(dst.GetValue(0));
    }
    template <class Fp8Type>
    // Generate MXFP8 data and scales per 32 elements, reusing q-counting scratch.
    __aicore__ inline void QuantMxfp8(LocalTensor<uint8_t> output, LocalTensor<XType> input) {
        const uint32_t scaleCount = uint32_t(data_.quantCount / 32U);
        auto work = qDstBuf_.Get<float>();
        auto* src = reinterpret_cast<__ubuf__ XType*>(input.GetPhyAddr());
        auto* maxExp = reinterpret_cast<__ubuf__ uint16_t*>(work.GetPhyAddr());
        auto* halfScale = reinterpret_cast<__ubuf__ uint16_t*>(work[DispatchDedupAlign(scaleCount, 32U)].GetPhyAddr());
        auto* dst = reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr());
        auto* scale = reinterpret_cast<__ubuf__ uint16_t*>(output[data_.quantCount].GetPhyAddr());
        DispatchDedupMxfp8Quant::ComputeMaxExp(src, maxExp, uint32_t(data_.quantCount));
        DispatchDedupMxfp8Quant::ComputeScale<Fp8Type>(maxExp, scale, halfScale, scaleCount);
        DispatchDedupMxfp8Quant::ComputeFp8Data<XType, Fp8Type, RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(src, halfScale, dst,
                                                                                                             uint32_t(data_.quantCount));
    }
    // Encode the source, route type and first route; FIRST_HOP tails also carry member expert/q/weight.
    __aicore__ inline void FillTail(LocalTensor<uint8_t> payload, uint32_t token, uint32_t firstK, uint32_t mask) {
        auto header = payload[data_.headerPayloadOffset].template ReinterpretCast<uint32_t>();
        header.SetValue(0U, uint32_t(args_->rank));
        header.SetValue(1U, token);
        header.SetValue(2U, firstK);
        auto index = payload[data_.readIndexPayloadOffset].template ReinterpretCast<uint32_t>();
        header.SetValue(3U, uint32_t(mask == 0U ? EpServerDedupTokenType::Direct : EpServerDedupTokenType::FirstHop));
        // Direct route ownership differs from the weight-copy token partition.
        // Its reserved relay payload is unused; never index routeWeights_ here.
        if (!enableDedup_) {
            index.SetValue(0U, 0U);
            return;
        }
        auto weights = index.template ReinterpretCast<float>();
        token -= start_;
        weights.SetValue(1U, routeWeights_.GetValue(token * k_ + firstK));
        uint32_t entries = 0U;
        for (uint32_t k = firstK + 1U; mask != 0U && k < k_; ++k)
            if (mask & (uint32_t(1) << k)) {
                const uint32_t routeIndex = (start_ + token) * k_ + k;
                const uint32_t expert = uint32_t(routeIds_.GetValue(routeIndex));
                const uint32_t q = CalTokenSendExpertCnt(expert, routeIndex);
                ++entries;
                index.SetValue(entries * 2U, expert | (q << 16U));
                weights.SetValue(entries * 2U + 1U, routeWeights_.GetValue(token * k_ + k));
            }
        index.SetValue(0U, entries);
    }
    // Load and optionally quantize a token, append its tail, then send blocks
    // of 480 payload bytes and 32 flag bytes. Events protect lane reuse.
    __aicore__ inline void SendPacket(TQue<QuePosition::VECIN, 1>& input, uint32_t token, uint32_t firstK, uint32_t expert, uint32_t q,
                                      uint32_t mask) {
        const uint32_t tokenId = token, destination = expert / localExperts_;
        auto x = input.template AllocTensor<XType>();
        if constexpr (Prequantized) {
            // The full payload queue slot stages H bytes and 4P scale bytes.
            // Only padding is synthesized; no input is read past its row end.
            Duplicate<uint8_t>(x.template ReinterpretCast<uint8_t>(), 0U, uint32_t(data_.packedPayloadBytes));
            Sync<HardEvent::V_MTE2>();
        }
        DataCopyPad(x, x_[uint64_t(tokenId) * h_], {1U, h_ * uint32_t(sizeof(XType)), 0U, 0U, 0U}, {NeedQuantize, 0U, 0U, XType(0)});
        if constexpr (Prequantized) {
            auto scale = x.template ReinterpretCast<uint8_t>()[uint32_t(data_.quantCount)];
            DataCopyPad(scale, inputScales_[uint64_t(tokenId) * data_.scaleBytes], {1U, uint32_t(data_.scaleBytes), 0U, 0U, 0U},
                        {false, 0U, 0U, uint8_t(0)});
        }
        input.EnQue(x);
        x = input.template DeQue<XType>();
        LocalTensor<uint8_t> payload;
        if constexpr (NeedQuantize) {
            payload = quantBuf_.Get<uint8_t>();
            if constexpr (E4M3)
                QuantMxfp8<fp8_e4m3fn_t>(payload, x);
            else
                QuantMxfp8<fp8_e5m2_t>(payload, x);
            input.template FreeTensor<XType>(x);
            Sync<HardEvent::V_S>();
        } else {
            Sync<HardEvent::MTE2_S>();
            payload = x.template ReinterpretCast<uint8_t>();
        }
        FillTail(payload, token, firstK, mask);
        Sync<HardEvent::S_V>();
        const uint32_t lane = packets_ % lanes_;
        WaitFlag<HardEvent::MTE3_V>(lane);
        auto out = outBuf_.Get<uint8_t>()[lane * data_.tokenStride];
        auto outWords = out.template ReinterpretCast<int32_t>();
        auto inWords = payload.template ReinterpretCast<int32_t>();
        Copy(outWords, inWords, uint64_t(64), uint8_t(data_.blockCount), {1, 1, 16, 15});
        Copy(outWords[64], inWords[64], uint64_t(56), uint8_t(data_.blockCount), {1, 1, 16, 15});
        // Refresh this packet's layout marker on every reused lane, before
        // the existing V_MTE3 publication dependency.
        uint64_t markerMask[2] = {1U, 0U};
        Duplicate<float>(out.template ReinterpretCast<float>()[PacketReceiveLayout::kMarkerOffset / 4U],
                         mask != 0U ? PacketReceiveLayout::kKeepFlags : PacketReceiveLayout::kCompact, markerMask, 1U, 1U, 8U);
        if constexpr (!NeedQuantize)
            input.template FreeTensor<XType>(x);
        SetFlag<HardEvent::V_MTE3>(lane);
        WaitFlag<HardEvent::V_MTE3>(lane);
        GlobalTensor<uint8_t> destinationGm;
        destinationGm.SetGlobalBuffer(args_->peerMems[destination] +
                                      DispatchServerDedupDispatchOffset(data_, args_->rank, destination, expert % localExperts_, q,
                                                                        args_->rankSize, localExperts_, bs_));
        DataCopy(destinationGm, out, uint32_t(data_.tokenStride));
        SetFlag<HardEvent::MTE3_V>(lane);
        ++packets_;
    }

    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    TPipe* pipe_ = nullptr;
    bool enableDedup_ = false;
    TBufPool<TPosition::VECCALC, 16>* scratch_ = nullptr;
    uint32_t bs_ = 0, h_ = 0, k_ = 0, localExperts_ = 0, start_ = 0, count_ = 0;
    uint32_t usedBytes_ = 0, lanes_ = 0, packets_ = 0, directCount_ = 0;
    EpServerDedupDataLayout data_{};
    GlobalTensor<XType> x_;
    GlobalTensor<uint8_t> inputScales_;
    GlobalTensor<int32_t> ids_;
    GlobalTensor<float> weights_, weightOut_;
    GlobalTensor<uint32_t> maskOut_;
    TBuf<> idsBuf_, qDstBuf_, qSubBuf_, qWorkBuf_, quantBuf_;
    TBuf<> weightInBuf_, weightOutBuf_, maskBuf_, directBuf_, serverBuf_, firstHopBuf_, outBuf_;
    LocalTensor<int32_t> routeIds_;
    LocalTensor<float> routeWeights_, sourceWeights_;
    LocalTensor<uint32_t> sourceMasks_, directTasks_, serverState_, firstHopMasks_;
};
}  // namespace DispatchDedup
