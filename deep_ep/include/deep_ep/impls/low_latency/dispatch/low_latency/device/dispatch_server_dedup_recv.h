#pragma once

// Copyright (c) 2026, Lu Lu


#include <kernel_operator.h>
#include <adv_api/reduce/sum.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_protocol.h>
#include <deep_ep/impls/low_latency/common/ep_packet_receive_layout.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_server_dedup_receive_tasks.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_server_dedup_receive_packet.h>

namespace DispatchDedup {
using namespace AscendC;

template <class XType, bool WireFp8, bool E4M3, bool Prequantized = false>
class ServerDedupReceiver {
    static_assert(!Prequantized || WireFp8, "prequantized input requires FP8 wire");
    static constexpr EpServerDedupInputMode InputMode =
        Prequantized ? EpServerDedupInputMode::PrequantizedFp8Packs
                     : (WireFp8 ? EpServerDedupInputMode::Quantize16 : EpServerDedupInputMode::Plain16);

public:
    __aicore__ inline void Init(const __gm__ DispatchDedup::CommArgs* args, GM_ADDR expandX, GM_ADDR dynamicScales, GM_ADDR tokenType,
                                GM_ADDR destinationIndex, GM_ADDR relayReadIndex, uint32_t bs, uint32_t h, uint32_t k, uint32_t experts,
                                TPipe* pipe, int64_t magic = 0, bool enableDedup = true) {
        args_ = args;
        pipe_ = pipe;
        enableDedup_ = enableDedup;
        bs_ = bs;
        k_ = k;
        experts_ = experts;
        localExperts_ = experts / uint32_t(args_->rankSize);
        receiver_ = GetBlockIdx() - kEpServerDedupSendCoreNum;
        hiddenBytes_ = WireFp8 ? h : h * uint32_t(sizeof(XType));
        data_ = DispatchServerDedupDataLayout(bs, h, k, experts, InputMode);
        windowOffset_ = DispatchServerDedupWindowOffset(magic);
        data_.dispatchOffset += windowOffset_;
        expandX_.SetGlobalBuffer(expandX);
        dynamicScales_.SetGlobalBuffer(dynamicScales);
        tokenType_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tokenType));
        destinationIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(destinationIndex));
        relayReadIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(relayReadIndex));
    }

    __aicore__ inline void Process() {
        // Count-stage writes must finish before resetting TPipe. Load this
        // receiver's prefix from the active window; MTE2_S guards scalar access.

        // Caller has completed count-stage writes before this Reset. The GM
        // prefix belongs to this receive core; old count UB is no longer used.
        pipe_->Reset();
        const uint32_t prefixBytes = uint32_t(DispatchDedupAlign(uint64_t(experts_) * 4U, 32U));
        pipe_->InitBuffer(prefixBuf_, prefixBytes);
        GlobalTensor<int32_t> prefixGlobal;
        const auto counts = DispatchServerDedupCountLayout(experts_);
        prefixGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args_->peerMems[args_->rank] + windowOffset_ + counts.cntOffset +
                                                                       counts.prefixOffset + uint64_t(receiver_) * counts.prefixStride));
        auto prefix = prefixBuf_.Get<int32_t>();
        DataCopyPad(prefix, prefixGlobal, {1U, prefixBytes, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        ServerDedupReceiveSync<HardEvent::MTE2_S>();

        // Partition output rows into a stable slot table and a compactable pending
        // table. j always identifies the original row/slot, not its pending position.
        tasks_.Build(prefix, uint32_t(args_->rank), uint32_t(args_->rankSize), experts_, bs_, receiver_, pipe_);

        // A core with no tasks owns no receive slots and can return without polling.
        if (tasks_.Count() == 0U) {
            return;
        }

        // Probe up to 16 packets per batch. flags and work include aligned
        // reduction tails; zero the unused flag area before probing.
        const uint32_t blocks = uint32_t(data_.blockCount);
        const uint32_t flagsBytes = kProbeBatch * blocks * 32U + 256U;
        // CANN 9.1/A5 Sum stores ceil(B*8/64) partials per row at a
        // 32-element stride. Reserve all rows plus a full-vector tail read.
        const uint32_t workBytes = kProbeBatch * uint32_t(DispatchDedupAlign((blocks + 7U) / 8U, 32U)) * 4U + 256U;
        pipe_->InitBuffer(flagsBuf_, flagsBytes);
        pipe_->InitBuffer(sumBuf_, kProbeBatch * 32U);
        pipe_->InitBuffer(workBuf_, workBytes);
        // LoadAlign may read past the masked tail, including inactive rows.
        Duplicate(flagsBuf_.Get<float>(), 0.0f, flagsBytes / 4U);
        ServerDedupReceiveSync<HardEvent::V_MTE2>();
        pipe_->InitBuffer(probeIdxBuf_, kProbeBatch * 4U);
        const uint32_t tablesBytes = 2U * uint32_t(DispatchDedupAlign(uint64_t(tasks_.Count()) * 4U, 32U));
        const uint32_t fixed = prefixBytes + tablesBytes + flagsBytes + workBytes + kProbeBatch * 36U;
        // Allocate up to eight prefetch lanes, each with a full packet and 64 bytes
        // of metadata. MTE3_MTE2 prevents reuse before output and relay DMA finish.
        laneStride_ = uint32_t(data_.tokenStride) + 64U;
        const uint32_t available = (kEpServerDedupCountUbBytes - fixed) / laneStride_;
        lanes_ = available < 8U ? available : 8U;  // Safe inputs guarantee at least one lane.
        pipe_->InitBuffer(lanesBuf_, lanes_ * laneStride_);
        for (uint32_t lane = 0; lane < lanes_; ++lane)
            SetFlag<HardEvent::MTE3_MTE2>(lane);

        auto snapshot = probeIdxBuf_.Get<uint32_t>();
        // Fill every allocated lane (up to eight); reuse still waits for MTE3.
        const uint32_t depth = lanes_;
        // issued and consumed advance in the same order across batches. Their
        // difference stays within lane depth, so consumed % lanes_ names the loaded packet.
        uint32_t scan = 0U, issued = 0U, consumed = 0U;
        while (tasks_.PendingCount() != 0U) {
            // Include all empty scans in one wait interval. Recording every
            // retry perturbs the polling loop and can exhaust the trace buffer.
            // Scan pending batches until some packet is ready. Probe checks all
            // block flags before allowing payload access.

            uint32_t count;
            ProbeResult probe;
            do {
                if (scan >= tasks_.PendingCount())
                    scan = 0U;
                const uint32_t remaining = tasks_.PendingCount() - scan;
                count = remaining < kProbeBatch ? remaining : kProbeBatch;
                probe = Probe(scan, count);
                if (probe.readyMask == 0U)
                    scan += count;
            } while (probe.readyMask == 0U);

            // Load and consume the immutable snapshot in the same descending
            // order. Higher-index removals cannot change lower pending entries.
            // Probe saves an immutable task snapshot. Process it in reverse index
            // order so RemovePending's swap-with-last cannot invalidate lower pending indices.
            uint32_t nextLoad = count;
            for (uint32_t i = count; i > 0U; --i) {
                const uint32_t index = i - 1U;
                if ((probe.readyMask & (uint32_t(1) << index)) == 0U)
                    continue;
                // Prefetch only ready packets, fill available lanes, then consume in the same reverse order.
                while (issued - consumed < depth && nextLoad != 0U) {
                    --nextLoad;
                    if ((probe.readyMask & (uint32_t(1) << nextLoad)) == 0U)
                        continue;
                    IssueLoad(snapshot.GetValue(nextLoad), issued % lanes_, (probe.keepFlagsMask & (uint32_t(1) << nextLoad)) != 0U);
                    ++issued;
                }
                // Consume writes token/scale and route metadata, plus FIRST_HOP relays.
                // Removing a pending task means its writes were issued; lane events guard completion.
                ConsumeLoaded(snapshot.GetValue(index), consumed % lanes_, (probe.keepFlagsMask & (uint32_t(1) << index)) != 0U);
                tasks_.RemovePending(scan + index);
                ++consumed;
            }
        }
        // After pending is empty, retire lane writes and clear all owned slot
        // flags using the stable task table. Keep payload data intact.

        for (uint32_t lane = 0; lane < lanes_; ++lane)
            WaitFlag<HardEvent::MTE3_MTE2>(lane);
        // This is the end of wait instruction submission, not a new Scalar
        // completion barrier. The existing final PIPE_ALL establishes completion.

        ClearFlags();

        // Wait for output, relay and flag-clear operations before returning.
        PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr uint32_t kProbeBatch = 16U;
    struct ProbeResult {
        uint32_t readyMask = 0U;
        uint32_t keepFlagsMask = 0U;
    };
    __aicore__ inline __gm__ uint8_t* Slot(uint32_t j) {
        return args_->peerMems[args_->rank] + data_.dispatchOffset + uint64_t(tasks_.Position(j)) * data_.tokenStride;
    }

    // Snapshot up to 16 pending tasks and return a bitmap after checking all packet block flags.
    __aicore__ inline ProbeResult Probe(uint32_t start, uint32_t count) {
        auto indices = probeIdxBuf_.Get<uint32_t>();
        auto flags = flagsBuf_.Get<float>();
        auto sums = sumBuf_.Get<float>();
        auto work = workBuf_.Get<uint8_t>();
        const uint32_t blocks = uint32_t(data_.blockCount);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t j = tasks_.PendingIndex(start + i);
            indices.SetValue(i, j);
            ReadServerDedupFlags(Slot(j), blocks, flags[i * blocks * 8U]);
        }
        ServerDedupReceiveSync<HardEvent::MTE2_V>();
        Sum<float>(sums, flags, work, SumParams{count, blocks * 8U, blocks * 8U});
        ServerDedupReceiveSync<HardEvent::V_S>();
        ProbeResult result;
        for (uint32_t i = 0; i < count; ++i) {
            const auto layout = PacketReceiveLayout::Decode(sums.GetValue(i), blocks);
            if (layout != PacketReceiveLayout::State::Pending)
                result.readyMask |= uint32_t(1) << i;
            if (layout == PacketReceiveLayout::State::KeepFlags)
                result.keepFlagsMask |= uint32_t(1) << i;
        }
        return result;
    }

    // Wait for prior lane outputs, prefetch GM to UB, then signal scalar and MTE3 consumers.
    __aicore__ inline void IssueLoad(uint32_t j, uint32_t lane, bool keepFlags) {
        WaitFlag<HardEvent::MTE3_MTE2>(lane);
        auto wire = lanesBuf_.Get<uint8_t>()[lane * laneStride_];
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(Slot(j));
        if (keepFlags) {
            DataCopy(wire, source, uint32_t(data_.tokenStride));
        } else {
            // Compact every 480B payload block, skipping the 32B GM Flag.
            // GM source stride is bytes; UB destination stride is 32B blocks.
            DataCopyPad(wire, source, {uint16_t(data_.blockCount), 480U, 32U, 0U, 0U}, {false, 0U, 0U, 0});
        }
        SetFlag<HardEvent::MTE2_MTE3>(lane);
        // Publish immediately after this load, before queuing later loads.
        SetFlag<HardEvent::MTE2_S>(lane);
    }

    // Write the prefetched token/scale and metadata, relay FIRST_HOP members, then release the lane.
    __aicore__ inline void ConsumeLoaded(uint32_t j, uint32_t lane, bool keepFlags) {
        WaitFlag<HardEvent::MTE2_S>(lane);
        auto wire = lanesBuf_.Get<uint8_t>()[lane * laneStride_];
        const uint32_t row = tasks_.Begin() + j;
        const uint32_t headerOffset = uint32_t(keepFlags ? data_.headerWireOffset : data_.headerPayloadOffset);
        if (!enableDedup_ && !keepFlags) {
            // Compact offsets are logical payload offsets. Retain this lane
            // until all payload, scale and metadata output reads complete.
            WaitFlag<HardEvent::MTE2_MTE3>(lane);
            DataCopyPad(expandX_[uint64_t(row) * hiddenBytes_], wire, {1U, hiddenBytes_, 0U, 0U, 0U});
            if constexpr (WireFp8)
                DataCopyPad(dynamicScales_[uint64_t(row) * data_.scaleBytes], wire[uint32_t(data_.quantCount)],
                            {1U, uint32_t(data_.scaleBytes), 0U, 0U, 0U});
            auto header = wire[headerOffset].template ReinterpretCast<uint32_t>();
            DataCopyPad(destinationIndex_[uint64_t(row) * 3U], header, {1U, 12U, 0U, 0U, 0U});
            SetFlag<HardEvent::MTE3_MTE2>(lane);
            return;
        }
        auto type = wire[uint32_t(data_.tokenStride)].template ReinterpretCast<uint32_t>();
        auto destination = wire[uint32_t(data_.tokenStride) + 32U].template ReinterpretCast<uint32_t>();
        ServerDedupPacketTail tail(wire, headerOffset, uint32_t(keepFlags ? data_.relayReadIndexWireOffset : data_.readIndexPayloadOffset));
        const bool firstHop = enableDedup_ && tail.Header().GetValue(3U) == uint32_t(EpServerDedupTokenType::FirstHop);
        if (enableDedup_)
            BuildServerDedupMetadata(type, destination, tail, tasks_.Position(j));
        else
            for (uint32_t i = 0U; i < 3U; ++i)
                destination.SetValue(i, tail.Header().GetValue(i));
        if (firstHop) {
            PrepareServerDedupForward(tail);
            // Forwarded packets only need compact payload. The hint remains
            // independent of the application-level token type in the header.
            wire.template ReinterpretCast<float>().SetValue(PacketReceiveLayout::kMarkerOffset / 4U, PacketReceiveLayout::kCompact);
        }
        ServerDedupReceiveSync<HardEvent::S_MTE3>();
        WaitFlag<HardEvent::MTE2_MTE3>(lane);
        if (keepFlags) {
            WriteServerDedupPayloadRange(expandX_[uint64_t(row) * hiddenBytes_], wire, 0U, hiddenBytes_);
            if constexpr (WireFp8)
                WriteServerDedupPayloadRange(dynamicScales_[uint64_t(row) * data_.scaleBytes], wire, uint32_t(data_.quantCount),
                                             uint32_t(data_.scaleBytes));
        } else {
            DataCopyPad(expandX_[uint64_t(row) * hiddenBytes_], wire, {1U, hiddenBytes_, 0U, 0U, 0U});
            if constexpr (WireFp8)
                DataCopyPad(dynamicScales_[uint64_t(row) * data_.scaleBytes], wire[uint32_t(data_.quantCount)],
                            {1U, uint32_t(data_.scaleBytes), 0U, 0U, 0U});
        }
        if (enableDedup_)
            DataCopyPad(tokenType_[row], type, {1U, 4U, 0U, 0U, 0U});
        DataCopyPad(destinationIndex_[uint64_t(row) * 3U], destination, {1U, 12U, 0U, 0U, 0U});
        if (firstHop) {
            DataCopyPad(relayReadIndex_[uint64_t(row) * k_ * 2U], tail.ReadIndex(), {1U, k_ * 8U, 0U, 0U, 0U});
            const uint32_t sourceRank = tail.Header().GetValue(0U);
            const uint32_t ranks = uint32_t(args_->rankSize);
            const uint32_t members = tail.ReadIndex().GetValue(0U);
            for (uint32_t i = 0U; i < members; ++i) {
                const uint32_t index = tail.Entry(i).GetValue(0U);
                const uint32_t expertId = index & 0xffffU;
                const uint32_t q = index >> 16U;
                const uint32_t destinationRank = expertId / localExperts_;
                GlobalTensor<uint8_t> output;
                output.SetGlobalBuffer(args_->peerMems[destinationRank] +
                                       DispatchServerDedupDispatchOffset(data_, sourceRank, destinationRank, expertId % localExperts_, q,
                                                                         ranks, localExperts_, bs_));
                DataCopy(output, wire, uint32_t(data_.tokenStride));
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(lane);
    }

    // Clear only the owned 32-byte block flags. The caller waits for completion.
    __aicore__ inline void ClearFlags() {
        auto zeros = flagsBuf_.Get<float>();
        Duplicate(zeros, 0.0f, uint32_t(data_.blockCount) * 8U);
        ServerDedupReceiveSync<HardEvent::V_MTE3>();
        for (uint32_t j = 0; j < tasks_.Count(); ++j) {
            GlobalTensor<float> output;
            output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Slot(j) + 480U));
            DataCopyPad(output, zeros, {uint16_t(data_.blockCount), 32U, 0U, 480U, 0U});
        }
    }

    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    TPipe* pipe_ = nullptr;
    bool enableDedup_ = false;
    uint64_t windowOffset_ = 0;
    static constexpr uint32_t recvCoreNum_ = kEpServerDedupRecvCoreNum;
    uint32_t bs_ = 0U, k_ = 0U, experts_ = 0U, localExperts_ = 0U, receiver_ = 0U;
    uint32_t hiddenBytes_ = 0U, laneStride_ = 0U, lanes_ = 0U;
    EpServerDedupDataLayout data_{};
    ServerDedupReceiveTasks tasks_;
    TBuf<> prefixBuf_, flagsBuf_, sumBuf_, workBuf_, probeIdxBuf_, lanesBuf_;
    GlobalTensor<uint8_t> expandX_, dynamicScales_;
    GlobalTensor<uint32_t> tokenType_, destinationIndex_, relayReadIndex_;
};
}  // namespace DispatchDedup
