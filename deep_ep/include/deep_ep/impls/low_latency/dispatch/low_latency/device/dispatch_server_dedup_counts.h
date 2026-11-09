#pragma once

// Copyright (c) 2026, Lu Lu


#include <kernel_operator.h>
#include <adv_api/reduce/sum.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h>

namespace DispatchDedup {
using namespace AscendC;

// Adapted from Memory's CalAndSendCntByRankAndExpert / WaitDispatch / cumsum
// chain. Counts cover the complete original routing tensor, including relay routes.
// The caller runs SendCnt on all participating cores, then RecvCnt,
// CalcCumsum and CompleteCumsum on receivers. Senders proceed to token
// transmission after SendCnt.
class ServerDedupCounts {
public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR expertIds, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, uint32_t bs,
                                uint32_t topK, uint32_t experts, uint32_t expertTokenNumsType, int64_t magic, TPipe* pipe) {
        tpipe_ = pipe;
        args_ = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs);
        epRankId_ = args_->rank;
        epWorldSize_ = args_->rankSize;
        aivId_ = GetBlockIdx();
        axisBS_ = bs;
        axisK_ = topK;
        expertIdsCnt_ = bs * topK;
        rscvStatusNum_ = experts;
        moeExpertNumPerRank_ = experts / epWorldSize_;
        expertTokenNumsType_ = expertTokenNumsType;
        useAllCoreCount_ = bs < kEpServerDedupAllCoreCountBsThreshold;
        layout_ = DispatchServerDedupCountLayout(experts);
        stateOffset_ = DispatchServerDedupWindowOffset(magic) + layout_.cntOffset;
        GM_ADDR localState = args_->peerMems[epRankId_] + stateOffset_;
        windowInstatusFp32Tensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localState));
        selfRankWinInGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localState));
        recvCntWorkspaceGM_ = localState + layout_.prefixOffset;
        sendCountsOutGM_ = sendCountsOut;
        expertIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertIds));
        expertTokenNumsOutGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(expertTokenNumsOut));
        // Memory's count reductions add integer bit patterns as FP32 subnormals.
        originalOverflowMode_ = GetCtrlSpr<60, 60>();
        SetCtrlSpr<60, 60>(0);
    }

    __aicore__ inline void SendCnt();
    __aicore__ inline void RecvCnt();
    __aicore__ inline void CalcCumsum() { CalRecvAndSetFlag(); }
    __aicore__ inline void CompleteCumsum();
    // Scratch excludes statusBuf_, which may still be read by count DMA.
    __aicore__ inline TBufPool<TPosition::VECCALC, 16>& ScratchPool() { return stagePool_; }
    __aicore__ inline uint32_t ReservedBytes() const { return statusCntAlign_ * kEpServerDedupCountRecordBytes; }
    // Valid after CompleteCumsum and until this core's TPipe is reset.
    __aicore__ inline LocalTensor<int32_t> Prefix() { return prefixBuf_.Get<int32_t>(); }
    __aicore__ inline void Finish() {
        PipeBarrier<PIPE_ALL>();
        SetCtrlSpr<60, 60>(originalOverflowMode_);
    }

private:
    static constexpr uint32_t UB_ALIGN = 32U;
    static constexpr uint32_t UB_ALIGN_DATA_COUNT = 8U;
    static constexpr uint32_t SIZE_ALIGN_256 = 256U;
    static constexpr uint32_t CNT_GM_STRIDE_BLOCKS = kEpServerDedupCountGmStride / UB_ALIGN;
    static constexpr uint32_t aivUsedAllToAll_ = kEpServerDedupSendCoreNum;
    static constexpr uint32_t aivUsedCumSum_ = kEpServerDedupRecvCoreNum;

    template <HardEvent event>
    __aicore__ inline void SyncFunc() {
        const TEventID id = tpipe_->FetchEventID(event);
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }
    __aicore__ inline void SplitToCore(uint32_t total, uint32_t cores, uint32_t core, uint32_t& start, uint32_t& end, uint32_t& count) {
        const uint32_t base = total / cores;
        const uint32_t remainder = total % cores;
        count = base + (core < remainder ? 1U : 0U);
        start = core * base + (core < remainder ? core : remainder);
        end = start + count;
    }
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t& curExpertCnt);
    __aicore__ inline void SendExpertCntRange(uint32_t start, uint32_t end);
    __aicore__ inline void CountByVectorRange(uint32_t start, uint32_t end, uint32_t maskBytes);
    __aicore__ inline void CountByExpert(uint32_t start, uint32_t end);
    __aicore__ inline void WaitDispatchClearStatus();
    __aicore__ inline void GatherSumRecvCnt(LocalTensor<float>& gatherMaskOutTensor, LocalTensor<uint32_t>& gatherTmpTensor,
                                            LocalTensor<float>& statusSumOutTensor);
    __aicore__ inline void PublishRecvCntSum(float sumOfRecvCnt);
    __aicore__ inline uint32_t GetCumSum(uint32_t newAivId);
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void CalRecvAndSetFlag();
    __aicore__ inline void WaitCumSumFlag();

    TPipe* tpipe_ = nullptr;
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    EpServerDedupCountLayout layout_{};
    uint64_t stateOffset_ = 0;
    int64_t originalOverflowMode_ = 0;
    uint32_t epRankId_ = 0, epWorldSize_ = 0, aivId_ = 0;
    uint32_t axisBS_ = 0, axisK_ = 0, expertIdsCnt_ = 0;
    uint32_t rscvStatusNum_ = 0, moeExpertNumPerRank_ = 0, expertTokenNumsType_ = 0;
    uint32_t statusCntAlign_ = 0;
    uint32_t startStatusIndex_ = 0, endStatusIndex_ = 0, recStatusNumPerCore_ = 0;
    uint32_t activeCumSumCores_ = 0;
    bool useAllCoreCount_ = false;
    GM_ADDR recvCntWorkspaceGM_ = nullptr;
    GM_ADDR sendCountsOutGM_ = nullptr;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_, selfRankWinInGMTensor_;
    LocalTensor<int32_t> statusTensor_, validExpertIdsTensor_;
    LocalTensor<float> workLocalTensor_, statusFp32Tensor_, statusCleanFp32Tensor_;
    // Both views outlive all their tensors. The record buffer is outside their
    // shared address range and remains immutable until the phase is retired.
    TBufPool<TPosition::VECCALC, 16> stagePool_;
    TBufPool<TPosition::VECCALC, 5> countPool_;
    TBuf<> expertIdsBuf_, dstExpBuf_, subExpBuf_, workBuf_, statusBuf_, histogramBuf_;
    TBuf<> waitStatusBuf_, gatherMaskOutBuf_, sumCoreBuf_, sumLocalBuf_, sumContinueBuf_, scalarBuf_;
    TBuf<> prefixBuf_, readyBuf_, cleanBuf_, tokenNumBuf_;
};

// Publish counts for the original routes; keep record UB alive until MTE3 completes.
__aicore__ inline void ServerDedupCounts::SendCnt() {
    // Partition by expert: small batches use all 64 cores, larger batches use
    // receiver cores. Each core reads the full routing tensor for its expert range.
    const uint32_t cores = useAllCoreCount_ ? kEpServerDedupCoreNum : aivUsedCumSum_;
    uint32_t start = 0, end = 0, count = 0;
    if (useAllCoreCount_ || aivId_ >= aivUsedAllToAll_) {
        const uint32_t core = useAllCoreCount_ ? aivId_ : aivId_ - aivUsedAllToAll_;
        SplitToCore(rscvStatusNum_, cores, core, start, end, count);
    }
    // Keep each 32-byte count record in statusBuf_ outside the reusable pool.
    // Initialize stagePool_ even for empty expert ranges; later phases reuse it.
    statusCntAlign_ = count == 0U ? 0U : Ceil(Ceil(rscvStatusNum_, cores), UB_ALIGN_DATA_COUNT) * UB_ALIGN_DATA_COUNT;
    if (count != 0U)
        tpipe_->InitBuffer(statusBuf_, ReservedBytes());
    tpipe_->InitBufPool(stagePool_, kEpServerDedupCountUbBytes - ReservedBytes());
    if (count == 0U)
        return;

    // Load all bs * topK routes before using the filtering and reduction scratch.
    const uint32_t routeBytes = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBufPool(countPool_, 4U * routeBytes + statusCntAlign_ * sizeof(int32_t), stagePool_);
    countPool_.InitBuffer(expertIdsBuf_, routeBytes);
    countPool_.InitBuffer(dstExpBuf_, routeBytes);
    countPool_.InitBuffer(subExpBuf_, routeBytes);
    countPool_.InitBuffer(workBuf_, routeBytes);
    countPool_.InitBuffer(histogramBuf_, statusCntAlign_ * sizeof(int32_t));
    validExpertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    workLocalTensor_ = workBuf_.Get<float>();
    statusTensor_ = statusBuf_.Get<int32_t>();
    const DataCopyExtParams idsCopy{1U, expertIdsCnt_ * uint32_t(sizeof(int32_t)), 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0};
    DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, idsCopy, pad);
    SyncFunc<HardEvent::MTE2_V>();

    // Use vector filtering and histograms for larger inputs, per-expert counts
    // otherwise. Both paths publish ready=1 even for zero counts.
    const uint32_t maskBytes = Ceil(Ceil(expertIdsCnt_, 8U), UB_ALIGN) * UB_ALIGN;

    if (expertIdsCnt_ >= SIZE_ALIGN_256 / sizeof(int32_t) && 2U * maskBytes <= routeBytes) {
        CountByVectorRange(start, end, maskBytes);
    } else {
        CountByExpert(start, end);
    }

    // Issue MTE3 writes without draining here. statusBuf_ remains immutable
    // while the filtering scratch can be reused by the next phase.
    SyncFunc<HardEvent::S_MTE3>();
    SendExpertCntRange(start, end);
    // Only statusBuf_ is still read by MTE3. Count's V_S dependencies already
    // retired the shared scratch; do not Reset either pool (A5 drains MTE3).
    // Keeping the pipe also preserves Init's CTRL[60] through FP8 quantization.
}

// Filter this core's expert range and produce ready/count records with a histogram.
__aicore__ inline void ServerDedupCounts::CountByVectorRange(uint32_t start, uint32_t end, uint32_t maskBytes) {
    LocalTensor<int32_t> histogram = histogramBuf_.Get<int32_t>();
    LocalTensor<int32_t> selected = dstExpBuf_.Get<int32_t>();
    LocalTensor<uint8_t> lower = subExpBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> upper = lower[maskBytes];
    Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * UB_ALIGN_DATA_COUNT);
    Duplicate<int32_t>(histogram, 0, statusCntAlign_);
    Compares(lower, validExpertIdsTensor_, int32_t(start), CMPMODE::GE, expertIdsCnt_);
    Compares(upper, validExpertIdsTensor_, int32_t(end), CMPMODE::LT, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    And(lower.ReinterpretCast<uint16_t>(), lower.ReinterpretCast<uint16_t>(), upper.ReinterpretCast<uint16_t>(),
        int32_t(maskBytes / sizeof(uint16_t)));
    PipeBarrier<PIPE_V>();
    uint64_t selectedCount = 0;
    GatherMask(selected, validExpertIdsTensor_, lower.ReinterpretCast<uint32_t>(), true, expertIdsCnt_, {1U, 1U, 0U, 0U}, selectedCount);
    SyncFunc<HardEvent::V_S>();
    for (uint32_t i = 0; i < selectedCount; ++i) {
        const uint32_t record = uint32_t(selected.GetValue(i)) - start;
        histogram.SetValue(record, histogram.GetValue(record) + 1);
    }
    for (uint32_t i = 0; i < end - start; ++i) {
        statusTensor_.SetValue(i * UB_ALIGN_DATA_COUNT, 0x3f800000);
        statusTensor_.SetValue(i * UB_ALIGN_DATA_COUNT + 1U, histogram.GetValue(i));
    }
}

// Count each expert separately for small inputs.
__aicore__ inline void ServerDedupCounts::CountByExpert(uint32_t start, uint32_t end) {
    uint64_t flagMask[2] = {0x0101010101010101ULL, 0ULL};
    Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * UB_ALIGN_DATA_COUNT);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3f800000, flagMask, statusCntAlign_ / 8U, 1U, 8U);
    SyncFunc<HardEvent::V_S>();
    for (uint32_t expert = start; expert < end; ++expert) {
        int32_t count = 0;
        CalTokenSendExpertCnt(expert, expertIdsCnt_, count);
        statusTensor_.SetValue((expert - start) * UB_ALIGN_DATA_COUNT + 1U, count);
    }
}

// Write 32-byte records to remote slots ordered by local expert and source rank.
__aicore__ inline void ServerDedupCounts::SendExpertCntRange(uint32_t start, uint32_t end) {
    GlobalTensor<int32_t> peer;
    for (uint32_t expert = start; expert < end;) {
        const uint32_t dst = expert / moeExpertNumPerRank_;
        const uint32_t localExpert = expert % moeExpertNumPerRank_;
        const uint32_t count = end - expert < moeExpertNumPerRank_ - localExpert ? end - expert : moeExpertNumPerRank_ - localExpert;
        const uint64_t offset = (uint64_t(localExpert) * epWorldSize_ + epRankId_) * kEpServerDedupCountGmStride;
        peer.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args_->peerMems[dst] + stateOffset_ + offset));
        // DataCopy strides are GM count slots: one record per source rank.
        const DataCopyParams copy{uint16_t(count), 1U, 0U, uint16_t(epWorldSize_ * CNT_GM_STRIDE_BLOCKS - 1U)};
        DataCopy<int32_t>(peer, statusTensor_[(expert - start) * UB_ALIGN_DATA_COUNT], copy);
        expert += count;
    }
}

// Wait for this receiver's count records and compute its local prefix and total.
__aicore__ inline void ServerDedupCounts::RecvCnt() {
    // The receiver table has localExperts * worldSize records, ordered by
    // local expert then source rank. Each receiver owns one contiguous segment.
    // Each count owns a 512B GM slot, so contiguous ownership can split at any
    // record. Keep all nonempty receivers active and their loads within one record.
    activeCumSumCores_ = rscvStatusNum_ < aivUsedCumSum_ ? rscvStatusNum_ : aivUsedCumSum_;
    stagePool_.InitBuffer(scalarBuf_, 3U * UB_ALIGN);
    const uint32_t receiver = aivId_ - aivUsedAllToAll_;
    // Inactive receivers only consume the final prefix in CompleteCumsum.
    if (receiver >= activeCumSumCores_)
        return;
    SplitToCore(rscvStatusNum_, aivUsedCumSum_, receiver, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_);
    stagePool_.InitBuffer(waitStatusBuf_, Ceil(recStatusNumPerCore_ * UB_ALIGN, SIZE_ALIGN_256) * SIZE_ALIGN_256);
    stagePool_.InitBuffer(gatherMaskOutBuf_, Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    // Vector publication/cleanup touches complete groups of eight 32B records.
    // Padding is UB-only; matrix strides and participation use actual core counts.
    const uint32_t sumBytes = Ceil(aivUsedCumSum_, 8U) * SIZE_ALIGN_256;
    stagePool_.InitBuffer(sumCoreBuf_, sumBytes);
    stagePool_.InitBuffer(sumLocalBuf_, sumBytes);
    stagePool_.InitBuffer(sumContinueBuf_, Ceil(aivUsedCumSum_ * sizeof(float), UB_ALIGN) * UB_ALIGN);
    // Publish the segment total after count readiness, without waiting for
    // token payloads. Empty segments must publish zero for the cross-core prefix.
    WaitDispatch();
}

// Load the complete prefix after all segments are ready; receiver 0 writes expert counts.
__aicore__ inline void ServerDedupCounts::CompleteCumsum() {
    // Convert the physical AIV index to its receive-group index.
    // Map the physical AIV index to the receiver group before loading its prefix copy.
    const uint32_t receiver = aivId_ - aivUsedAllToAll_;
    stagePool_.InitBuffer(readyBuf_, activeCumSumCores_ * UB_ALIGN);
    stagePool_.InitBuffer(cleanBuf_, activeCumSumCores_ * UB_ALIGN);
    stagePool_.InitBuffer(prefixBuf_, Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    statusFp32Tensor_ = readyBuf_.Get<float>();
    statusCleanFp32Tensor_ = cleanBuf_.Get<float>();
    // Each core publishes completion only after its prefix writes; wait for all cores.
    WaitCumSumFlag();
    LocalTensor<int32_t> prefix = prefixBuf_.Get<int32_t>();
    GlobalTensor<int32_t> workspace;
    workspace.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvCntWorkspaceGM_));
    const DataCopyExtParams copy{1U, rscvStatusNum_ * uint32_t(sizeof(int32_t)), 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0};
    DataCopyPad(prefix, workspace[receiver * rscvStatusNum_], copy, pad);
    SyncFunc<HardEvent::MTE2_S>();
    // Only receiver 0 writes public expert counts. The last source prefix
    // is the expert endpoint: type=0 returns cumulative counts, other validated
    // types return the difference from the preceding expert endpoint.
    if (receiver == 0U) {
        stagePool_.InitBuffer(tokenNumBuf_, Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN);
        LocalTensor<int64_t> expertCounts = tokenNumBuf_.Get<int64_t>();
        int64_t previous = 0;
        for (uint32_t expert = 0; expert < moeExpertNumPerRank_; ++expert) {
            const int64_t cumulative = prefix.GetValue((expert + 1U) * epWorldSize_ - 1U);
            expertCounts.SetValue(expert, expertTokenNumsType_ == 0U ? cumulative : cumulative - previous);
            previous = cumulative;
        }
        SyncFunc<HardEvent::S_MTE3>();
        const DataCopyExtParams output{1U, moeExpertNumPerRank_ * uint32_t(sizeof(int64_t)), 0U, 0U, 0U};
        DataCopyPad(expertTokenNumsOutGMTensor_, expertCounts, output);
    }
    // No listening table or payload receive/copy is constructed in this increment.
}

// Match and reduce the original routes for one expert.
__aicore__ inline void ServerDedupCounts::CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t& curExpertCnt) {
    if (calCnt < axisK_) {
        curExpertCnt = 0;
        return;
    }
    LocalTensor<int32_t> dstExpIdTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> subExpIdTensor = subExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(dstExpIdTensor, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor, validExpertIdsTensor_, dstExpIdTensor, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor, dstExpIdTensor, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

// Clear consumed ready markers while retaining counts for window reuse.
__aicore__ inline void ServerDedupCounts::WaitDispatchClearStatus() {
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, static_cast<uint16_t>(CNT_GM_STRIDE_BLOCKS - 1U)};
    uint64_t duplicateMask[2] = {0x0101010101010101ULL, 0ULL};
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * kEpServerDedupCountGmStride / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    // The snapshot is no longer modified after this issue. Its DMA can overlap
    // cumsum; the caller retires all count-stage writes before receiver Reset.
}

// Extract counts, compute the segment prefix and publish its total to other receivers.
__aicore__ inline void ServerDedupCounts::GatherSumRecvCnt(LocalTensor<float>& gatherMaskOutTensor, LocalTensor<uint32_t>& gatherTmpTensor,
                                                           LocalTensor<float>& statusSumOutTensor) {
    gatherTmpTensor.SetValue(0, 2U);
    uint32_t mask = 2U;
    SyncFunc<AscendC::HardEvent::S_V>();

    uint64_t recvCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask, {1, (uint16_t)recStatusNumPerCore_, 1, 0}, recvCnt);
    PipeBarrier<PIPE_V>();

    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    float sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<float>().GetValue(0);

    // Compute the relative prefix while outgoing cnt may still be in flight.
    // Gather preserved the integer count bit patterns used by the FP32 Sum.
    LocalTensor<int32_t> prefix = gatherMaskOutTensor.ReinterpretCast<int32_t>();
    uint32_t cumulative = 0;
    for (uint32_t i = 0; i < recStatusNumPerCore_; ++i) {
        cumulative += uint32_t(prefix.GetValue(i));
        prefix.SetValue(i, cumulative);
    }
    PublishRecvCntSum(sumOfRecvCnt);
}

// Publish this segment's total and ready marker, including for empty segments.
__aicore__ inline void ServerDedupCounts::PublishRecvCntSum(float sumOfRecvCnt) {
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;

    LocalTensor<float> sumCoreFP32Tensor = sumCoreBuf_.Get<float>();
    uint64_t maskArrayCount[2] = {0x0101010101010101, 0};
    uint8_t repeatTimes = Ceil(activeCumSumCores_, 8);

    Duplicate<float>(sumCoreFP32Tensor, sumOfRecvCnt, maskArrayCount, repeatTimes, 1, 8);
    uint64_t maskArrayFlag[2] = {0x0202020202020202, 0};
    Duplicate<float>(sumCoreFP32Tensor, static_cast<float>(1.0), maskArrayFlag, repeatTimes, 1, 8);
    // Keep the matrix row stride, but publish only to active cumsum consumers.
    DataCopyParams sumIntriParams{static_cast<uint16_t>(activeCumSumCores_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(layout_.cumsumOffset + newAivId * aivUsedCumSum_ * UB_ALIGN) / sizeof(float)], sumCoreFP32Tensor,
             sumIntriParams);
    // sumCoreBuf_ stays immutable until CalRecvAndSetFlag reuses it. Polling
    // incoming sums must not wait here for this core's outgoing cnt.
}

// Wait for all receiver totals, sum preceding segments and clear consumed markers.
__aicore__ inline uint32_t ServerDedupCounts::GetCumSum(uint32_t newAivId) {
    uint32_t offset = 0;
    DataCopyParams sumIntriParams{static_cast<uint16_t>(activeCumSumCores_), 1, static_cast<uint16_t>(aivUsedCumSum_ - 1), 0};
    LocalTensor<float> sumLocalTensor = sumLocalBuf_.Get<float>();
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> sumContinueTensor = sumContinueBuf_.Get<float>();
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);

    uint32_t mask = 2;
    uint64_t recvCnt = 0;
    uint32_t innerSumParams = Ceil(activeCumSumCores_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, innerSumParams, activeCumSumCores_};
    int32_t cumSumFlag = 0;
    gatherSumPattern.SetValue(0, 2);
    SyncFunc<AscendC::HardEvent::S_V>();

    while (true) {
        DataCopy(sumLocalTensor, selfRankWinInGMTensor_[(layout_.cumsumOffset + newAivId * UB_ALIGN) / sizeof(float)], sumIntriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, static_cast<uint16_t>(activeCumSumCores_), 1, 0},
                   recvCnt);
        PipeBarrier<PIPE_V>();
        Sum(recvCntSumOutTensor, sumContinueTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = static_cast<int32_t>(recvCntSumOutTensor.GetValue(0));

        if (cumSumFlag == activeCumSumCores_) {
            break;
        }
    }
    if (newAivId != 0U) {
        mask = 1;
        recvCnt = 0;
        gatherSumPattern.SetValue(0, 1);
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, static_cast<uint16_t>(newAivId), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        uint32_t innerCumSumParams = Ceil(newAivId * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
        SumParams cumSumParams{1, innerCumSumParams, newAivId};
        Sum(recvCntSumOutTensor, sumContinueTensor, cumSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        offset = uint32_t(recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
    }

    LocalTensor<float> sumCoreFp32Tensor = sumLocalBuf_.Get<float>();

    uint8_t repeatTimes = Ceil(activeCumSumCores_, 8);

    Duplicate<float>(sumCoreFp32Tensor, static_cast<float>(0), 64, repeatTimes, 1, 8);
    DataCopyParams cleanParams{static_cast<uint16_t>(activeCumSumCores_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(layout_.cumsumOffset + newAivId * UB_ALIGN) / sizeof(float)], sumCoreFp32Tensor, cleanParams);
    return offset;
}

// Wait for every source record, then extract counts, publish the total and clear readiness.
__aicore__ inline void ServerDedupCounts::WaitDispatch() {
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    uint32_t mask = 1U;
    gatherTmpTensor.SetValue(0, 1U);
    float compareTarget = static_cast<float>(1.0) * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);

    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, static_cast<uint16_t>(CNT_GM_STRIDE_BLOCKS - 1U), 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * kEpServerDedupCountGmStride / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1U);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }

    GatherSumRecvCnt(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
    WaitDispatchClearStatus();
}

// Add the cross-core offset, write all prefix copies and publish completion.
__aicore__ inline void ServerDedupCounts::CalRecvAndSetFlag() {
    if (recStatusNumPerCore_ == 0U)
        return;
    LocalTensor<int32_t> outCountLocal = gatherMaskOutBuf_.Get<int32_t>();
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    // Wait for segment totals and add those from preceding receivers.
    // Each core can then adjust its own segment to the full receive-table prefix.
    const uint32_t offset = GetCumSum(newAivId);

    if (recStatusNumPerCore_ != 0U) {
        for (uint32_t i = 0; i < recStatusNumPerCore_; ++i) {
            outCountLocal.SetValue(i, uint32_t(outCountLocal.GetValue(i)) + offset);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
        sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
        workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvCntWorkspaceGM_));
        DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(recStatusNumPerCore_ * sizeof(int32_t)), 0U, 0U, 0U};
        // sendCountsOut stores cumulative prefixes.
        DataCopyPad(sendCountsGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams);

        // Every receiver keeps a complete prefix copy; write this segment to all copies.
        for (uint32_t index = 0; index < aivUsedCumSum_; index++) {
            DataCopyPad(workspaceGlobal[index * rscvStatusNum_ + startStatusIndex_], outCountLocal, dataCopyOutParams);
        }
    }
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    LocalTensor<int32_t> syncOnCoreTensor = sumCoreBuf_.Get<int32_t>();
    LocalTensor<float> syncOnCoreFP32Tensor = sumCoreBuf_.Get<float>();

    // Wait until the previous total DMA no longer reads sumCoreBuf_ before
    // reusing it for completion markers. Preserve MTE3 ordering after prefix writes.
    // This is the first overwrite of the asynchronous local-sum publication.
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Duplicate<int32_t>(syncOnCoreTensor, static_cast<int32_t>(1), SIZE_ALIGN_256 / sizeof(int32_t), repeatTimes, 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    PipeBarrier<PIPE_MTE3>();

    DataCopy(selfRankWinInGMTensor_[(layout_.cumsumFlagOffset + newAivId * UB_ALIGN) / sizeof(float)], syncOnCoreFP32Tensor,
             sumIntriParams);
}

// Wait for all prefix writes and clear this receiver's completion-marker row.
__aicore__ inline void ServerDedupCounts::WaitCumSumFlag() {
    int32_t cumSumFlag = 0;
    int32_t targetFlag = activeCumSumCores_ * UB_ALIGN_DATA_COUNT;
    uint32_t cumSumFlagOffset = (layout_.cumsumFlagOffset + (aivId_ - aivUsedAllToAll_) * aivUsedCumSum_ * UB_ALIGN) / sizeof(float);
    uint32_t innerSumParams = activeCumSumCores_ * UB_ALIGN / sizeof(float);
    SumParams sumFlagParams{1, innerSumParams, activeCumSumCores_ * UB_ALIGN_DATA_COUNT};
    LocalTensor<float> statusSumOutTensor = scalarBuf_.Get<float>();

    while (true) {
        DataCopy(statusFp32Tensor_, selfRankWinInGMTensor_[cumSumFlagOffset], activeCumSumCores_ * UB_ALIGN_DATA_COUNT);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusSumOutTensor, statusFp32Tensor_, sumFlagParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);

        if (cumSumFlag == targetFlag) {
            break;
        }
    }
    Duplicate<float>(statusCleanFp32Tensor_, static_cast<float>(0), activeCumSumCores_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<HardEvent::S_MTE3>();

    SyncFunc<HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[cumSumFlagOffset], statusCleanFp32Tensor_, activeCumSumCores_ * UB_ALIGN_DATA_COUNT);
}

}  // namespace DispatchDedup
