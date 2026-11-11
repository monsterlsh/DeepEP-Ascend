#pragma once

#include <climits>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>
#include <deep_ep/layout/ep/token.hpp>

#include <kernel_operator.h>
#include <simt_api/asc_simt.h>
#include <c_api/sync/sync.h>
#include <c_api/vector_datamove/vector_datamove.h>

namespace deep_ep {

static constexpr int kNumOriginMapThreads = 512;

// One block initializes the map before inserting rows. Multiple local experts
// can receive the same token, so any of its rows supplies the unweighted payload.
template <int kNumOrigins, int kNumLocalExperts, int kNumThreads = kNumOriginMapThreads>
__simt_vf__ __aicore__ __launch_bounds__(kNumThreads) void dispatch_ll_build_origin_map(
    __gm__ const int* native_source_ids, __gm__ const int64_t* expert_recv_counts,
    __gm__ int* origin_to_row, const int capacity_rows) {
    const auto thread_idx = static_cast<int>(threadIdx.x);
    int64_t num_rows = 0;
    for (int expert = 0; expert < kNumLocalExperts; ++ expert) {
        const auto count = expert_recv_counts[expert];
        if (count < 0 or count > capacity_rows - num_rows)
            __asc_simt_vf::__trap();
        num_rows += count;
    }

    for (int64_t origin = thread_idx; origin < kNumOrigins; origin += kNumThreads)
        asc_stcg(origin_to_row + origin, -1);
    asc_threadfence();
    asc_syncthreads();

    for (int64_t row = thread_idx; row < num_rows; row += kNumThreads) {
        const auto origin = native_source_ids[row];
        if (origin < 0 or origin >= kNumOrigins)
            __asc_simt_vf::__trap();
        asc_atomic_max(origin_to_row + origin, static_cast<int>(row));
    }
    asc_threadfence();
}

template <int kNumRanks, int kNumMaxTokensPerRank, int kNumLocalExperts>
__global__ __vector__ void dispatch_ll_origin_map_impl(
    __gm__ const int* native_source_ids, __gm__ const int64_t* expert_recv_counts,
    __gm__ int* origin_to_row, const int capacity_rows) {
    AscendC::InitSocState();
    EP_STATIC_ASSERT(kNumRanks > 0 and kNumMaxTokensPerRank > 0 and kNumLocalExperts > 0);
    EP_STATIC_ASSERT(static_cast<int64_t>(kNumRanks) * kNumMaxTokensPerRank <= INT_MAX);
    asc_vf_call<dispatch_ll_build_origin_map<kNumRanks * kNumMaxTokensPerRank, kNumLocalExperts>>(
        dim3(kNumOriginMapThreads, 1, 1), native_source_ids, expert_recv_counts, origin_to_row, capacity_rows);
    asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
}

// Fill the hidden payload once per rank-deduplicated receive slot. The dispatch
// epilogue expands those slots to experts using the metadata prepared earlier.
template <int kNumRanks, int kNumMaxTokensPerRank, int kNumTopk,
          int kNumHiddenBytes, int kNumSFPacks, int kNumVecCores,
          int kNumUbBytes, bool kCachedMode>
__global__ __vector__ void dispatch_ll_copy_payload_impl(
    __gm__ const uint8_t* ll_x, __gm__ const int* origin_to_row, __gm__ const int* psum_num_recv_tokens_per_rank,
    __gm__ const int* cached_recv_src_metadata, __gm__ uint8_t* buffer, const int capacity_rows) {
    AscendC::InitSocState();
    EP_STATIC_ASSERT(kNumRanks > 0 and kNumMaxTokensPerRank > 0 and kNumTopk > 0);
    EP_STATIC_ASSERT(kNumVecCores > 0 and kNumHiddenBytes > 0);
    EP_STATIC_ASSERT(kNumHiddenBytes % kNumUbAlignmentBytes == 0);
    EP_STATIC_ASSERT(kNumHiddenBytes <= kNumUbBytes);
    EP_STATIC_ASSERT(static_cast<int64_t>(kNumRanks) * kNumMaxTokensPerRank <= INT_MAX);

    const auto token_layout = layout::TokenLayout(
        kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, not kCachedMode);
    const auto recv_buffer = layout::BufferLayout(
        token_layout, kNumRanks, kNumMaxTokensPerRank, math::advance_ptr<void>(buffer, 0));
    const auto num_recv_tokens = psum_num_recv_tokens_per_rank[kNumRanks - 1];
    EP_DEVICE_ASSERT(0 <= num_recv_tokens and num_recv_tokens <= kNumRanks * kNumMaxTokensPerRank);
    const auto per_core = math::ceil_div<int64_t>(num_recv_tokens, kNumVecCores);
    const auto begin = static_cast<int>(math::min<int64_t>(
        num_recv_tokens, static_cast<int64_t>(AscendC::GetBlockIdx()) * per_core));
    const auto end = static_cast<int>(math::min<int64_t>(num_recv_tokens, begin + per_core));
    const auto hidden = reinterpret_cast<__ubuf__ uint8_t*>(0);
    int source_rank = 0, source_begin = 0;

    asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (int token_idx = begin; token_idx < end; ++ token_idx) {
        while (source_rank + 1 < kNumRanks and token_idx >= psum_num_recv_tokens_per_rank[source_rank]) {
            source_begin = psum_num_recv_tokens_per_rank[source_rank];
            ++ source_rank;
        }
        const auto local_slot = token_idx - source_begin;
        EP_DEVICE_ASSERT(0 <= local_slot and local_slot < kNumMaxTokensPerRank);
        const auto token = recv_buffer.get_rank_buffer(source_rank).get_token_buffer(local_slot);
        int origin;
        if constexpr (kCachedMode) {
            // Cached tokens keep the global source ID in the dispatch handle.
            origin = cached_recv_src_metadata[static_cast<int64_t>(token_idx) * (kNumTopk + 2) + kNumTopk];
        } else {
            origin = *token.template get_src_token_global_idx_ptr<__gm__ int>();
        }
        EP_DEVICE_ASSERT(source_rank * kNumMaxTokensPerRank <= origin and
                         origin < (source_rank + 1) * kNumMaxTokensPerRank);
        const auto row = origin_to_row[origin];
        EP_DEVICE_ASSERT(0 <= row and row < capacity_rows);

        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        asc_copy_gm2ub_align(
            hidden, const_cast<__gm__ uint8_t*>(ll_x) + static_cast<int64_t>(row) * kNumHiddenBytes,
            1, kNumHiddenBytes, 0, 0, false,
            asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 0, 0);
        asc_sync_notify(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        asc_copy_ub2gm_align(
            token.template get_hidden_ptr<__gm__ uint8_t>(), hidden,
            1, kNumHiddenBytes, asc_store_l2_cache_mode::NOTALLOC_CLEAN, 0, 0);
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    asc_sync_notify(PIPE_MTE2, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_S, EVENT_ID0);
}

}  // namespace deep_ep
