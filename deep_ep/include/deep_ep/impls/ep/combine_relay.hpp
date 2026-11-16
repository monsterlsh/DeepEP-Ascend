#pragma once

#include <deep_ep/impls/ep/combine_layout.hpp>
#include <deep_ep/impls/ep/combine_utils.hpp>
#include <deep_ep/layout/common/signals.hpp>

namespace deep_ep {

template <bool kDomainRelay>
__aicore__ __forceinline__ int combine_group_rank(__gm__ const int* server_ids, int rank) {
    if constexpr (kDomainRelay) {
        const int server = server_ids[rank];
        for (int peer = 0; peer < rank; ++peer)
            if (server_ids[peer] == server)
                return peer;
    }
    return rank;
}

// Every expert rank publishes one local row per source token it received. The
// master is the last top-k lane served by that rank, or -1 for an absent row.
template <int kNumRanks, int kCapacity, int kTokenBytes>
struct CombineRelayScratch {
    __gm__ uint8_t* partials;
    __gm__ int* masters;

    __aicore__ explicit CombineRelayScratch(__gm__ void* base)
        : partials(static_cast<__gm__ uint8_t*>(base)),
          masters(base ? reinterpret_cast<__gm__ int*>(partials + static_cast<int64_t>(kNumRanks) * kCapacity * kTokenBytes)
                       : nullptr) {}

    template <int kNumVecCores>
    __aicore__ void clear_masters(int core) const {
        for (int token = core; token < kNumRanks * kCapacity; token += kNumVecCores)
            AscendC::WriteGmByPassDCache(masters + token, -1);
    }
};

template <int kNumRanks, int kCapacity, int kHidden, int kNumTopk, int kNumUbBytes,
          int kNumVecCores, bool kHasTopkWeights, typename element_t>
__aicore__ __forceinline__ void relay_combine_partials(
        __gm__ const layout::CommonSignals& signals, __gm__ const int* server_ids,
        __gm__ void* scratch_base, __gm__ uint8_t* recv_buffer, int rank, int core, __ubuf__ void* ub_base) {
    constexpr int kWeightBytes = get_combine_weight_bytes(kNumTopk);
    constexpr int kTokenBytes = kHidden * sizeof(element_t) + kWeightBytes;
    constexpr int kSlots = get_combine_hidden_slots(kNumUbBytes, kTokenBytes);
    EP_STATIC_ASSERT(kSlots >= math::min(kNumRanks, kNumTopk));
    using Reducer = RingReducer<kHidden, kNumTopk, kSlots, 0, kWeightBytes, element_t>;
    auto ub = static_cast<__ubuf__ typename Reducer::Buffer*>(ub_base);
    Reducer reducer(ub);
    CombineRelayScratch<kNumRanks, kCapacity, kTokenBytes> scratch(scratch_base);

    int num_members = 0, local_index = 0;
    int members[kNumRanks];
    for (int peer = 0; peer < kNumRanks; ++peer) {
        if (server_ids[peer] == server_ids[rank]) {
            if (peer == rank)
                local_index = num_members;
            members[num_members++] = peer;
        }
    }
    // Each domain elects exactly one relay per source token, spread over its ranks.
    for (int token = local_index + core * num_members; token < kNumRanks * kCapacity;
         token += kNumVecCores * num_members) {
        const int first_input = reducer.slot_idx;
        int num_inputs = 0, master = -1;
        for (int i = 0; i < num_members; ++i) {
            const int peer = members[i];
            const int peer_master = AscendC::ReadGmByPassDCache(layout::get_sym_ptr(signals, scratch.masters + token, peer));
            if (peer_master < 0)
                continue;
            master = math::max(master, peer_master);
            auto row = layout::get_sym_ptr(signals, scratch.partials + static_cast<int64_t>(token) * kTokenBytes, peer);
            reducer.load_next_token(reinterpret_cast<__gm__ element_t*>(row), asc_load_l2_cache_mode::NOTALLOC_KEEP,
                                    kHasTopkWeights ? kWeightBytes : 0);
            ++num_inputs;
        }
        if (num_inputs == 0)
            continue;
        // Each valid rank owns at least one distinct top-k slot for this token.
        EP_DEVICE_ASSERT(num_inputs <= kNumTopk);
        reducer.reduce(num_inputs);
        const auto last = reducer.slot_idx == 0 ? kSlots - 1 : reducer.slot_idx - 1;
        if constexpr (kHasTopkWeights) {
            restore_combine_weights<kNumTopk, Reducer::kNumTokenElems, kSlots, kHidden>(
                reinterpret_cast<__ubuf__ uint32_t*>(ub->buf[0]),
                reinterpret_cast<__ubuf__ uint32_t*>(ub->buf[last] + kHidden), first_input, num_inputs);
            asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID0);
            asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID0);
            asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
            asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
        }
        const int owner = token / kCapacity;
        // The regular buffer has min(K, R) groups. For R < K each server owns
        // the group of its lowest rank, even if that rank handled no route.
        const int group = kNumRanks < kNumTopk ? members[0] : master;
        auto dst = layout::get_sym_ptr(signals, recv_buffer, owner) +
            (static_cast<int64_t>(group) * kCapacity + token % kCapacity) * kTokenBytes;
        asc_lock(PIPE_MTE3, last);
        asc_copy_ub2gm_align(dst, reinterpret_cast<__ubuf__ uint8_t*>(ub->buf[last]), 1,
                            kHasTopkWeights ? kTokenBytes : kHidden * sizeof(element_t),
                            asc_store_l2_cache_mode::NOTALLOC_CLEAN, 0, 0);
        asc_unlock(PIPE_MTE3, last);
    }
}

}  // namespace deep_ep
