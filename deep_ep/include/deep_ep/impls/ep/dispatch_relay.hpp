#pragma once

#include <deep_ep/layout/common/signals.hpp>
#include <deep_ep/layout/ep/token.hpp>

namespace deep_ep {

// Each thread owns a whole token. Walk backwards so earlier lanes remain intact
// while looking for the first destination rank on each remote server. The
// handle has already saved the complete rank-to-slot mapping for cached calls.
template <int kNumTopk, int kCapacity, typename routing_t>
__simt_callee__ void publish_dispatch_relay_slots(
    __ubuf__ routing_t* routing, __gm__ const layout::CommonSignals& signals,
    __gm__ const int* server_ids, __gm__ int* relay_slots, int source_rank,
    int local_token) {
    const int first = local_token * kNumTopk;
    for (int k = kNumTopk - 1; k >= 0; --k) {
        const int entry = first + k;
        const int dst = routing->dst_rank_idx[entry];
        if (dst < 0)
            continue;

        int relay = -1;
        if (server_ids[dst] != server_ids[source_rank]) {
            for (int earlier = first; earlier < entry; ++earlier) {
                const int peer = routing->dst_rank_idx[earlier];
                if (peer >= 0 and server_ids[peer] == server_ids[dst]) {
                    relay = peer * kCapacity + routing->dst_slot_idx[earlier];
                    routing->dst_rank_idx[entry] = -1;
                    break;
                }
            }
        }
        // Every valid receive slot gets a descriptor, including direct slots.
        // Unused tail slots are never read, so no scratch clear is necessary.
        const int index = source_rank * kCapacity + routing->dst_slot_idx[entry];
        asc_stcg(layout::get_sym_ptr(signals, relay_slots + index, dst), relay);
    }
}

// The first recipient retains its native packet in place. Other recipients on
// that server fetch it locally after publication. The packet includes scales
// and the appropriate fresh/cached metadata, so the expand epilogue is shared.
template <int kNumRanks, int kCapacity, int kNumVecCores>
__aicore__ __forceinline__ void gather_dispatch_relay_slots(
    const layout::BufferLayout& received, const __gm__ layout::CommonSignals& signals,
    __gm__ int* relay_slots, __gm__ int* recv_rank_psum, int vec_core_idx,
    __ubuf__ uint8_t* packet) {
    const auto packet_bytes = static_cast<uint32_t>(received.get_num_bytes_per_token());
    int prefix = 0;
    for (int source = 0; source < kNumRanks; ++source) {
        const int next = AscendC::ReadGmByPassDCache(recv_rank_psum + source);
        const auto source_buffer = received.get_rank_buffer(source);
        for (int slot = vec_core_idx; slot < next - prefix; slot += kNumVecCores) {
            const int relay = AscendC::ReadGmByPassDCache(relay_slots + source * kCapacity + slot);
            if (relay < 0)
                continue;
            auto relay_addr = source_buffer.get_token_buffer(relay % kCapacity).template get_base_ptr<__gm__ uint8_t>();
            auto remote = layout::get_sym_ptr(signals, relay_addr, relay / kCapacity);
            auto local = source_buffer.get_token_buffer(slot).template get_base_ptr<__gm__ uint8_t>();
            asc_copy_gm2ub_align(packet, remote, 1, packet_bytes, 0, 0, false,
                                 asc_load_l2_cache_mode::NOTALLOC_CLEAN, 0, 0);
            asc_sync_notify(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            asc_sync_wait(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            asc_copy_ub2gm_align(local, packet, 1, packet_bytes,
                                 asc_store_l2_cache_mode::NOTALLOC_CLEAN, 0, 0);
            asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        prefix = next;
    }
}

}  // namespace deep_ep
