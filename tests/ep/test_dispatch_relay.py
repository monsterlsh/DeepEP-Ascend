"""Run both production relay stages with peer windows and synchronous CPU copies."""

import unittest

from _host_test_helpers import ROOT, run_host_cpp


class DispatchRelayTests(unittest.TestCase):
    def test_sparse_routes_with_interleaved_servers_and_unequal_slots(self):
        source = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <deep_ep/layout/ep/token.hpp>

namespace AscendC { inline int ReadGmByPassDCache(int* value) { return *value; } }
inline void asc_stcg(int* p, int value) { *p = value; }
enum class asc_store_l2_cache_mode { NOTALLOC_CLEAN };
enum class asc_load_l2_cache_mode { NOTALLOC_CLEAN };
enum { PIPE_MTE2, PIPE_MTE3, EVENT_ID0 };
void asc_sync_notify(int, int, int) {}
void asc_sync_wait(int, int, int) {}
int copies = 0;
void asc_copy_gm2ub_align(uint8_t* ub, uint8_t* gm, int, uint32_t bytes,
                          int, int, bool, asc_load_l2_cache_mode, int, int) {
    std::memcpy(ub, gm, bytes); ++copies;
}
void asc_copy_ub2gm_align(uint8_t* gm, uint8_t* ub, int, uint32_t bytes, asc_store_l2_cache_mode, int, int) {
    std::memcpy(gm, ub, bytes);
}
#include <deep_ep/impls/ep/dispatch_relay.hpp>
using namespace deep_ep;
constexpr int R = 6, M = 3, K = 6, max_packet = 256;
struct Memory {
    layout::CommonSignals signals{};
    int descriptors[R*M];
    uint8_t data[R*M*max_packet];
} memory[R];
struct Routing { int dst_rank_idx[M*K], dst_slot_idx[M*K]; };

void round_trip(bool cached, int scale_bytes, bool same_server) {
    const int interleaved[R] = {8, 19, 8, 19, 8, 19};
    const int flat[R]{};
    const int* servers = same_server ? flat : interleaved;
    const layout::TokenLayout packet(64, scale_bytes, K, !cached);
    const auto stride = packet.get_num_bytes();
    assert(stride <= max_packet);
    const uint8_t fill = 0xee;
    int counts[R][R]{}, expected_token[R][R][M]{};
    int forwarded = 0;
    for (auto& peer : memory) {
        std::memset(peer.data, fill, sizeof(peer.data));
        std::fill(std::begin(peer.descriptors), std::end(peer.descriptors), 777);
        for (int r = 0; r < R; ++r)
            peer.signals.peer_ptrs[r] = reinterpret_cast<int64_t>(&memory[r].signals);
    }

    // Rank zero's first token gives different receive offsets to later relays.
    // Other ranks own zero, one or three tokens. Masked, local-only, multiple
    // remote destinations and stale descriptors in unused capacity are covered.
    for (int source = 0; source < R; ++source) {
        Routing plan;
        std::fill(std::begin(plan.dst_rank_idx), std::end(plan.dst_rank_idx), -1);
        const int tokens = source % 3 == 2 ? 0 : source % 3 == 1 ? 1 : M;
        for (int t = 0; t < tokens; ++t) {
            int destinations[M][K] = {{-1,3,-1,-1,-1,-1}, {1,0,3,-1,5,4}, {2,5,1,3,-1,4}};
            bool contacted[R]{};
            for (int k = 0; k < K; ++k) {
                int dst = destinations[t][k];
                if (dst < 0) continue;
                int slot = counts[dst][source]++;
                plan.dst_rank_idx[t*K+k] = dst;
                plan.dst_slot_idx[t*K+k] = slot;
                expected_token[dst][source][slot] = t;
                if (servers[dst] != servers[source]) {
                    bool prior = false;
                    for (int rank = 0; rank < R; ++rank)
                        prior |= contacted[rank] && servers[rank] == servers[dst];
                    forwarded += prior;
                }
                contacted[dst] = true;
            }
            publish_dispatch_relay_slots<K,M>(&plan, memory[source].signals, servers, memory[source].descriptors, source, t);
            for (int k = 0; k < K; ++k) {
                int dst = plan.dst_rank_idx[t*K+k];
                if (dst < 0) continue;
                auto out = memory[dst].data + (source*M + plan.dst_slot_idx[t*K+k]) * stride;
                for (int i = 0; i < stride; ++i) out[i] = static_cast<uint8_t>(i+source*31+t*7+cached);
            }
        }
    }
    copies = 0;
    for (int dst = 0; dst < R; ++dst) {
        int psum[R]{};
        for (int src = 0; src < R; ++src)
            psum[src] = counts[dst][src] + (src ? psum[src-1] : 0);
        for (int core = 0; core < 2; ++core) {
            uint8_t ub[max_packet];
            gather_dispatch_relay_slots<R,M,2>(layout::BufferLayout(packet, R, M, memory[dst].data), memory[dst].signals,
                                               memory[dst].descriptors, psum, core, ub);
        }
        for (int src = 0; src < R; ++src)
            for (int slot = 0; slot < M; ++slot)
                for (int i = 0; i < stride; ++i) {
                    auto expected = slot >= counts[dst][src] ? fill :
                        static_cast<uint8_t>(i+src*31+expected_token[dst][src][slot]*7+cached);
                    assert(memory[dst].data[(src*M+slot)*stride+i] == expected);
                }
    }
    assert(copies == forwarded && (same_server ? !copies : copies > 0));
}

int main() {
    for (bool cached : {false, true})
        for (int scales : {0, 36})
            for (bool one_server : {true, false}) round_trip(cached, scales, one_server);
}
'''
        run_host_cpp(source, include_dirs=(ROOT / 'deep_ep/include',),
                     headers={'utils/debug/asc_assert.h': '#include <cassert>\n'})
