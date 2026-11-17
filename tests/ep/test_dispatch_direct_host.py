"""CPU checks of native MTE peer addressing and token layout.

Routing examples are independent oracles, not execution of the SIMT planner.
Device memory ordering requires a multi-rank NPU test.
"""

import unittest

from _host_test_helpers import ROOT, run_host_cpp


def run_layout_check(source):
    run_host_cpp(source, include_dirs=(ROOT / 'deep_ep/include',),
                 headers={'utils/debug/asc_assert.h': '#pragma once\n#include <cassert>\n'})


class DirectDispatchHostTests(unittest.TestCase):
    def test_direct_metadata_fresh_and_cached_at_capacity_16(self):
        device = (ROOT / 'deep_ep/include/deep_ep/impls/ep/decode/dispatch.hpp').read_text()
        # Compile the production allocation, stage bounds and pointer expression.
        # Kernel execution and pipeline events still require device validation.
        layout = device[device.index('template <typename ub_layout_t,'):
                        device.index('\ntemplate <int kNumMaxTokensPerRank,')]
        direct = device[device.index('constexpr auto kNumMTEStages ='):]
        bounds = direct[:direct.index('        for (int stage_idx =')]
        pointer = direct[direct.index('const auto ub_metadata ='):direct.index('            const auto event_idx =')]
        resident_start = device.index('constexpr bool kKeepMetadataInUB =')
        resident = device[resident_start:device.index('    const auto ub =', resident_start)]
        source = r'''
#include <cassert>
#include <cstring>
#include <deep_ep/impls/ep/dispatch_direct_layout.hpp>
#include <deep_ep/layout/ep/token.hpp>
using namespace deep_ep;
''' + layout + r'''
template<int SFPacks, bool Cached>
void check() {
    constexpr int kNumTopk = 8, kNumSFPacks = SFPacks;
    constexpr int kNumMaxTokensPerVecCore = math::ceil_div(16, 64);
    constexpr bool kLowLatencyTransport = true;
    constexpr int hidden_bytes = SFPacks ? 7168 : 2 * 7168;
    struct alignas(32) Workspace {
        uint8_t bytes[get_dispatch_routing_workspace_bytes(16, 128, kNumMaxTokensPerVecCore, kNumTopk)];
    };
    using ub_layout_with_buffer_t = UBLayoutWithBuffer<Workspace, hidden_bytes, SFPacks, kNumTopk,
                                                       kNumMaxTokensPerVecCore, 216 * 1024, Cached>;
    static_assert(sizeof(ub_layout_with_buffer_t) <= 216 * 1024);
    ub_layout_with_buffer_t allocation;
    std::memset(&allocation, 0xcd, sizeof(allocation));
    auto ub = &allocation;
    constexpr auto kNumHiddenMTEStages = ub_layout_with_buffer_t::kNumHiddenMTEStages;
    constexpr auto kNumMetadataMTEStages = ub_layout_with_buffer_t::kNumMetadataMTEStages;
    constexpr auto kNumTokensPerMetadataMTE = ub_layout_with_buffer_t::kNumTokensPerMetadataMTE;
''' + resident + bounds + r'''
    static_assert(kNumMTEStages == 1);
    static_assert(kKeepMetadataInUB == (SFPacks == 0));
    const auto native_layout = layout::TokenLayout(0, SFPacks * sizeof(sf_pack_t), kNumTopk, !Cached);
    const auto base = reinterpret_cast<uint8_t*>(ub->metadata);
    const int i = 0, stage_idx = 0;
''' + pointer + r'''
    assert(ub_metadata == base);
    assert(reinterpret_cast<uintptr_t>(ub_metadata) % 32 == 0);
    assert(kMetadataBytes == native_layout.get_num_bytes());
    std::memset(ub_metadata, 1, kMetadataBytes);
    for (size_t byte = 0; byte < sizeof(ub->metadata); ++byte)
        assert(base[byte] == (byte < kMetadataBytes ? 1 : 0xcd));
    for (size_t byte = 0; byte < sizeof(ub->hidden); ++byte)
        assert(reinterpret_cast<uint8_t*>(ub->hidden)[byte] == 0xcd);
}
int main() {
    check<0, false>();
    check<0, true>();
    check<112, false>();
    check<112, true>();
}
'''
        run_layout_check(source)

    def test_native_addresses_and_cached_stride(self):
        device = (ROOT / 'deep_ep/include/deep_ep/impls/ep/decode/dispatch.hpp').read_text()
        begin = device.index('    const auto get_remote_token =')
        mapping = device[begin:device.index('    };', begin) + len('    };')]
        source = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <deep_ep/impls/ep/dispatch_direct_layout.hpp>
#include <deep_ep/layout/common/signals.hpp>
#include <deep_ep/layout/ep/token.hpp>

using namespace deep_ep;

void boundaries() {
    for (int ranks : {1, 3, 8, 128, 255})
        for (int M : {1, 5, 4096, 65537})
            for (int K : {1, 4, 8, 31, 32})
                for (int sf : {0, 2, 34})
                    for (bool cached : {false, true}) {
                        const auto token = layout::TokenLayout(256, sf, K, !cached);
                        const auto align32 = [](int n) { return 32 * ((n + 31) / 32); };
                        const auto metadata = K * 4 + (cached ? 0 : align32(K * 8) + 4);
                        const auto bytes = 256 + align32(sf) + align32(metadata);
                        assert(bytes == token.get_num_bytes());
                        const auto buffer = layout::BufferLayout(token, ranks, M, nullptr);
                        assert(buffer.get_num_bytes() == int64_t(ranks) * M * bytes);
                    }
}

struct Workspace {
    layout::CommonSignals signals{};
    const auto& get_common_signals() const { return signals; }
};

void routed_examples(bool cached, bool reverse_order) {
    constexpr int P = 4, M = 5, K = 4, H = 32, E = 16;
    const int counts[P] = {3, 0, 5, 1};
    const int64_t routes[P][M][K] = {
        {{0,1,4,-1}, {8,10,-1,9}, {-1,-1,-1,-1}},
        {},
        {{0,4,8,12}, {0,1,2,3}, {-1,7,-1,-1}, {15,14,13,12}, {-1,-1,-1,-1}},
        {{3,5,11,13}}
    };
    // Expected counts are tokens per destination/source, with same-rank routes
    // sharing one payload while each original expert entry retains its weight.
    const int expected_counts[P][P] = {{1,0,2,1}, {1,0,2,1}, {1,0,1,1}, {0,0,2,1}};
    const auto token_layout = layout::TokenLayout(H, 0, K, !cached);
    const auto stride = token_layout.get_num_bytes();
    // Keep a nonzero offset between the symmetric signal base and the receive window.
    struct Region { Workspace workspace; alignas(32) uint8_t data[P * M * 256]; };
    Region peers[P]{};
    assert(P * M * stride <= static_cast<int64_t>(sizeof(peers[0].data)));
    for (auto& peer : peers) {
        std::memset(peer.data, 0xcd, sizeof(peer.data));
        for (int i = 0; i < P; ++i)
            peer.workspace.signals.peer_ptrs[i] = reinterpret_cast<int64_t>(&peers[i].workspace.signals);
    }
    std::vector<std::vector<int>> origins(P, std::vector<int>(P * M, -1));
    int sent[P][P] = {};
    for (int src = 0; src < P; ++src) {
        const auto& workspace_layout = peers[src].workspace;
        auto recv_buffer = layout::BufferLayout(token_layout, P, M, peers[src].data).get_rank_buffer(src);
''' + mapping + r'''
        for (int i = 0; i < counts[src]; ++i) {
            const int t = reverse_order ? counts[src] - i - 1 : i;
            bool targets[P] = {};
            for (int k = 0; k < K; ++k)
                if (routes[src][t][k] >= 0)
                    targets[routes[src][t][k] / (E / P)] = true;
            for (int dst = 0; dst < P; ++dst) {
                if (!targets[dst])
                    continue;
                const int slot = sent[dst][src]++;
                auto out = get_remote_token(dst, slot);
                assert(out == peers[dst].data + (src * M + slot) * stride);
                for (int byte = 0; byte < H; ++byte)
                    out[byte] = static_cast<uint8_t>(src * 37 + t * 11 + byte + (cached ? 71 : 0));
                auto token = token_layout;
                token.set_base_ptr(out);
                if (!cached) {
                    std::memcpy(token.get_topk_idx_ptr<>(), routes[src][t], sizeof(routes[src][t]));
                    *token.get_src_token_global_idx_ptr<>() = src * M + t;
                }
                for (int k = 0; k < K; ++k)
                    token.get_topk_weights_ptr<>()[k] = src * 16 + t + k * 0.125f + (cached ? 100.f : 0.f);
                origins[dst][src * M + slot] = src * M + t;
            }
        }
    }
    for (int dst = 0; dst < P; ++dst) {
        auto buffer = layout::BufferLayout(token_layout, P, M, peers[dst].data);
        for (int src = 0; src < P; ++src) {
            assert(sent[dst][src] == expected_counts[dst][src]);
            for (int slot = 0; slot < M; ++slot) {
                auto token = buffer.get_rank_buffer(src).get_token_buffer(slot);
                const int origin = origins[dst][src * M + slot];
                if (origin < 0) {
                    for (int64_t byte = 0; byte < stride; ++byte)
                        assert(token.get_base_ptr<>()[byte] == 0xcd);
                    continue;
                }
                assert(origin / M == src);
                const int t = origin % M;
                for (int byte = 0; byte < H; ++byte)
                    assert(token.get_hidden_ptr<>()[byte] ==
                           static_cast<uint8_t>(src * 37 + t * 11 + byte + (cached ? 71 : 0)));
                if (!cached) {
                    assert(*token.get_src_token_global_idx_ptr<>() == origin);
                    for (int k = 0; k < K; ++k)
                        assert(token.get_topk_idx_ptr<>()[k] == routes[src][t][k]);
                }
                for (int k = 0; k < K; ++k)
                    assert(token.get_topk_weights_ptr<>()[k] ==
                           src * 16 + t + k * 0.125f + (cached ? 100.f : 0.f));
            }
        }
    }
}

int main() {
    boundaries();
    for (bool cached : {false, true})
        for (bool reversed : {false, true})
            routed_examples(cached, reversed);
    std::cout << "native address, route holes, empty rank, masked entries and cached weight examples passed\n";
}
'''
        run_layout_check(source)


if __name__ == '__main__':
    unittest.main()
