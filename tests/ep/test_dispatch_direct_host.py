"""CPU checks of production direct-address helpers against native TokenLayout.

Routing examples are independent oracles, not execution of the SIMT planner.
These checks do not establish device memory ordering or SHMEM completion.
"""

import unittest

from _host_test_helpers import ROOT, run_host_cpp


def run_layout_check(source):
    run_host_cpp(source, include_dirs=(ROOT / 'deep_ep/include',),
                 headers={'utils/debug/asc_assert.h': '#pragma once\n#include <cassert>\n'})


class DirectDispatchHostTests(unittest.TestCase):
    def test_direct_metadata_slots_and_capacity_fallback(self):
        device = (ROOT / 'deep_ep/include/deep_ep/impls/ep/dispatch.hpp').read_text()
        # Compile the production allocation, stage bounds and pointer expression.
        # Kernel execution and pipeline events still require device validation.
        layout = device[device.index('template <typename ub_layout_t,'):
                        device.index('\ntemplate <int kNumMaxTokensPerRank,')]
        direct = device[device.index('constexpr auto kNumMTEStages ='):]
        bounds = direct[:direct.index('        for (int stage_idx =')]
        pointer = direct[direct.index('const auto ub_metadata ='):direct.index('            const auto event_idx =')]
        resident_start = device.index('constexpr bool kKeepMetadataInUB =')
        resident = device[resident_start:device.index('    const auto ub =', resident_start)]
        staging = device[device.index('    const auto metadata_token_layout ='):
                         device.index('    // Barrier before dispatch')]
        source = r'''
#include <cassert>
#include <cstring>
#include <deep_ep/impls/ep/dispatch_direct_layout.hpp>
#include <deep_ep/layout/ep/token.hpp>
using namespace deep_ep;
''' + layout + r'''
template<int T, int K, bool Cached, int UB = 194560, int H = 14336, bool Direct = true,
         bool LayoutOnly = false, int Ranks = 8, int Capacity = T * 64>
void check(int expected_stages, bool expected_resident, bool expected_packed = false) {
    struct alignas(32) Workspace {
        uint8_t bytes[get_dispatch_routing_workspace_bytes(Ranks, 128, T, K)];
    };
    constexpr int kNumTopk = K, kNumSFPacks = 0, kNumMaxTokensPerVecCore = T;
    constexpr int kNumHiddenBytes = H, kNumRanks = Ranks, kNumMaxTokensPerRank = Capacity;
    constexpr bool kCachedMode = Cached, kLowLatencyTransport = Direct, kLayoutOnly = LayoutOnly;
    using Allocation = UBLayoutWithBuffer<Workspace, H, 0, K, T, UB, Cached>;
    using ub_layout_with_buffer_t = Allocation;
    static_assert(sizeof(Allocation) <= UB);
    Allocation allocation;
    std::memset(&allocation, 0xcd, sizeof(allocation));
    auto ub = &allocation;
    constexpr auto kNumHiddenMTEStages = Allocation::kNumHiddenMTEStages;
    constexpr auto kNumMetadataMTEStages = Allocation::kNumMetadataMTEStages;
    constexpr auto kNumTokensPerMetadataMTE = Allocation::kNumTokensPerMetadataMTE;
''' + resident + staging + bounds + r'''
    assert(kNumMTEStages == expected_stages);
    assert(kKeepMetadataInUB == expected_resident);
    assert(kPackDirectTokens == expected_packed);
    if constexpr (kPackDirectTokens) {
        const auto stride = layout::TokenLayout(H, 0, K, !Cached).get_num_bytes();
        const auto base = reinterpret_cast<uint8_t*>(&ub->hidden);
        assert(stride == kDirectTokenBytes && T * stride <= sizeof(ub->hidden));
        for (int i = 0; i < T; ++i) {
            const auto packet = ub_metadata_buffer.get_token_buffer(i);
            assert(packet.template get_hidden_ptr<>() == base + i * stride);
            assert(packet.template get_topk_idx_ptr<uint8_t>() == base + i * stride + H);
            assert(packet.template get_topk_weights_ptr<uint8_t>() ==
                   base + i * stride + H + (Cached ? 0 : math::align(K * 8, 32)));
        }
        return;
    }
    const auto native_layout = layout::TokenLayout(0, 0, K, !Cached);
    const auto base = reinterpret_cast<uint8_t*>(ub->metadata);
    const int slots = expected_resident ? T : expected_stages;
    for (int i = 0; i < slots; ++i) {
        const int stage_idx = i % kNumMTEStages;
''' + pointer + r'''
        assert(ub_metadata == base + i * native_layout.get_num_bytes());
        assert(reinterpret_cast<uintptr_t>(ub_metadata) % 32 == 0);
        std::memset(ub_metadata, i + 1, native_layout.get_num_bytes());
    }
    // Sentinel checks expose batch-stride addressing and overlapping token slots.
    for (size_t byte = 0; byte < sizeof(ub->metadata); ++byte) {
        const auto slot = byte / native_layout.get_num_bytes();
        assert(base[byte] == (slot < static_cast<size_t>(slots) ? slot + 1 : 0xcd));
    }
    for (size_t byte = 0; byte < sizeof(ub->hidden); ++byte)
        assert(reinterpret_cast<uint8_t*>(ub->hidden)[byte] == 0xcd);
}
int main() {
    check<1, 8, false>(1, true); check<1, 8, true>(1, true);
    check<4, 8, false>(4, true); check<4, 8, true>(4, true);
    check<64, 8, false>(8, true); check<64, 8, true>(8, true);
    check<65, 8, false>(8, true); check<65, 8, true>(8, true);
    // Four-token batches under a tight UB budget limit the metadata slots.
    check<8, 32, false, 8192, 32>(4, false);
    check<64, 8, false, 194560, 14336, false>(8, false);
    check<64, 8, true, 194560, 14336, true, true>(8, false);
    check<1, 8, false, 194560, 14336, true, false, 16, 31>(1, true);
    check<1, 8, false, 194560, 14336, true, false, 16, 32>(1, true, true);
    check<4, 8, true, 194560, 14336, true, false, 16, 256>(4, true, true);
    check<5, 8, true, 194560, 14336, true, false, 16, 257>(5, true);
    check<4, 8, false, 32768, 14336, true, false, 16, 256>(1, true);
}
'''
        run_layout_check(source)

    def test_native_addresses_and_cached_stride(self):
        source = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <deep_ep/impls/ep/dispatch_direct_layout.hpp>
#include <deep_ep/layout/ep/token.hpp>

using namespace deep_ep;

void boundaries() {
    for (int ranks : {1, 3, 8, 128, 255})
        for (int M : {1, 5, 4096, 65537})
            for (int K : {1, 4, 8, 31, 32})
                for (int sf : {0, 2, 34})
                    for (bool cached : {false, true}) {
                        const auto token = layout::TokenLayout(256, sf, K, !cached);
                        const auto bytes = get_dispatch_token_bytes(256, sf, K, cached);
                        assert(bytes == token.get_num_bytes());
                        for (int src : {0, ranks - 1})
                            for (int slot : {0, M - 1}) {
                                constexpr uint64_t base = 0x123456780000;
                                auto buffer = layout::BufferLayout(token, ranks, M,
                                    reinterpret_cast<void*>(base));
                                auto expected = buffer.get_rank_buffer(src).get_token_buffer(slot);
                                assert(dispatch_direct_token_address(base, src, M, slot, bytes) ==
                                       reinterpret_cast<uint64_t>(expected.get_hidden_ptr<>()));
                                const auto end = dispatch_direct_token_address(base, src, M, slot, bytes) + bytes;
                                assert(end <= base + static_cast<uint64_t>(buffer.get_num_bytes()));
                            }
                    }
    constexpr uint64_t base = 0x100000000;
    const auto fresh = get_dispatch_token_bytes(32, 0, 4, false);
    const auto cached = get_dispatch_token_bytes(32, 0, 4, true);
    assert(fresh == 96 && cached == 64);
    assert(dispatch_direct_token_address(base, 3, 5, 4, fresh) == base + 1824);
    assert(dispatch_direct_token_address(base, 3, 5, 4, cached) == base + 1216);
}

void routed_examples(bool cached, bool reverse_order) {
    constexpr int P = 4, M = 5, K = 4, H = 32, E = 16;
    const int counts[P] = {3, 0, 5, 1};
    const int64_t routes[P][M][K] = {
        {{0,1,4,-1}, {8,8,-1,9}, {-1,-1,-1,-1}},
        {},
        {{0,4,8,12}, {1,1,2,3}, {-1,7,-1,-1}, {15,14,13,12}, {-1,-1,-1,-1}},
        {{3,5,11,13}}
    };
    // Expected counts are tokens per destination/source, with same-rank routes
    // sharing one payload while each original expert entry retains its weight.
    const int expected_counts[P][P] = {{1,0,2,1}, {1,0,2,1}, {1,0,1,1}, {0,0,2,1}};
    const auto token_layout = layout::TokenLayout(H, 0, K, !cached);
    const auto stride = get_dispatch_token_bytes(H, 0, K, cached);
    const auto size = static_cast<size_t>(P * M * stride);
    std::vector<std::vector<uint8_t>> peer(P, std::vector<uint8_t>(size, 0xcd));
    std::vector<std::vector<int>> origins(P, std::vector<int>(P * M, -1));
    int sent[P][P] = {};
    for (int src = 0; src < P; ++src) {
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
                auto address = dispatch_direct_token_address(
                    reinterpret_cast<uint64_t>(peer[dst].data()), src, M, slot, stride);
                auto out = reinterpret_cast<uint8_t*>(address);
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
        auto buffer = layout::BufferLayout(token_layout, P, M, peer[dst].data());
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
