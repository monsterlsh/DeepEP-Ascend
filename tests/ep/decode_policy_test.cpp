#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>

#include "csrc/kernels/ep/dispatch_low_latency_policy.hpp"

namespace {

struct alignas(8) PeerInfoSize {
    uint64_t words[6];
};
struct alignas(8) JettyInfoSize {
    uint64_t addresses[8];
    uint32_t words[6];
};

template <int E, int R, int T, int K>
struct alignas(32) RoutingWorkspaceSize {
    int expert_histogram[E], rank_histogram[R], rank_psum[R];
    int sgep_idx[T * K], sgep_to_entry[T * K + R * 6];
    int dst_rank_idx[T * K], dst_slot_idx[T * K];
    int lsqe_counter, psqe_counter, lsqe_base_idx[R];
    int local_copy_ready, metadata_ready, local_copy_dst_slot_idx[T];
    JettyInfoSize jetty_info;
    PeerInfoSize peer_info[R];
};

template <int E, int R, int T, int K>
void check_workspace_size() {
    static_assert(deep_ep::get_dispatch_routing_workspace_bytes(R, E, T, K) == sizeof(RoutingWorkspaceSize<E, R, T, K>));
}

}  // namespace

int main() {
    using namespace deep_ep;
    for (int ep = -1; ep <= 256; ++ep) {
        for (int m = -1; m <= 513; ++m) {
            const bool in_grid = (ep == 2 || ep == 4 || ep == 8 || ep == 16 ||
                                  ep == 32 || ep == 64 || ep == 128) && m >= 1 && m <= 16;
            assert(use_decode_implementation(m) == (m >= 1 && m <= 16));
            assert(is_decode_capacity(ep, m) == in_grid);
            assert(is_decode_shape(ep, m, 7168, 8, 128, 64, 221184, false) == in_grid);
            assert(supports_dispatch_mte(3510, ep, m, 7168, 8, 128, 64, 221184, 0) == in_grid);
        }
    }
    const auto plan = [](int arch, int h, int k, int e, int cores, int ub, int sf_words) {
        return supports_dispatch_mte(arch, 8, 16, h, k, e, cores, ub, sf_words);
    };
    assert(!plan(2201, 7168, 8, 128, 64, 221184, 0));
    assert(plan(3510, 4096, 8, 128, 64, 221184, 0));
    assert(plan(3510, 7168, 4, 128, 64, 221184, 0));
    assert(plan(3510, 7168, 8, 256, 64, 221184, 0));
    assert(!plan(3510, 7168, 8, 128, 32, 221184, 0));
    assert(!plan(3510, 7168, 8, 128, 64, 221183, 0));
    assert(!plan(3510, 7168, 8, 128, 64, 221185, 0));
    for (int ranks : {8, 16, 32, 64, 128}) {
        assert(supports_dispatch_mte(3510, ranks, 16, 7168, 6, 384, 64, 221184, 112));
    }
    assert(plan(3510, 384, 8, 128, 64, 221184, 0));
    assert(!plan(3510, 384, 8, 128, 64, 221184, 6));  // FP8 zero padding is one 256-byte vector
    assert(!plan(3510, 7169, 8, 128, 64, 221184, 0));
    assert(!plan(3510, 0, 8, 128, 64, 221184, 0));
    assert(!plan(3510, 7168, 0, 128, 64, 221184, 0));
    assert(!plan(3510, 4096, 9, 8, 64, 221184, 0));
    assert(!plan(3510, 4096, 8, 130, 64, 221184, 0));
    assert(!plan(3510, 4096, 8, 520, 64, 221184, 0));
    assert(!plan(3510, 4096, 8, 128, 64, 221184, -1));
    assert(!plan(3510, 4096, 8, 128, 64, 221184, std::numeric_limits<int>::max()));
    assert(!plan(3510, 4096, 8, 128, 64, 221184, 27000));
    // 32 routes fit the issue ring for a small hidden size. Same shape with
    // EP32 needs 34 reduction slots, exceeding the 32 available mutex IDs.
    assert(plan(3510, 2048, 32, 512, 64, 221184, 0));
    assert(!plan(3510, 4096, 32, 512, 64, 221184, 0));
    assert(!is_decode_shape(32, 16, 2048, 32, 512, 64, 221184, false));
    assert(!plan(3510, 2048, 33, 512, 64, 221184, 0));
    assert(!plan(3510, std::numeric_limits<int>::max() - 127, 8, 128, 64, 221184, 0));
    check_workspace_size<128, 2, 1, 8>();
    check_workspace_size<128, 128, 4, 8>();
    check_workspace_size<384, 64, 4, 6>();
    // EP128, E384, K6: each token can select at most three local experts.
    constexpr auto rows = get_dispatch_max_expert_rows(128, 256);
    static_assert(get_dispatch_max_expanded_rows(rows, 6, 3, 128) == 98688);
    static_assert(get_dispatch_max_expanded_rows(rows, 2, 3, 1) == rows * 2);
    static_assert(get_dispatch_max_expert_rows(128, 16384) == 2097152);
    static_assert(get_dispatch_max_expert_rows(128, 131071) <= reduction::kReductionMaxLowCount);
    static_assert(get_dispatch_max_expert_rows(128, 131072) > reduction::kReductionMaxLowCount);
    static_assert(get_dispatch_max_expanded_rows(1, 1, 2, 1 << 30) > std::numeric_limits<int>::max());
    const auto routing = get_dispatch_routing_workspace_bytes(8, 128, 4, 8);
    const auto stages = [](int hidden_bytes, int sf_bytes, int64_t ub_bytes, bool cached) {
        return get_dispatch_mte_stages(8, 128, 4, hidden_bytes, sf_bytes, 8, ub_bytes, cached);
    };
    assert(!stages(14336, 0, routing + 14336 + 4 * 128 - 1, false).fits());
    assert(stages(14336, 0, routing + 14336 + 4 * 128, false).fits());
    assert(stages(14336, 0, routing + 14336 + 4 * 32, true).fits());
    assert(!stages(7168, 224, routing + 7168 + 4 * (128 + 224) - 1, false).fits());
    assert(stages(7168, 224, routing + 7168 + 4 * (128 + 224), false).fits());
}
