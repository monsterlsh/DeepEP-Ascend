#include <cassert>
#include <climits>
#include <cstdint>
#include <iostream>

#include "csrc/kernels/ep/dispatch_low_latency_policy.hpp"

namespace {

struct alignas(8) PeerInfoSize { uint64_t words[6]; };
struct alignas(8) JettyInfoSize { uint64_t addresses[8]; uint32_t words[6]; };

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
    static_assert(deep_ep::get_dispatch_routing_workspace_bytes(R, E, T, K) == sizeof(RoutingWorkspaceSize<E,R,T,K>));
}

}  // namespace

int main() {
    using deep_ep::get_dispatch_low_latency_plan;
    const auto plan = [](int p, int m, int h, int k, int e, int v = 64, int ub = 190 * 1024) {
        return get_dispatch_low_latency_plan(3510, p, m, h, k, e, v, ub, false);
    };
    check_workspace_size<128,8,1,8>();
    check_workspace_size<512,128,64,32>();
    check_workspace_size<255,255,3,1>();
    check_workspace_size<1,1,1025,1>();

    for (int p: {1,2,4,8,16,32,64,128}) {
        for (int v: {16,32,64}) {
            assert(plan(p, 1, 7168, 8, 128, v));
            assert(plan(p, 256, 7168, 8, 128, v));
        }
        assert(plan(p, 4096, 7168, 8, 128));
    }
    assert(plan(255, 1, 512, 1, 255));
    assert(plan(3, 257, 512, 4, 12));
    assert(plan(1, 65537, 16, 1, 1));
    assert(plan(8, 257, 512, 1, 8));
    assert(plan(8, 512, 1024, 2, 16));
    assert(plan(16, 64, 3584, 16, 512));
    assert(plan(8, 1, 7168, 8, 128, 32, 189 * 1024));

    const auto reference = plan(8, 256, 7168, 8, 128);
    assert(reference && reference->buffer_bytes == int64_t(8) * 256 * (14336 + 128));
    int first_accepted = 0;
    for (int ub = 1; ub <= 190 * 1024; ++ub) {
        if (plan(8, 256, 7168, 8, 128, 64, ub)) {
            first_accepted = ub;
            break;
        }
    }
    assert(first_accepted > 0);
    assert(!plan(8, 256, 7168, 8, 128, 64, first_accepted - 1));
    assert(plan(8, 256, 7168, 8, 128, 64, first_accepted));
    const auto routing = deep_ep::get_dispatch_routing_workspace_bytes(8,128,4,8);
    assert(!deep_ep::dispatch_direct_ub_fits(routing,4,14336,8,routing + 14336 + 4 * 128 - 1,false));
    assert(deep_ep::dispatch_direct_ub_fits(routing,4,14336,8,routing + 14336 + 4 * 128,false));
    assert(deep_ep::dispatch_direct_ub_fits(routing,4,14336,8,routing + 14336 + 4 * 32,true));

    assert(!plan(0, 1, 7168, 8, 128));
    assert(!plan(256, 1, 512, 1, 512));
    assert(!plan(8, 0, 7168, 8, 128));
    assert(!plan(8, -1, 7168, 8, 128));
    assert(!plan(8, 1, 7169, 8, 128));
    assert(!plan(8, 1, 0, 8, 128));
    assert(!plan(8, 1, 7168, 0, 128));
    assert(!plan(8, 1, 7168, 33, 128));
    assert(!plan(16, 1, 7168, 8, 127));
    assert(!plan(8, 1, 7168, 8, 0));
    assert(!plan(16, 1, 3584, 16, 896));
    assert(!plan(8, 1, 7168, 8, 128, 15));
    assert(!plan(8, 1, 7168, 8, 128, 65));
    assert(!plan(8, 1, 7168, 8, 128, 64, 0));
    assert(!plan(8, 1, 7168, 8, 128, 64, 1));
    assert(!plan(1, 1, INT_MAX - 15, 1, 1, 64, INT_MAX));
    assert(!plan(1, 1, (INT_MAX / 2) & ~15, 32, 1, 64, INT_MAX));
    assert(!plan(1, (1 << 24), 16, 1, 1, 64, INT_MAX));
    assert(plan(1, (1 << 24) - 1, 16, 1, 1, 64, INT_MAX));
    assert(!plan(128, (1 << 24) / 128, 16, 1, 128, 64, INT_MAX));
    // The native template still constrains its logical WQEBB index, including
    // the direct specialization which does not issue URMA SQEs.
    assert(!plan(128, 131000, 16, 32, 128, 16, INT_MAX));
    assert(!get_dispatch_low_latency_plan(3510, 8, 32, 7168, 8, 128, 64, 190 * 1024, true));
    assert(!get_dispatch_low_latency_plan(2201, 8, 32, 7168, 8, 128, 64, 190 * 1024, false));
    assert(deep_ep::prefer_dispatch_low_latency(8,4096,7168,8,128));
    std::cout << "native-layout dispatch admission and UB boundaries passed\n";
}
