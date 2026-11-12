#include <cassert>
#include <climits>
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

using deep_ep::mte_bs_threshold_override;

static void set_threshold_override(const char* value) {
#ifdef _WIN32
    const auto status = _putenv_s("EP_MTE_BS_THRESHOLD", value == nullptr ? "" : value);
#else
    const auto status = value == nullptr ? unsetenv("EP_MTE_BS_THRESHOLD") :
                                          setenv("EP_MTE_BS_THRESHOLD", value, 1);
#endif
    assert(status == 0);
}

static void test_override_parsing() {
    struct Case {
        const char* value;
        int expected;
    };
    const Case cases[] = {
        {nullptr, -1}, {"", -1}, {"-1", -1}, {"0", 0}, {"16", 16}, {"0016", 16},
        {"256", 256}, {"2147483647", std::numeric_limits<int>::max()},
        {"-2", -1}, {"invalid", -1}, {"16tail", -1}, {" 16", -1}, {"16 ", -1},
        {"+16", -1}, {"0x10", -1}, {"1e2", -1}, {"2147483648", -1},
        {"99999999999999999999999999999999999999", -1},
    };
    for (const auto& test: cases) {
        set_threshold_override(test.value);
        assert(mte_bs_threshold_override() == test.expected);
    }
}


int main() {
    using namespace deep_ep;
    test_override_parsing();
    set_threshold_override(nullptr);
    assert(mte_bs_threshold_override() == -1);
    for (int ep : {1, 2, 4, 8, 16, 24, 32, 64, 128}) {
        for (int m = -1; m <= 513; ++m) {
            const bool in_grid = (ep == 8 || ep == 16 || ep == 32) &&
                (m == 1 || m == 2 || m == 4 || m == 8 || m == 16 || m == 32 || m == 64 || m == 128 || m == 256);
            assert(is_decode_capacity(ep, m) == in_grid);
            assert(is_decode_shape(ep, m, 7168, 8, 128, 64, 221184, false) == in_grid);
            const auto plan = get_dispatch_low_latency_plan(3510, ep, m, 7168, 8, 128, 64, 221184, false);
            assert(bool(plan) == in_grid);
            if (plan)
                assert(plan->buffer_bytes == int64_t(ep) * m * (14336 + 128));
            for (int limit : {-1, 0, 16, 60, INT_MAX}) {
                const bool shmem = in_grid && ((ep == 8 && m <= 32) || (ep == 16 && m <= 16));
                assert(combine_shmem_capacity(ep, m, limit) == (shmem && (limit < 0 || m <= limit)));
                for (int rank = 0; rank < ep; ++rank) {
                    const int tokens = m > rank ? m - rank : 1;
                    const bool direct = (ep == 8 && m == 64 && rank >= 4) || (ep == 16 && m == 32 && rank == 15);
                    assert(combine_native_mte_rank(ep, m, rank, tokens, limit) == (direct && (limit < 0 || tokens <= limit)));
                    assert(!combine_native_mte_rank(ep, m, rank, tokens + 1, limit));
                }
            }
        }
    }

    const int thresholds[][5] = {{8,256,256,16,8}, {16,32,16,16,0}, {32,64,32,16,0},
                                  {1,0,0,0,0}, {24,0,0,0,0}, {64,0,0,0,0}, {128,0,0,0,0}};
    for (const auto& [ep, fresh_prologue, cached_prologue, fresh_epilogue, cached_epilogue] : thresholds) {
        assert(get_dispatch_low_latency_threshold(ep, false, true) == fresh_prologue);
        assert(get_dispatch_low_latency_threshold(ep, true, true) == cached_prologue);
        assert(get_dispatch_low_latency_threshold(ep, false, false) == fresh_epilogue);
        assert(get_dispatch_low_latency_threshold(ep, true, false) == cached_epilogue);
    }
    const auto plan = [](int arch, int h, int k, int e, int cores, int ub, bool fp8) {
        return get_dispatch_low_latency_plan(arch, 8, 64, h, k, e, cores, ub, fp8);
    };
    assert(!plan(2201,7168,8,128,64,221184,false));
    assert(!plan(3510,4096,8,128,64,221184,false));
    assert(!plan(3510,7168,4,128,64,221184,false));
    assert(!plan(3510,7168,8,256,64,221184,false));
    assert(!plan(3510,7168,8,128,32,221184,false));
    assert(!plan(3510,7168,8,128,64,221183,false));
    assert(!plan(3510,7168,8,128,64,221185,false));
    assert(!plan(3510,7168,8,128,64,221184,true));
    check_workspace_size<128, 8, 1, 8>();
    check_workspace_size<128, 32, 4, 8>();
    const auto routing = get_dispatch_routing_workspace_bytes(8, 128, 4, 8);
    assert(!dispatch_direct_ub_fits(routing, 4, 14336, 8, routing + 14336 + 4 * 128 - 1, false));
    assert(dispatch_direct_ub_fits(routing, 4, 14336, 8, routing + 14336 + 4 * 128, false));
    assert(dispatch_direct_ub_fits(routing, 4, 14336, 8, routing + 14336 + 4 * 32, true));
}
