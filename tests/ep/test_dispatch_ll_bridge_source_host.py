"""CPU-only execution of source-extracted bridge logic, with DMA/event stubs.

This validates byte/slot mapping and rejects malformed provenance. It does not
validate Ascend compilation, memory ordering, DMA semantics or SHMEM transport.
"""

import re
import unittest

from _host_test_helpers import ROOT, run_host_cpp

FILES = [
    "deep_ep/include/deep_ep/common/compiled.hpp",
    "deep_ep/include/deep_ep/common/math.hpp",
    "deep_ep/include/deep_ep/layout/ep/token.hpp",
    "deep_ep/include/deep_ep/impls/ep/dispatch_ll_bridge.hpp",
]


def body(path):
    return re.sub(r"^#(?:include|pragma).*\n", "", (ROOT / path).read_text(), flags=re.M)


def host_test_source():
    prefix = r'''
    #include <algorithm>
    #include <cassert>
    #include <climits>
    #include <cstdint>
    #include <cstring>
    #include <iostream>
    #include <stdexcept>
    #include <vector>
    #define __launch_bounds__(...)
    #define EP_STATIC_ASSERT(cond, ...) static_assert(cond)
    #define EP_DEVICE_ASSERT(cond) do { if (!(cond)) throw std::runtime_error(#cond); } while (0)
    struct { int x = 0; } threadIdx;
    struct dim3 { dim3(int, int, int) {} };
    namespace AscendC { int block = 0; void InitSocState() {} int GetBlockIdx() { return block; } }
    namespace __asc_simt_vf { void __trap() { throw std::runtime_error("trap"); } }
    void asc_threadfence() {}
    void asc_syncthreads() {}
    template<class T> void asc_stcg(T* p, T v) { *p = v; }
    int asc_atomic_max(int* p, int v) { const int old = *p; *p = std::max(old, v); return old; }
    template<auto F, class... Args> void asc_vf_call(dim3, Args... args) { F(args...); }
    enum { PIPE_V, PIPE_S, PIPE_MTE2, PIPE_MTE3, EVENT_ID0 };
    void asc_sync_notify(int, int, int) {}
    void asc_sync_wait(int, int, int) {}
    namespace asc_load_l2_cache_mode { constexpr int NORMAL_FIRST_VICTIM = 0; }
    namespace asc_store_l2_cache_mode { constexpr int NOTALLOC_CLEAN = 0; }
    void asc_copy_gm2ub_align(uint8_t* dst, const uint8_t* src, int bursts, int bytes,
                            int, int, bool, int, int, int) { assert(bursts == 1); std::memcpy(dst, src, bytes); }
    void asc_copy_ub2gm_align(uint8_t* dst, const uint8_t* src, int bursts, int bytes,
                            int, int, int) { assert(bursts == 1); std::memcpy(dst, src, bytes); }
    alignas(32) uint8_t test_ub[4096];
    '''
    source = prefix + "\n".join(body(p) for p in FILES)
    # UB address zero is an Ascend address space; CPU execution uses a real array.
    source = source.replace("reinterpret_cast<__ubuf__ uint8_t*>(0)", "test_ub")
    source += r'''
    using namespace deep_ep;
    int passed = 0;

    template<bool Cached, bool Empty = false> void check_mapping() {
        constexpr int P = 4, M = 5, K = 4, H = 32, SF = 2, V = 3;
        const int psum[P] = {Empty ? 0 : 2, Empty ? 0 : 2, Empty ? 0 : 5, Empty ? 0 : 6};
        const int origins[6] = {3, 0, 14, 11, 12, 16};
        const int ll_ids[9] = {11, 3, 16, 14, 0, 12, 11, 0, 3};
        const int64_t counts[2] = {Empty ? 0 : 4, Empty ? 0 : 5};
        const auto tl = layout::TokenLayout(H, SF * sizeof(sf_pack_t), K, !Cached);
        std::vector<uint8_t> buffer(P * M * tl.get_num_bytes(), 0xcd);
        const auto native = layout::BufferLayout(tl, P, M, buffer.data());
        std::vector<int> cached_meta(6 * (K + 2), -1);
        std::vector<uint8_t> ll_x(9 * H);
        for (int row = 0; row < 9; ++row)
            for (int byte = 0; byte < H; ++byte)
                ll_x[row * H + byte] = uint8_t(ll_ids[row] * 7 + byte);
        int source_rank = 0, source_begin = 0;
        for (int idx = 0; idx < psum[P - 1]; ++idx) {
            while (idx >= psum[source_rank]) source_begin = psum[source_rank++];
            auto token = native.get_rank_buffer(source_rank).get_token_buffer(idx - source_begin);
            if constexpr (!Cached) *token.get_src_token_global_idx_ptr<int>() = origins[idx];
            cached_meta[idx * (K + 2) + K] = origins[idx];
        }
        const auto before = buffer;
        int map[P * M];
        std::fill(map, map + P * M, 1234567);
        dispatch_ll_build_origin_map<P * M, 2, 1>(ll_ids, counts, map, 9);
        if constexpr (Empty) for (int v : map) assert(v == -1);
        else {
            assert(map[3] == 8 && map[0] == 7 && map[11] == 6);
            assert(map[1] == -1 && map[19] == -1);
        }
        for (AscendC::block = 0; AscendC::block < V; ++AscendC::block)
            dispatch_ll_copy_payload_impl<P, M, K, H, SF, V, 4096, Cached>(
                ll_x.data(), map, psum, Cached ? cached_meta.data() : nullptr, buffer.data(), 9);
        auto expected = before;
        source_rank = 0; source_begin = 0;
        for (int idx = 0; idx < psum[P - 1]; ++idx) {
            while (idx >= psum[source_rank]) source_begin = psum[source_rank++];
            const auto physical_slot = source_rank * M + idx - source_begin;
            for (int byte = 0; byte < H; ++byte)
                expected[physical_slot * tl.get_num_bytes() + byte] = uint8_t(origins[idx] * 7 + byte);
        }
        assert(buffer == expected);  // also proves SF, weights, routing bytes and holes unchanged
        ++passed;
    }

    template<class F> void expect_error(F fn) {
        bool caught = false;
        try { fn(); } catch (const std::runtime_error&) { caught = true; }
        assert(caught); ++passed;
    }

    int main() {
        check_mapping<false>(); check_mapping<true>();
        check_mapping<false, true>(); check_mapping<true, true>();
        int map[20], ids[2] = {0, 20}; int64_t counts[2] = {1, 1};
        expect_error([&] { dispatch_ll_build_origin_map<20, 2, 1>(ids, counts, map, 2); });
        ids[1] = -1;
        expect_error([&] { dispatch_ll_build_origin_map<20, 2, 1>(ids, counts, map, 2); });
        counts[1] = -1;
        expect_error([&] { dispatch_ll_build_origin_map<20, 2, 1>(ids, counts, map, 2); });
        counts[1] = 2;
        expect_error([&] { dispatch_ll_build_origin_map<20, 2, 1>(ids, counts, map, 2); });
        const int psum[4] = {1, 1, 1, 1};
        int cached[6] = {-1, -1, -1, -1, 0, -1};
        uint8_t x[32] = {}, buffer[4 * 5 * 96] = {};
        std::fill(map, map + 20, -1); AscendC::block = 0;
        expect_error([&] { dispatch_ll_copy_payload_impl<4, 5, 4, 32, 2, 3, 4096, true>(x, map, psum, cached, buffer, 1); });
        cached[4] = 5;
        expect_error([&] { dispatch_ll_copy_payload_impl<4, 5, 4, 32, 2, 3, 4096, true>(x, map, psum, cached, buffer, 1); });
        std::cout << "PASS " << passed << " CPU bridge source cases\n";
    }
    '''
    return source


class DispatchLLBridgeSourceTests(unittest.TestCase):
    def test_source_execution(self):
        output = run_host_cpp(host_test_source())
        self.assertIn('PASS 10 CPU bridge source cases', output)


if __name__ == '__main__':
    unittest.main()
