"""Exercise the combine relay over simulated symmetric peer windows."""

import unittest

from _host_test_helpers import ROOT, run_host_cpp


class CombineRelayTests(unittest.TestCase):
    def test_masked_weighted_partials_and_reuse(self):
        # Only the device SIMD, DMA and barriers are simulated. The production
        # code clears and inspects masters, chooses relays and addresses peers.
        utils = (ROOT / 'deep_ep/include/deep_ep/impls/ep/combine_utils.hpp').read_text()
        restore = utils[utils.index('template <int kNumTopk, int kNumTokenElems'):
                        utils.index('template <uint32_t kNumHiddenElems')]
        primitives = r'''
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deep_ep/common/compiled.hpp>
#define EP_DEVICE_ASSERT assert
#define EP_STATIC_ASSERT static_assert
namespace AscendC {
int ReadGmByPassDCache(int* x) { return *x; }
void WriteGmByPassDCache(int* x, int value) { *x = value; }
}
enum { PIPE_V, PIPE_MTE2, PIPE_MTE3, EVENT_ID0 };
enum class asc_load_l2_cache_mode { NOTALLOC_KEEP };
enum class asc_store_l2_cache_mode { NOTALLOC_CLEAN };
void asc_sync_notify(int,int,int) {}
void asc_sync_wait(int,int,int) {}
void asc_lock(int,int) {}
void asc_unlock(int,int) {}
int writes;
void asc_copy_ub2gm_align(uint8_t* dst, uint8_t* src, int, int len, asc_store_l2_cache_mode, int,int) {
    std::memcpy(dst, src, len); ++writes;
}
using vector_uint32_t = std::array<uint32_t, 64>;
int asc_update_mask_b32(uint32_t& n) { return n; }
void asc_duplicate_scalar(vector_uint32_t& dst, uint32_t value, int n) { std::fill_n(dst.begin(),n,value); }
void asc_loadalign_brc_v2(vector_uint32_t& dst, uint32_t* src) {
    for (int i=0; i<64; ++i) dst[i]=src[i%8];
}
void asc_or(vector_uint32_t& dst, const vector_uint32_t& a, const vector_uint32_t& b, int n) {
    for (int i=0; i<n; ++i) dst[i]=a[i]|b[i];
}
void asc_storealign(uint32_t* dst, const vector_uint32_t& src, int n) { std::copy_n(src.begin(),n,dst); }
namespace deep_ep {
template <int H, int K, int S, int, int W, typename T> struct RingReducer {
    static constexpr int kNumTokenElems = H+W/sizeof(T);
    struct Buffer { T buf[S][kNumTokenElems]; };
    Buffer* ub;
    int slot_idx = 0;
    explicit RingReducer(Buffer* buffer) : ub(buffer) {}
    void load_next_token(T* ptr, asc_load_l2_cache_mode, int extra) {
        std::memcpy(ub->buf[slot_idx], ptr, H*sizeof(T)+extra);
        slot_idx = (slot_idx+1)%S;
    }
    void reduce(int count) {
        for (int h=0; h<H; ++h) {
            int sum=0;
            for (int i=1; i<=count; ++i) sum += ub->buf[(slot_idx+S-i)%S][h];
            ub->buf[(slot_idx+S-1)%S][h] = sum;
        }
    }
};
''' + restore + '}'
        run_host_cpp(r'''
#include <array>
#include <deep_ep/impls/ep/combine_relay.hpp>
using namespace deep_ep;
constexpr int R=6, M=5, H=16, max_k=19, cores=3, ub_bytes=768;
constexpr int max_bytes=H*2+get_combine_weight_bytes(max_k);
struct Window {
    layout::CommonSignals signals{};
    alignas(32) uint8_t scratch[R*M*(max_bytes+4)];
    alignas(32) uint8_t received[max_k*M*max_bytes];
} mem[R];

template <int K>
void run(const int* servers, int round) {
    constexpr int bytes=H*2+get_combine_weight_bytes(K);
    struct Expected { int sum=0; uint32_t weights[K]{}; };
    Expected expected[R][M][K]{};
    for (int r=0; r<R; ++r) {
        std::memset(mem[r].received, 0x5a, sizeof(mem[r].received));
        CombineRelayScratch<R,M,bytes> scratch(mem[r].scratch);
        for (int c=0;c<cores;++c) scratch.template clear_masters<cores>(c);
        for (int p=0;p<R;++p) mem[r].signals.peer_ptrs[p]=reinterpret_cast<int64_t>(&mem[p].signals);
    }
    for (int r=0;r<R;++r) for (int t=0;t<(r+round)%(M+1);++t) {
        int routes[K]={0,2,0,3};
        if (t%2) { routes[0]=1; routes[1]=-1; routes[2]=5; routes[3]=3; }
        if ((t+r+round)%4==0) std::fill(routes, routes+K, -1);
        const int global=r*M+t;
        for (int p=0;p<R;++p) {
            CombineRelayScratch<R,M,bytes> scratch(mem[p].scratch);
            auto row=scratch.partials+global*bytes;
            std::memset(row, 0, bytes);
            int partial=0, master=-1;
            for (int k=0;k<K;++k) {
                if (routes[k]!=p) continue;
                partial += 19 + global + k; // already expert-weighted output
                auto w=std::bit_cast<uint32_t>(-2.125f*(k+1)-global);
                reinterpret_cast<uint32_t*>(row+H*2)[k]=w;
                master=k;
                int domain_master=k;
                for (int j=k+1;j<K;++j)
                    if (routes[j]>=0 && servers[routes[j]]==servers[p]) domain_master=j;
                if constexpr (R<K) {
                    domain_master=0;
                    while (servers[domain_master]!=servers[p]) ++domain_master;
                    assert(combine_group_rank<true>(servers,p)==domain_master);
                }
                expected[r][t][domain_master].weights[k]=w;
                expected[r][t][domain_master].sum+=19+global+k;
            }
            std::fill_n(reinterpret_cast<int16_t*>(row), H, partial);
            scratch.masters[global]=master;
        }
    }
    writes=0;
    for (int r=0;r<R;++r) for (int c=0;c<cores;++c) {
        alignas(32) uint8_t ub[ub_bytes]{};
        relay_combine_partials<R,M,H,K,ub_bytes,cores,true,int16_t>(
            mem[r].signals, servers, mem[r].scratch, mem[r].received, r,c,ub);
    }
    int expected_writes=0;
    for (int r=0;r<R;++r) for (int t=0;t<M;++t) for (int k=0;k<K;++k) {
        const auto row=mem[r].received+(k*M+t)*bytes;
        const auto& e=expected[r][t][k];
        if (!e.sum) {
            assert(std::all_of(row,row+bytes,[](uint8_t v){ return v==0x5a; }));
            continue;
        }
        ++expected_writes;
        for (int h=0;h<H;++h) assert(reinterpret_cast<const int16_t*>(row)[h]==e.sum);
        assert(std::memcmp(row+H*2,e.weights,K*4)==0);
    }
    assert(writes==expected_writes && writes>0);
}
int main() {
    int interleaved[R]={19,8,19,8,19,4}, single[R]{};
    for (int round=0;round<3;++round) {
        run<4>(interleaved, round);
        run<4>(single, round);
        run<7>(interleaved, round);
        run<7>(single, round);
        run<19>(interleaved, round);
    }
}
''', include_dirs=(ROOT / 'deep_ep/include',),
                     headers={'deep_ep/impls/ep/combine_utils.hpp': primitives})


if __name__ == '__main__':
    unittest.main()
