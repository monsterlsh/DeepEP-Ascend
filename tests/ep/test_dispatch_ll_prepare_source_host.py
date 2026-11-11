"""CPU source execution of the new padding/route-normalization device header."""

import re
import unittest

from _host_test_helpers import ROOT, run_host_cpp

HEADER = ROOT / "deep_ep/include/deep_ep/impls/ep/dispatch_low_latency_prepare.hpp"


def host_test_source():
    source = r'''
    #include <algorithm>
    #include <cassert>
    #include <cstdint>
    #include <cstring>
    #include <iostream>
    #include <vector>
    #include <deep_ep/common/math.hpp>
    using GM_ADDR = uint8_t*;
    constexpr int EVENT_ID0 = 0;
    namespace AscendC {
    int block = 0;
    void InitSocState() {}
    uint32_t GetBlockIdx() { return block; }
    uint32_t GetBlockNum() { return 64; }
    enum class HardEvent { MTE2_S, V_MTE3, S_MTE3, MTE3_MTE2, MTE3_S, MTE3_V };
    template<HardEvent> void SetFlag(int) {}
    template<HardEvent> void WaitFlag(int) {}
    enum { PIPE_ALL };
    template<int> void PipeBarrier() {}
    template<class T> struct LocalTensor {
        T* ptr;
        T GetValue(uint32_t i) const { return ptr[i]; }
        void SetValue(uint32_t i, T v) { ptr[i] = v; }
    };
    template<class T> struct GlobalTensor {
        T* ptr = nullptr;
        void SetGlobalBuffer(T* value) { ptr = value; }
        GlobalTensor operator[](uint64_t i) { return {ptr + i}; }
    };
    template<int = 0> struct TBuf {
        std::vector<uint8_t> bytes;
        template<class T> LocalTensor<T> Get() { return {reinterpret_cast<T*>(bytes.data())}; }
    };
    struct TPipe { template<int P> void InitBuffer(TBuf<P>& b, uint32_t n) { b.bytes.resize(n); } };
    struct DataCopyParams { uint16_t count; uint32_t bytes, source, destination, reserved; };
    template<class T> struct PadParams { bool enabled; uint8_t left, right; T value; };
    template<class T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyParams p, PadParams<T>) {
        assert(p.count == 1 && p.source == 0 && p.destination == 0); std::memcpy(dst.ptr, src.ptr, p.bytes);
    }
    template<class T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyParams p) {
        assert(p.count == 1 && p.source == 0 && p.destination == 0); std::memcpy(dst.ptr, src.ptr, p.bytes);
    }
    template<class T> void Duplicate(LocalTensor<T> out, T value, uint32_t n) { std::fill(out.ptr, out.ptr + n, value); }
    }
    '''
    source += re.sub(r"^#(?:include|pragma).*\n", "", HEADER.read_text(), flags=re.M)
    source += r'''
    template<int M> void check(int B) {
        constexpr int H = 16, K = 4, E = 8;
        std::vector<uint16_t> input(B * H), output(M * H, 0xffff);
        std::vector<int64_t> routes(B * K);
        const int64_t patterns[3][K] = {{3,3,-1,1}, {-1,-1,0,7}, {2,4,2,4}};
        for (int t=0;t<B;++t) {
            for(int j=0;j<H;++j) input[t*H+j] = uint16_t(t*H+j+1);
            for(int k=0;k<K;++k) routes[t*K+k] = patterns[t%3][k];
        }
        const auto before = routes;
        std::vector<int32_t> wire(M * K, 9999);
        std::vector<float> weights(M * K, 123.0f);
        for (AscendC::block = 0; AscendC::block < 64; ++AscendC::block)
            deep_ep::dispatch_low_latency_prepare<H,K,M,E>(
                reinterpret_cast<uint8_t*>(input.data()), reinterpret_cast<uint8_t*>(routes.data()),
                reinterpret_cast<uint8_t*>(output.data()), reinterpret_cast<uint8_t*>(wire.data()),
                reinterpret_cast<uint8_t*>(weights.data()), B);
        assert(routes == before);
        for (int t=0;t<M;++t) {
            for(int j=0;j<H;++j) assert(output[t*H+j] == (t < B ? input[t*H+j] : 0));
            for(int k=0;k<K;++k) {
                int64_t expected = t < B ? routes[t*K+k] : -1;
                if(t<B) for(int prev=0;prev<k;++prev) if(routes[t*K+prev] == expected) { expected=-1; break; }
                assert(wire[t*K+k] == expected);
                assert(weights[t*K+k] == 0.0f);
            }
        }
    }
    int main() {
        check<7>(3); check<7>(7); check<7>(0); check<130>(2); check<130>(130);
        std::cout << "PASS 5 CPU prepare source cases\n";
    }
    '''
    return source


class DispatchLLPrepareSourceTests(unittest.TestCase):
    def test_source_execution(self):
        output = run_host_cpp(host_test_source(), include_dirs=(ROOT / 'deep_ep/include',))
        self.assertIn('PASS 5 CPU prepare source cases', output)


if __name__ == '__main__':
    unittest.main()
