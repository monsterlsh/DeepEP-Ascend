// Host-only policy checks. No Torch, CANN or NPU dependencies.
// c++ -std=c++17 -I. tests/ep/low_latency_policy_test.cpp -o /tmp/ep-policy
#include <cassert>
#include <initializer_list>
#include <csrc/kernels/ep/low_latency_policy.hpp>

using deep_ep::mte_threshold_bs;
using deep_ep::prefer_low_latency_transport;

// Fixed thresholds (G-p4 conservative values):
//   dispatch: EP8=100 EP16=77 EP32=70 EP64=67 EP128=32
//   combine : EP8=60  EP16=17 EP128=32  others 0
static void test_thresholds() {
    assert(mte_threshold_bs(false, 8) == 100);
    assert(mte_threshold_bs(false, 16) == 77);
    assert(mte_threshold_bs(false, 32) == 70);
    assert(mte_threshold_bs(false, 64) == 67);
    assert(mte_threshold_bs(false, 128) == 32);
    assert(mte_threshold_bs(true, 8) == 60);
    assert(mte_threshold_bs(true, 16) == 17);
    assert(mte_threshold_bs(true, 32) == 0);
    assert(mte_threshold_bs(true, 64) == 0);
    assert(mte_threshold_bs(true, 128) == 32);
    // EP not in the table -> 0 (never MTE).
    assert(mte_threshold_bs(false, 1) == 0);
    assert(mte_threshold_bs(false, 4) == 0);
    assert(mte_threshold_bs(false, 96) == 0);
    assert(mte_threshold_bs(false, 256) == 0);
    assert(mte_threshold_bs(true, 96) == 0);
}

static void test_hard_guards() {
    assert(prefer_low_latency_transport(false, 3510, 8, 16, 256, 7168, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(false, 3510, 8, 16, 256, 7168, 8, 128, 64, false, 190 * 1024 - 1));
    assert(!prefer_low_latency_transport(false, 3500, 8, 16, 256, 7168, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(false, 3510, 8, 16, 256, 4096, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(false, 3510, 8, 16, 256, 7168, 6, 128, 64, false));
    assert(!prefer_low_latency_transport(false, 3510, 8, 16, 256, 7168, 8, 256, 64, false));
    assert(!prefer_low_latency_transport(false, 3510, 8, 16, 256, 7168, 8, 128, 32, false));
    assert(!prefer_low_latency_transport(false, 3510, 8, 16, 256, 7168, 8, 128, 64, true));
    assert(!prefer_low_latency_transport(false, 3510, 8, 0, 256, 7168, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(false, 3510, 8, 16, 15, 7168, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(false, 3510, 8, 16, 257, 7168, 8, 128, 64, false));
}

static void test_decision() {
    // EP8 dispatch (threshold 100).
    assert(prefer_low_latency_transport(false, 3510, 8, 64, 256, 7168, 8, 128, 64, false));
    assert(prefer_low_latency_transport(false, 3510, 8, 100, 256, 7168, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(false, 3510, 8, 101, 256, 7168, 8, 128, 64, false));
    // EP16 dispatch (threshold 77).
    assert(prefer_low_latency_transport(false, 3510, 16, 77, 256, 7168, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(false, 3510, 16, 78, 256, 7168, 8, 128, 64, false));
    // EP8 combine (threshold 60).
    assert(prefer_low_latency_transport(true, 3510, 8, 60, 256, 7168, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(true, 3510, 8, 61, 256, 7168, 8, 128, 64, false));
    // EP16 combine (threshold 17).
    assert(prefer_low_latency_transport(true, 3510, 16, 17, 256, 7168, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(true, 3510, 16, 18, 256, 7168, 8, 128, 64, false));
    // EP128 dispatch and combine (historical 32).
    assert(prefer_low_latency_transport(false, 3510, 128, 32, 256, 7168, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(false, 3510, 128, 33, 256, 7168, 8, 128, 64, false));
    assert(prefer_low_latency_transport(true, 3510, 128, 32, 256, 7168, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(true, 3510, 128, 33, 256, 7168, 8, 128, 64, false));
    // Large batches never MTE.
    for (int bs : {512, 1024, 2048, 4096})
        assert(!prefer_low_latency_transport(false, 3510, 8, bs, bs, 7168, 8, 128, 64, false));
}

int main() {
    test_thresholds();
    test_hard_guards();
    test_decision();
    // Exhaustive sweep: decision == (BS <= fixed threshold) under the hard
    // guards and B <= capacity <= 256.
    const int worlds[] = {1, 2, 4, 8, 16, 32, 56, 64, 96, 128, 256};
    for (const int world : worlds) {
        for (const bool is_combine : {false, true}) {
            const int thr = mte_threshold_bs(is_combine, world);
            for (int bs = 0; bs <= 257; ++bs) {
                const bool hard_ok = bs >= 1 && bs <= 256;
                const bool expected = hard_ok && thr > 0 && bs <= thr;
                assert(prefer_low_latency_transport(is_combine, 3510, world, bs, 256, 7168, 8, 128, 64, false) == expected);
            }
        }
    }
    return 0;
}
