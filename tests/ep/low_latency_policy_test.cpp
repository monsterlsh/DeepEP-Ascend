// Host-only policy checks. No Torch, CANN or NPU dependencies.
// c++ -std=c++17 -I. tests/ep/low_latency_policy_test.cpp -o /tmp/ep-policy
#include <cassert>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <csrc/kernels/ep/low_latency_policy.hpp>

using deep_ep::mte_bs_threshold_override;
using deep_ep::mte_threshold_bs;
using deep_ep::prefer_low_latency_transport;
using deep_ep::prefer_low_latency_transport_runtime;

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
    assert(not prefer_low_latency_transport(false, 3510, 8, 16, 256, 7168, 8, 128, 64, false, 190 * 1024 - 1));
    assert(not prefer_low_latency_transport(false, 3500, 8, 16, 256, 7168, 8, 128, 64, false));
    assert(not prefer_low_latency_transport(false, 3510, 8, 16, 256, 4096, 8, 128, 64, false));
    assert(not prefer_low_latency_transport(false, 3510, 8, 16, 256, 7168, 6, 128, 64, false));
    assert(not prefer_low_latency_transport(false, 3510, 8, 16, 256, 7168, 8, 256, 64, false));
    assert(not prefer_low_latency_transport(false, 3510, 8, 16, 256, 7168, 8, 128, 32, false));
    assert(not prefer_low_latency_transport(false, 3510, 8, 16, 256, 7168, 8, 128, 64, true));
    assert(not prefer_low_latency_transport(false, 3510, 8, 0, 256, 7168, 8, 128, 64, false));
    assert(not prefer_low_latency_transport(false, 3510, 8, 16, 15, 7168, 8, 128, 64, false));
    assert(not prefer_low_latency_transport(false, 3510, 8, 16, 257, 7168, 8, 128, 64, false));
}

static void test_decision() {
    // EP8 dispatch (threshold 100).
    assert(prefer_low_latency_transport(false, 3510, 8, 64, 256, 7168, 8, 128, 64, false));
    assert(prefer_low_latency_transport(false, 3510, 8, 100, 256, 7168, 8, 128, 64, false));
    assert(not prefer_low_latency_transport(false, 3510, 8, 101, 256, 7168, 8, 128, 64, false));
    // EP16 dispatch (threshold 77).
    assert(prefer_low_latency_transport(false, 3510, 16, 77, 256, 7168, 8, 128, 64, false));
    assert(not prefer_low_latency_transport(false, 3510, 16, 78, 256, 7168, 8, 128, 64, false));
    // EP8 combine (threshold 60).
    assert(prefer_low_latency_transport(true, 3510, 8, 60, 256, 7168, 8, 128, 64, false));
    assert(not prefer_low_latency_transport(true, 3510, 8, 61, 256, 7168, 8, 128, 64, false));
    // EP16 combine (threshold 17).
    assert(prefer_low_latency_transport(true, 3510, 16, 17, 256, 7168, 8, 128, 64, false));
    assert(not prefer_low_latency_transport(true, 3510, 16, 18, 256, 7168, 8, 128, 64, false));
    // EP128 dispatch and combine (historical 32).
    assert(prefer_low_latency_transport(false, 3510, 128, 32, 256, 7168, 8, 128, 64, false));
    assert(not prefer_low_latency_transport(false, 3510, 128, 33, 256, 7168, 8, 128, 64, false));
    assert(prefer_low_latency_transport(true, 3510, 128, 32, 256, 7168, 8, 128, 64, false));
    assert(not prefer_low_latency_transport(true, 3510, 128, 33, 256, 7168, 8, 128, 64, false));
    // Large batches never MTE.
    for (int bs : {512, 1024, 2048, 4096})
        assert(not prefer_low_latency_transport(false, 3510, 8, bs, bs, 7168, 8, 128, 64, false));
}

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

static void test_runtime_override() {
    // Invalid overrides use the original threshold, not a partially parsed value.
    const char* defaults[] = {nullptr, "", "-1", "-2", "100tail", "2147483648"};
    for (const auto value: defaults) {
        set_threshold_override(value);
        assert(prefer_low_latency_transport_runtime(false, 3510, 8, 100, 256, 7168, 8, 128, 64, false));
        assert(not prefer_low_latency_transport_runtime(false, 3510, 8, 101, 256, 7168, 8, 128, 64, false));
    }

    set_threshold_override("128");
    assert(prefer_low_latency_transport_runtime(false, 3510, 8, 101, 256, 7168, 8, 128, 64, false));
    assert(prefer_low_latency_transport_runtime(false, 3510, 8, 128, 256, 7168, 8, 128, 64, false));
    assert(not prefer_low_latency_transport_runtime(false, 3510, 8, 129, 256, 7168, 8, 128, 64, false));

    // Even the largest valid override cannot bypass any hardware/shape guard.
    set_threshold_override("2147483647");
    for (const bool is_combine: {false, true}) {
        assert(prefer_low_latency_transport_runtime(is_combine, 3510, 8, 256, 256, 7168, 8, 128, 64, false));
        assert(not prefer_low_latency_transport_runtime(is_combine, 3500, 8, 16, 256, 7168, 8, 128, 64, false));
        assert(not prefer_low_latency_transport_runtime(is_combine, 3510, 8, 16, 256, 4096, 8, 128, 64, false));
        assert(not prefer_low_latency_transport_runtime(is_combine, 3510, 8, 16, 256, 7168, 6, 128, 64, false));
        assert(not prefer_low_latency_transport_runtime(is_combine, 3510, 8, 16, 256, 7168, 8, 256, 64, false));
        assert(not prefer_low_latency_transport_runtime(is_combine, 3510, 8, 16, 256, 7168, 8, 128, 32, false));
        assert(not prefer_low_latency_transport_runtime(is_combine, 3510, 8, 16, 256, 7168, 8, 128, 64, true));
        assert(not prefer_low_latency_transport_runtime(is_combine, 3510, 8, 16, 256, 7168, 8, 128, 64, false, 190 * 1024 - 1));
        assert(not prefer_low_latency_transport_runtime(is_combine, 3510, 8, 0, 256, 7168, 8, 128, 64, false));
        assert(not prefer_low_latency_transport_runtime(is_combine, 3510, 8, -1, 256, 7168, 8, 128, 64, false));
        assert(not prefer_low_latency_transport_runtime(is_combine, 3510, 8, 16, 15, 7168, 8, 128, 64, false));
        assert(not prefer_low_latency_transport_runtime(is_combine, 3510, 8, 16, 257, 7168, 8, 128, 64, false));
        assert(not prefer_low_latency_transport_runtime(is_combine, 3510, 8, 4096, 4096, 7168, 8, 128, 64, false));
    }

    // An override cannot enable an EP/operation disabled by the policy table.
    // Exercise zero (disable), a smaller threshold and thresholds above capacity.
    for (const auto value: {"0", "32", "256", "4096"}) {
        set_threshold_override(value);
        const auto override = mte_bs_threshold_override();
        for (const int world: {1, 2, 4, 8, 16, 32, 56, 64, 96, 128, 256}) {
            for (const bool is_combine: {false, true}) {
                for (int bs = 0; bs <= 257; ++ bs) {
                    const auto expected = mte_threshold_bs(is_combine, world) > 0 and
                                          bs > 0 and bs <= 256 and bs <= override;
                    assert(prefer_low_latency_transport_runtime(is_combine, 3510, world, bs, 256,
                                                                 7168, 8, 128, 64, false) == expected);
                }
            }
        }
    }
    set_threshold_override(nullptr);
}

static void test_high_throughput_fallbacks() {
    // Large capacities keep URMA even when the actual local batch is small or
    // the diagnostic threshold override requests MTE for every batch size.
    for (const auto value: std::initializer_list<const char*>{nullptr, "0", "2147483647"}) {
        set_threshold_override(value);
        for (const int world: {8, 16, 32, 64, 128}) {
            for (const bool is_combine: {false, true}) {
                for (const int capacity: {257, 512, 1024, 2048, 4096}) {
                    for (const int bs: {1, 16, 256, capacity}) {
                        assert(not prefer_low_latency_transport_runtime(is_combine, 3510, world, bs, capacity,
                                                                         7168, 8, 128, 64, false));
                    }
                }
                assert(not prefer_low_latency_transport_runtime(is_combine, 3510, world, 16, 256,
                                                                 7168, 8, 128, 64, true));
            }
        }
    }
    set_threshold_override(nullptr);
}

int main() {
    test_thresholds();
    test_hard_guards();
    test_decision();
    test_override_parsing();
    test_runtime_override();
    test_high_throughput_fallbacks();
    // Exhaustive sweep: decision == (BS <= fixed threshold) under the hard
    // guards and B <= capacity <= 256.
    const int worlds[] = {1, 2, 4, 8, 16, 32, 56, 64, 96, 128, 256};
    for (const int world: worlds) {
        for (const bool is_combine: {false, true}) {
            const int thr = mte_threshold_bs(is_combine, world);
            for (int bs = 0; bs <= 257; ++ bs) {
                const bool hard_ok = bs >= 1 and bs <= 256;
                const bool expected = hard_ok and thr > 0 and bs <= thr;
                assert(prefer_low_latency_transport(is_combine, 3510, world, bs, 256, 7168, 8, 128, 64, false) == expected);
            }
        }
    }
    return 0;
}
