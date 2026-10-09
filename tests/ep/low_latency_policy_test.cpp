// Host-only policy checks. No Torch, CANN or NPU dependencies.
// c++ -std=c++17 -I. tests/ep/low_latency_policy_test.cpp -o /tmp/ep-policy
// Compilation and execution are intentionally deferred until validation resumes.
#include <cassert>
#include <csrc/kernels/ep/low_latency_policy.hpp>

using deep_ep::prefer_low_latency_transport;

static_assert(prefer_low_latency_transport(3510, 8, 64, 256, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 8, 65, 256, 7168, 8, 128, 64, false));
static_assert(prefer_low_latency_transport(3510, 128, 32, 256, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 128, 33, 256, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 8, 0, 256, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 8, 16, 15, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 8, 16, 257, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 8, 16, 256, 7168, 8, 128, 64, false, 190 * 1024 - 1));
static_assert(prefer_low_latency_transport(3510, 8, 16, 256, 7168, 8, 128, 64, false, 190 * 1024));
static_assert(!prefer_low_latency_transport(3500, 8, 16, 256, 7168, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 8, 16, 256, 4096, 8, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 8, 16, 256, 7168, 6, 128, 64, false));
static_assert(!prefer_low_latency_transport(3510, 8, 16, 256, 7168, 8, 256, 64, false));
static_assert(!prefer_low_latency_transport(3510, 8, 16, 256, 7168, 8, 128, 32, false));
static_assert(!prefer_low_latency_transport(3510, 8, 16, 256, 7168, 8, 128, 64, true));

int main() {
    const int worlds[] = {4, 8, 16, 32, 64, 128, 256};
    for (const int world : worlds) {
        const int limit = world == 8 ? 64 : (world >= 16 && world <= 128 ? 32 : 0);
        for (int bs = 0; bs <= 257; ++bs) {
            assert(prefer_low_latency_transport(3510, world, bs, 256, 7168, 8, 128, 64, false)
                   == (bs > 0 && bs <= limit));
        }
    }
    // Different local B values may choose different transports, so protocol
    // interoperability is a separate multi-rank correctness requirement.
    assert(prefer_low_latency_transport(3510, 8, 2, 256, 7168, 8, 128, 64, false));
    assert(!prefer_low_latency_transport(3510, 8, 128, 256, 7168, 8, 128, 64, false));
    return 0;
}
