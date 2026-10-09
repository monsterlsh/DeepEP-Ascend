#pragma once

#include <cstdlib>
#include <cstring>

namespace deep_ep {

// Candidate region for the compatible MTE transport. Historical measurements
// used the standalone fused LL kernels, so this region must be re-profiled after
// integration; it is not a measured speedup guarantee for this implementation.
// The wire layout, allocation, routing metadata and collective sequence remain
// those of EPBuffer. Peers may consequently use different transports when their
// actual token counts differ, without introducing another agreement collective.
constexpr bool prefer_low_latency_transport(int arch, int num_ranks, int num_tokens,
                                            int num_max_tokens_per_rank, int hidden,
                                            int num_topk, int num_experts, int num_vec_cores,
                                            bool use_fp8, int num_ub_bytes = 190 * 1024) {
    if (arch != 3510 || use_fp8 || hidden != 7168 || num_topk != 8 || num_experts != 128 ||
        num_vec_cores != 64 || num_ub_bytes < 190 * 1024 || num_tokens <= 0 ||
        num_tokens > num_max_tokens_per_rank || num_max_tokens_per_rank > 256)
        return false;
    switch (num_ranks) {
        case 8: return num_tokens <= 64;
        case 16:
        case 32:
        case 64:
        case 128: return num_tokens <= 32;
        default: return false;
    }
}

// Diagnostic escape hatch for comparing the same public API against its original
// transport. No public constructor or dispatch/combine option is required.
inline bool low_latency_transport_enabled() {
    const auto disabled = std::getenv("EP_DISABLE_LOW_LATENCY_TRANSPORT");
    return disabled == nullptr || std::strcmp(disabled, "1") != 0;
}

}  // namespace deep_ep
