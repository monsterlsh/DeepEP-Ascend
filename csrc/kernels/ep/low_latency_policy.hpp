#pragma once

#include <cstdlib>
#include <cstring>

namespace deep_ep {

// Retain the native small-batch MTE policy only for EP128 after transport
// comparisons. Other EP sizes use the original URMA transport. EP128 keeps
// its existing model, hardware and capacity guards and large-batch fallback.
// The wire layout, allocation, routing metadata and collective sequence remain
// those of EPBuffer. Peers may consequently use different transports when their
// actual token counts differ, without introducing another agreement collective.
constexpr bool prefer_low_latency_transport(int arch, int num_ranks, int num_tokens,
                                            int num_max_tokens_per_rank, int hidden,
                                            int num_topk, int num_experts, int num_vec_cores,
                                            bool use_fp8, int num_ub_bytes = 190 * 1024) {
    if (arch != 3510 or use_fp8 or hidden != 7168 or num_topk != 8 or num_experts != 128 or
        num_vec_cores != 64 or num_ub_bytes < 190 * 1024 or num_tokens <= 0 or
        num_tokens > num_max_tokens_per_rank or num_max_tokens_per_rank > 256)
        return false;
    return num_ranks == 128 and num_tokens <= 32;
}

// Diagnostic escape hatch for comparing the same public API against its original
// transport. No public constructor or dispatch/combine option is required.
inline bool low_latency_transport_enabled() {
    const auto disabled = std::getenv("EP_DISABLE_LOW_LATENCY_TRANSPORT");
    return disabled == nullptr or std::strcmp(disabled, "1") != 0;
}

// LL server-dedup gate (T08/T09). Off by default; see docs/design/ll-default-integration.md.
// Only enables the explicit LL server-dedup path when the caller opts in. This is
// independent of, and never overrides, low_latency_transport_enabled().
inline bool ll_server_dedup_enabled() {
    const auto enabled = std::getenv("EP_ENABLE_LL_SERVER_DEDUP");
    return enabled != nullptr and std::strcmp(enabled, "1") == 0;
}

// All-rank deterministic admission for the LL server-dedup dispatch path. The route
// legality (no masked/duplicate experts) is still checked separately on-device (D06);
// this function only covers the shape/feature preconditions that are identical across
// ranks. Server-dedup FIRST_HOP relay only pays off with at least two 8-rank servers.
constexpr bool prefer_ll_server_dedup(int num_ranks, int num_tokens, int num_max_tokens_per_rank,
                                      int num_vec_cores, bool cached, bool deferred,
                                      int expert_alignment, bool use_fp8, bool has_stats) {
    if (cached or deferred or use_fp8 or has_stats or expert_alignment != 1)
        return false;
    if (num_vec_cores != 64)
        return false;
    if (num_tokens <= 0 or num_tokens > num_max_tokens_per_rank or num_max_tokens_per_rank > 256)
        return false;
    return num_ranks >= 16;
}

}  // namespace deep_ep
