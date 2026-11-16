#pragma once

#include <charconv>
#include <cstdlib>
#include <cstring>

#include <deep_ep/impls/ep/combine_layout.hpp>
#include <deep_ep/impls/ep/dispatch_direct_layout.hpp>

namespace deep_ep {

constexpr int kMaxDecodeRanks = 128;
constexpr int kMaxDecodeCapacity = 256;

// M is the common per-rank slot capacity used to address each peer's communication buffer.
constexpr bool is_decode_capacity(int num_ranks, int capacity) {
    return num_ranks > 1 and num_ranks <= kMaxDecodeRanks and math::is_power_of_2(num_ranks) and
           capacity >= 1 and capacity <= kMaxDecodeCapacity;
}

constexpr bool is_decode_batch(int batch, int capacity) {
    return batch > 0 and batch <= capacity;
}

constexpr bool is_decode_geometry(int num_ranks, int capacity, int hidden, int num_topk, int num_experts) {
    return is_decode_capacity(num_ranks, capacity) and
           hidden > 0 and hidden % (kNumVectorRegisterBytes / 2) == 0 and
           num_experts > 0 and num_experts <= kDispatchWorkerThreads and num_experts % num_ranks == 0 and
           num_topk > 0 and num_topk <= 32 and num_topk <= num_experts;
}

constexpr bool is_decode_shape(int num_ranks, int capacity, int hidden, int num_topk, int num_experts,
                               int num_vec_cores, int num_ub_bytes, bool use_fp8) {
    if (not is_decode_geometry(num_ranks, capacity, hidden, num_topk, num_experts) or
        num_vec_cores != 64 or num_ub_bytes != 216 * 1024 or
        (use_fp8 and hidden % kNumVectorRegisterBytes != 0))
        return false;
    // BF16 and FP16 expert outputs use two bytes per element after any dispatch dtype.
    const int64_t hidden_bytes = int64_t{2} * hidden;
    const int issue_bytes = num_ub_bytes - get_combine_fixed_bytes(num_ranks, num_topk, true);
    const auto reduce_slots = get_combine_hidden_slots(num_ub_bytes, hidden_bytes + get_combine_weight_bytes(num_topk));
    return num_topk <= get_combine_hidden_slots(issue_bytes, hidden_bytes) and
           math::min(num_ranks, num_topk) + 2 <= reduce_slots;
}

// An explicit combine limit can narrow the decode capacity range.
inline int mte_bs_threshold_override() {
    const auto value = std::getenv("EP_MTE_BS_THRESHOLD");
    if (value == nullptr or *value == '\0')
        return -1;
    const auto end = value + std::strlen(value);
    int threshold = -1;
    const auto result = std::from_chars(value, end, threshold);
    if (result.ec != std::errc{} or result.ptr != end or threshold < -1)
        return -1;
    return threshold;
}

// Diagnostic escape hatch for comparing the same public API against its original
// transport. No public constructor or dispatch/combine option is required.
inline bool low_latency_transport_enabled() {
    const auto disabled = std::getenv("EP_DISABLE_LOW_LATENCY_TRANSPORT");
    return disabled == nullptr or std::strcmp(disabled, "1") != 0;
}

}  // namespace deep_ep
