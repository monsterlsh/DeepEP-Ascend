#pragma once

#include <climits>
#include <format>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

// All tensors are contiguous and on the current NPU stream:
//   ll_x: [capacity_rows, hidden_bytes] raw unweighted LL payload;
//   native_source_ids: int32[capacity_rows], src_rank * M + src_token;
//   expert_recv_counts: int64[num_local_experts], valid LL rows by expert;
//   origin_to_row: int32[P * M] scratch, initialized by this launch;
//   cached_recv_src_metadata: native int32[recv_capacity, K + 2], or nullptr.
// Run after LL dispatch and native layout-only dispatch have both completed on
// this stream. Native metadata notification must have completed on every rank
// (launch_dispatch(..., do_barrier=true, layout_only=true)). Retain all inputs
// and scratch until the ordinary dispatch epilogue / completion event retires.
static void launch_dispatch_ll_bridge(
        void* ll_x, void* native_source_ids, void* expert_recv_counts,
        void* origin_to_row, void* psum_num_recv_tokens_per_rank,
        void* cached_recv_src_metadata, void* buffer,
        const int& num_ranks, const int& num_max_tokens_per_rank,
        const int& num_local_experts, const int& num_topk,
        const int& num_hidden_bytes, const int& num_sf_packs,
        const int& num_vec_cores, const int& num_ub_bytes,
        const int& capacity_rows, const bool& cached_mode) {
    EP_HOST_ASSERT(ll_x != nullptr and native_source_ids != nullptr and expert_recv_counts != nullptr);
    EP_HOST_ASSERT(origin_to_row != nullptr and psum_num_recv_tokens_per_rank != nullptr and buffer != nullptr);
    EP_HOST_ASSERT(not cached_mode or cached_recv_src_metadata != nullptr);
    EP_HOST_ASSERT(num_ranks > 0 and num_max_tokens_per_rank > 0 and num_local_experts > 0);
    EP_HOST_ASSERT(static_cast<int64_t>(num_ranks) * num_max_tokens_per_rank <= INT_MAX);
    EP_HOST_ASSERT(num_topk > 0 and num_sf_packs >= 0 and num_vec_cores > 0 and capacity_rows > 0);
    EP_HOST_ASSERT(num_hidden_bytes > 0 and num_hidden_bytes % kNumUbAlignmentBytes == 0);
    EP_HOST_ASSERT(num_hidden_bytes <= num_ub_bytes);

    const auto map_kernel = jit->compile("dispatch_ll_origin_map", std::format(R"(
#include <deep_ep/impls/ep/dispatch_ll_bridge.hpp>
using namespace deep_ep;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_ll_origin_map_impl<{}, {}, {}>);
}}
)", num_ranks, num_max_tokens_per_rank, num_local_experts));
    const auto copy_kernel = jit->compile("dispatch_ll_copy_payload", std::format(R"(
#include <deep_ep/impls/ep/dispatch_ll_bridge.hpp>
using namespace deep_ep;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_ll_copy_payload_impl<{}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", num_ranks, num_max_tokens_per_rank, num_topk, num_hidden_bytes,
        num_sf_packs, num_vec_cores, num_ub_bytes, cached_mode));

    jit->launch(map_kernel, {.num_blocks = 1, .num_ubuf_bytes = num_ub_bytes},
                native_source_ids, expert_recv_counts, origin_to_row, capacity_rows);
    jit->launch(copy_kernel, {.num_blocks = num_vec_cores, .num_ubuf_bytes = num_ub_bytes},
                ll_x, origin_to_row, psum_num_recv_tokens_per_rank,
                cached_recv_src_metadata, buffer, capacity_rows);
}

}  // namespace deep_ep
