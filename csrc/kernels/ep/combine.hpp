#pragma once

#include <cstdint>
#include <format>

#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>

#include "../../runtime/jit.hpp"
#include "kernel_mode.hpp"

namespace deep_ep {

static layout::TokenLayout get_combine_token_layout(const int& num_hidden_bytes, const int& num_topk) {
    return layout::TokenLayout(num_hidden_bytes, 0, num_topk, false);
}

static void* launch_ep_combine(EPKernelMode mode, auto&& before_launch,
                               void* x,
                               void* topk_weights, bool has_topk_weights,
                               void* src_metadata,
                               void* psum_num_recv_tokens_per_rank,
                               void* buffer,
                               void* send_buffer,
                               void* workspace,
                               void* jetty_ptrs,
                               const int& rank_idx, const int& num_ranks,
                               const int& num_max_tokens_per_rank,
                               const int& num_topk,
                               const int& hidden,
                               const int& num_ub_bytes,
                               const int& num_vec_cores,
                               const int64_t& num_timeout_cycles,
                               const bool& do_barrier = false, bool use_fp16 = false) {
    EP_HOST_ASSERT(num_vec_cores > 0);
    EP_HOST_ASSERT(num_ub_bytes > 0);
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0);

    const bool native = mode == EPKernelMode::Native;
    const bool mte = mode == EPKernelMode::DecodeMTE;
    const auto decode_args = native ? std::string{} : std::format(", {}, {}", mte, use_fp16 ? "half" : "bfloat16_t");
    const auto kernel = jit->compile(mte ? "combine_mte" : "combine", std::format(R"(
#include <deep_ep/impls/ep/{}combine.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}{}>);
}}
)", native ? "" : "decode/", native ? "" : "deep_ep::decode::",
        num_ranks, num_topk, hidden, num_max_tokens_per_rank, num_vec_cores,
        num_timeout_cycles, num_ub_bytes, do_barrier, native ? topk_weights != nullptr : has_topk_weights, decode_args));

    before_launch();
    jit->launch(
        kernel, {.num_blocks = num_vec_cores, .num_ubuf_bytes = num_ub_bytes},
        x,
        topk_weights,
        src_metadata,
        psum_num_recv_tokens_per_rank,
        buffer,
        send_buffer,
        workspace,
        jetty_ptrs,
        rank_idx);
    return buffer;
}

static void* launch_combine(void* x,
                            void* topk_weights,
                            void* src_metadata,
                            void* psum_num_recv_tokens_per_rank,
                            void* buffer,
                            void* send_buffer,
                            void* workspace,
                            void* jetty_ptrs,
                            const int& rank_idx, const int& num_ranks,
                            const int& num_max_tokens_per_rank,
                            const int& num_topk,
                            const int& hidden,
                            const int& num_ub_bytes,
                            const int& num_vec_cores,
                            const int64_t& num_timeout_cycles,
                            const bool& do_barrier = false) {
    return launch_ep_combine(EPKernelMode::Native, [] {}, x, topk_weights, topk_weights != nullptr,
                             src_metadata, psum_num_recv_tokens_per_rank, buffer, send_buffer,
                             workspace, jetty_ptrs, rank_idx, num_ranks, num_max_tokens_per_rank,
                             num_topk, hidden, num_ub_bytes, num_vec_cores, num_timeout_cycles, do_barrier);
}

static void launch_ep_combine_reduce_epilogue(EPKernelMode mode, void* combined_x,
                                              void* combined_topk_weights,
                                              void* combined_topk_idx,
                                              void* reduce_buffer,
                                              void* bias_0,
                                              void* bias_1,
                                              void* workspace,
                                              void* jetty_ptrs,
                                              const int& num_ranks,
                                              const int& num_combined_tokens,
                                              const int& num_max_tokens_per_rank,
                                              const int& num_experts, const int& num_topk,
                                              const int& hidden,
                                              const int& num_ub_bytes,
                                              const int& num_vec_cores,
                                              const int64_t& num_timeout_cycles,
                                              const bool& do_barrier = true, bool use_fp16 = false) {
    EP_HOST_ASSERT(num_vec_cores > 0);
    EP_HOST_ASSERT(num_ub_bytes > 0);
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
    EP_HOST_ASSERT(num_topk > 0 and num_topk <= 32);
    EP_HOST_ASSERT(hidden % (kNumVectorRegisterBytes / static_cast<int>(sizeof(uint16_t))) == 0);

    const bool native = mode == EPKernelMode::Native;
    const auto decode_args = native ? std::string{} : std::format(", {}, {}, {}",
        hidden * sizeof(uint16_t), use_fp16 ? "half" : "bfloat16_t", mode == EPKernelMode::DecodeMTE);
    const auto kernel = jit->compile("combine_reduce_epilogue", std::format(R"(
#include <deep_ep/impls/ep/{}combine_reduce_epilogue.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}{}>);
}}
)", native ? "" : "decode/", native ? "" : "deep_ep::decode::",
        num_ranks, num_experts, num_topk,
        hidden, num_max_tokens_per_rank, num_ub_bytes, num_vec_cores, num_timeout_cycles, do_barrier, decode_args));

    // Launch
    jit->launch(
        kernel, {.num_blocks = num_vec_cores, .num_ubuf_bytes = num_ub_bytes},
        combined_x,
        combined_topk_weights,
        combined_topk_idx,
        reduce_buffer,
        bias_0,
        bias_1,
        workspace,
        jetty_ptrs,
        num_combined_tokens);
}

static void launch_combine_reduce_epilogue(void* combined_x,
                                           void* combined_topk_weights,
                                           void* combined_topk_idx,
                                           void* reduce_buffer,
                                           void* bias_0,
                                           void* bias_1,
                                           void* workspace,
                                           void* jetty_ptrs,
                                           const int& num_ranks,
                                           const int& num_combined_tokens,
                                           const int& num_max_tokens_per_rank,
                                           const int& num_experts, const int& num_topk,
                                           const int& hidden,
                                           const int& num_ub_bytes,
                                           const int& num_vec_cores,
                                           const int64_t& num_timeout_cycles,
                                           const bool& do_barrier = true) {
    launch_ep_combine_reduce_epilogue(EPKernelMode::Native, combined_x, combined_topk_weights, combined_topk_idx,
        reduce_buffer, bias_0, bias_1, workspace, jetty_ptrs, num_ranks, num_combined_tokens,
        num_max_tokens_per_rank, num_experts, num_topk, hidden, num_ub_bytes, num_vec_cores,
        num_timeout_cycles, do_barrier);
}

}  // namespace deep_ep
