#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
#include <deep_ep/common/reduction.hpp>

#include "base.hpp"
#include "low_latency_state.hpp"
#include "../comm/api.hpp"
#include "../utils/tensor.hpp"
#include "../kernels/ep/api.hpp"
#include "../kernels/ep/low_latency_policy.hpp"
#include "../utils/event.hpp"
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

namespace deep_ep {

class EPBuffer: public BufferBase {
    int64_t num_buffer_bytes;
    void* host_workspace = nullptr;
    void* mapped_host_workspace = nullptr;
    std::shared_ptr<EPLowLatencyState> low_latency_state;
    using Epilogue = std::function<pybind11::object()>;
    mutable std::shared_ptr<Epilogue> deferred_epilogue;

    static int get_num_used_vec_cores(int num_ai_cores) {
        const auto cores = static_cast<int64_t>(num_ai_cores) * runtime->get_num_vec_cores_per_ai_core();
        EP_HOST_ASSERT(num_ai_cores > 0 and cores > 0 and cores <= std::numeric_limits<int>::max());
        return static_cast<int>(cores);
    }

    static void record_use(const torch::Tensor& tensor, const c10_npu::NPUStream& stream) {
        if (tensor.numel() != 0)
            c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }

    pybind11::tuple defer(Epilogue epilogue) const {
        // A direct runtime caller may release the buffer before invoking the callback.
        auto owner = pybind11::cast(this, pybind11::return_value_policy::reference);
        deferred_epilogue = std::make_shared<Epilogue>(std::move(epilogue));
        Epilogue hook = [this, owner = std::move(owner), pending = std::weak_ptr(deferred_epilogue)] {
            EP_HOST_ASSERT(not destroyed and "EP epilogue cancelled by buffer destruction");
            low_latency_state->require_healthy();
            EP_HOST_ASSERT(deferred_epilogue and pending.lock() == deferred_epilogue and "EP epilogue already consumed");
            auto once = std::move(deferred_epilogue);
            return (*once)();
        };
        return pybind11::make_tuple(pybind11::none(), EventHandle(), std::move(hook));
    }

public:
    std::shared_ptr<comm::Context> context;
    torch::Tensor lb_storage;

    EPBuffer(const int& rank_idx, const int& num_ranks, const std::string& hccl_group_name,
             const int64_t& num_buffer_bytes, const int64_t& num_lb_buffer_bytes,
             const bool& use_huge1g_page,
             const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
             const bool& explicitly_destroy):
        BufferBase(explicitly_destroy), num_buffer_bytes(num_buffer_bytes),
        low_latency_state(std::make_shared<EPLowLatencyState>()) {
        EP_HOST_ASSERT(num_buffer_bytes >= 0 and num_buffer_bytes % kNumAllocationAlignmentBytes == 0);
        EP_HOST_ASSERT(num_lb_buffer_bytes >= 0 and num_lb_buffer_bytes % kNumAllocationAlignmentBytes == 0);
        EP_HOST_ASSERT(num_buffer_bytes <= std::numeric_limits<int64_t>::max() - num_lb_buffer_bytes);
        const auto arena_bytes = num_buffer_bytes + num_lb_buffer_bytes;
        EP_HOST_ASSERT(num_cpu_timeout_secs > 0 and num_gpu_timeout_secs > 0);
        EP_HOST_ASSERT(arena_bytes <= std::numeric_limits<int64_t>::max() -
                       static_cast<int64_t>(sizeof(layout::EPSignals)));
        auto topology = EPLowLatencyState::local_topology(num_ranks);
        context = std::make_shared<comm::Context>(
            hccl_group_name, rank_idx, num_ranks, sizeof(layout::EPSignals), arena_bytes, 0,
            use_huge1g_page, num_cpu_timeout_secs, num_gpu_timeout_secs, nullptr, &topology);
        main_context = context;
        low_latency_state->attach_context(context, topology);
        auto& workspace = *static_cast<layout::EPSignals*>(context->workspace);
        context->set_common_signals(&workspace.common_signals);
        lb_storage = torch::from_blob(
            context->buffer, {arena_bytes}, [context = context](void*) {},
            torch::TensorOptions().dtype(torch::kByte).device(TORCH_DEVICE_NPU)
        ).narrow(0, num_buffer_bytes, num_lb_buffer_bytes);
        ACL_CHECK(aclrtMallocHost(&host_workspace, sizeof(layout::EPSignals)));
        ACL_CHECK(aclrtHostRegister(host_workspace, sizeof(layout::EPSignals), ACL_HOST_REGISTER_MAPPED, &mapped_host_workspace));
        EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(host_workspace) % alignof(layout::EPSignals) == 0 and
                       reinterpret_cast<uintptr_t>(mapped_host_workspace) % alignof(layout::EPSignals) == 0);
        std::memset(host_workspace, 0, sizeof(layout::EPSignals));
    }

    ~EPBuffer() noexcept override {
        try {
            destroy_on_destruction("EP");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "DeepEP: EP buffer cleanup failed: %s\n", e.what());
        } catch (...) {
            std::fputs("DeepEP: EP buffer cleanup failed.\n", stderr);
        }
    }

    void destroy() override {
        EP_HOST_ASSERT(not destroyed);
        low_latency_state->require_healthy();
        try {
            comm::barrier(*context, true);
            // All ranks have drained the issuing kernels and their remote writes.
            // Release even an abandoned epilogue's tensor captures only after this point.
            deferred_epilogue.reset();
            low_latency_state->record_completion(false);
            ACL_CHECK(aclrtHostUnregister(host_workspace));
            ACL_CHECK(aclrtFreeHost(host_workspace));
            context->finalize_ep();
            lb_storage = torch::Tensor();
            destroyed = true;
        } catch (...) {
            low_latency_state->fail();
            throw;
        }
    }

    EventHandle lb_prefetch_weights(const std::vector<torch::Tensor>& redundant_expert_weights,
                                    const std::vector<torch::Tensor>& expert_weights,
                                    const torch::Tensor& redundancy_mapping, const int& num_sms,
                                    const std::optional<EventHandle>& previous_event) const {
        EP_HOST_UNREACHABLE("EPBuffer::lb_prefetch_weights is not implemented on Ascend");
    }

    EventHandle lb_reduce_grads(const torch::Tensor& redundant_expert_grads,
                                const torch::Tensor& expert_grads, const torch::Tensor& redundancy_mapping,
                                const int& num_sms, const std::optional<EventHandle>& previous_event) const {
        EP_HOST_UNREACHABLE("EPBuffer::lb_reduce_grads is not implemented on Ascend");
    }

    pybind11::tuple
    dispatch(const torch::Tensor& x,
             const std::optional<torch::Tensor>& sf,
             const torch::Tensor& topk_idx,
             const std::optional<torch::Tensor>& topk_weights,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const std::optional<int>& cached_num_recv_tokens,
             const std::optional<int>& cached_num_expanded_tokens,
             const std::optional<std::vector<int>>& cached_num_recv_tokens_per_expert_list,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_rank,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_num_unaligned_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_dst_buffer_slot_idx,
             const std::optional<torch::Tensor>& cached_dst_gsge_idx,
             const std::optional<torch::Tensor>& cached_recv_src_metadata,
             const int& num_max_tokens_per_rank,
             const int& num_experts, const int& expert_alignment,
             const int& num_ai_cores,
             const std::optional<EventHandle>& previous_event,
             const bool& async_with_compute_stream,
             const bool& allocate_on_comm_stream,
             const bool& do_handle_copy, const bool& do_cpu_sync,
             const bool& do_expand, const bool& do_zero_padding,
             const bool& use_cube_aligned_col_major_sf,
             const bool& defer_epilogue) const {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(not defer_epilogue or async_with_compute_stream);
        const bool decode = use_decode_implementation(num_max_tokens_per_rank);

        // Ascend EP has some limitations over NVIDIA version
        EP_HOST_ASSERT(do_expand);
        EP_HOST_ASSERT(not do_handle_copy);

        // Cached dispatch is only supported for the expand-only, non-hybrid Ascend path
        const bool cached_mode = cached_num_recv_tokens.has_value();
        if (cached_mode) {
            EP_HOST_ASSERT(not do_cpu_sync and "Cannot do CPU sync with cached mode");
            EP_HOST_ASSERT(not cumulative_local_expert_recv_stats.has_value());
            EP_HOST_ASSERT(cached_num_expanded_tokens.has_value());
            EP_HOST_ASSERT(cached_num_recv_tokens_per_expert_list.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_rank.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_expert.has_value());
            EP_HOST_ASSERT(cached_num_unaligned_recv_tokens_per_expert.has_value());
            EP_HOST_ASSERT(cached_dst_buffer_slot_idx.has_value());
            EP_HOST_ASSERT(cached_dst_gsge_idx.has_value());
            EP_HOST_ASSERT(cached_recv_src_metadata.has_value());
        }

        // Check data tensor
        EP_HOST_ASSERT(x.dim() == 2 and x.device() == lb_storage.device() and x.is_contiguous());
        EP_HOST_ASSERT(num_experts > 0 and num_experts % context->num_ranks == 0);
        EP_HOST_ASSERT(x.size(0) <= num_max_tokens_per_rank and x.size(1) > 0 and
                       x.size(1) <= std::numeric_limits<int>::max() / x.element_size());
        const auto num_tokens = static_cast<int>(x.size(0));
        const auto hidden = static_cast<int>(x.size(1));
        const auto num_hidden_bytes = hidden * static_cast<int>(x.element_size());
        const auto num_experts_per_rank = num_experts / context->num_ranks;
        EP_HOST_ASSERT((not do_zero_padding or num_hidden_bytes % kNumVectorRegisterBytes == 0) and
                       "Zero padding requires a vector-aligned hidden row");

        int num_sf_packs = 0;
        int sf_token_stride = 0;
        int sf_hidden_stride = 0;
        bool pack_scales = false;
        if (sf.has_value()) {
            EP_HOST_ASSERT(x.scalar_type() == torch::kFloat8_e4m3fn);
            EP_HOST_ASSERT(sf->dim() == 2 and sf->size(0) == num_tokens);
            EP_HOST_ASSERT(sf->device() == x.device());
            EP_HOST_ASSERT(sf->stride(0) == 1 or sf->stride(1) == 1);
            const bool wide = decode and (sf->scalar_type() == torch::kFloat32 or sf->scalar_type() == torch::kInt32);
            EP_HOST_ASSERT(wide or sf->element_size() == sizeof(sf_pack_t));
            // FP32 per-128 scales and packed E8M0 INT32 use two consecutive
            // 16-bit wire words per entry. Preserve their bits and public dtype.
            if (wide) {
                EP_HOST_ASSERT(num_hidden_bytes % kNumUbAlignmentBytes == 0 and
                               "32-bit FP8 scales require a 32-byte aligned hidden row");
                EP_HOST_ASSERT(sf->size(1) == math::ceil_div<int64_t>(hidden, 128));
            }
            pack_scales = wide or (decode and (sf->stride(0) == 0 or sf->stride(1) == 0));
            const int words_per_scale = wide ? 2 : 1;
            const auto columns = sf->size(1);
            // Singleton/empty dimensions can carry arbitrary PyTorch strides.
            const auto row_stride = (pack_scales or (decode and (num_tokens <= 1 or columns == 0))) ? columns : sf->stride(0);
            const auto column_stride = (pack_scales or (decode and (num_tokens == 0 or columns <= 1))) ? 1 : sf->stride(1);
            EP_HOST_ASSERT(columns <= std::numeric_limits<int>::max() / (words_per_scale * static_cast<int>(sizeof(sf_pack_t))) and
                           row_stride <= std::numeric_limits<int>::max() / words_per_scale and
                           column_stride <= std::numeric_limits<int>::max());
            num_sf_packs = static_cast<int>(columns) * words_per_scale;
            sf_token_stride = static_cast<int>(row_stride) * words_per_scale;
            sf_hidden_stride = static_cast<int>(column_stride);
        } else {
            EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16 or (decode and x.scalar_type() == torch::kFloat16));
        }

        // Check top-k stuffs
        EP_HOST_ASSERT(1 <= topk_idx.size(1) and topk_idx.size(1) <= 32);
        const auto num_topk = static_cast<int>(topk_idx.size(1));
        EP_HOST_ASSERT(topk_idx.size(0) == num_tokens);
        EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(num_tokens == 0 or topk_idx.device() == x.device());
        EP_HOST_ASSERT(topk_idx.is_contiguous());
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and math::is_power_of_2(expert_alignment));
        const auto max_expert_rows = get_dispatch_max_expert_rows(context->num_ranks, num_max_tokens_per_rank);
        EP_HOST_ASSERT(max_expert_rows <= reduction::kReductionMaxLowCount and
                       "Per-expert route count exceeds the 24-bit dispatch reduction");
        EP_HOST_ASSERT(not do_cpu_sync or math::align<int64_t>(max_expert_rows, expert_alignment) <= reduction::kReductionMaxLowCount);
        const auto max_expanded_rows = get_dispatch_max_expanded_rows(
            max_expert_rows, num_topk, num_experts_per_rank, expert_alignment);
        EP_HOST_ASSERT(max_expanded_rows <= std::numeric_limits<int>::max() and
                       "Expanded row capacity exceeds the 32-bit dispatch slot index");

        // Weights are optional for training backward
        float* topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            EP_HOST_ASSERT(topk_weights->size(0) == num_tokens and topk_weights->size(1) == num_topk);
            EP_HOST_ASSERT(num_tokens == 0 or topk_weights->device() == x.device());
            EP_HOST_ASSERT(topk_weights->is_contiguous());
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        // Expert receiving counter
        int* cumulative_local_expert_recv_stats_ptr = nullptr;
        if (cumulative_local_expert_recv_stats.has_value()) {
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and
                           cumulative_local_expert_recv_stats->size(0) == num_experts_per_rank);
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->device() == x.device());
            cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
        }

        // Resource configs
        const auto num_used_vec_cores = get_num_used_vec_cores(num_ai_cores);
        const auto num_ub_bytes = static_cast<int>(runtime->get_num_ubuf_bytes_per_vec_core());
        EP_HOST_ASSERT(num_ub_bytes > 0);

        // Check buffer size
        EP_HOST_ASSERT(get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden, num_sf_packs, num_topk,
            static_cast<int>(x.element_size()), context->num_ranks) <= num_buffer_bytes);

        // Handle tensors written by the dispatch notify path
        const auto int_options = at::TensorOptions(x.device()).dtype(torch::kInt);
        const auto psum_num_recv_tokens_per_rank = cached_mode ?
            cached_psum_num_recv_tokens_per_rank.value() :
            torch::empty({context->num_ranks}, int_options);
        const auto psum_num_recv_tokens_per_expert = cached_mode ?
            cached_psum_num_recv_tokens_per_expert.value() :
            torch::empty({num_experts_per_rank}, int_options);
        const auto num_unaligned_recv_tokens_per_expert = cached_mode ?
            cached_num_unaligned_recv_tokens_per_expert.value() :
            torch::empty({num_experts_per_rank}, int_options);
        const auto dst_buffer_slot_idx = cached_mode ?
            cached_dst_buffer_slot_idx.value() :
            torch::empty({num_tokens, num_topk}, int_options);
        const auto dst_gsge_idx = cached_mode ?
            cached_dst_gsge_idx.value() :
            torch::empty({num_tokens, num_topk}, int_options);
        if (cached_mode) {
            const auto& metadata = *cached_recv_src_metadata;
            EP_HOST_ASSERT(*cached_num_recv_tokens >= 0 and *cached_num_expanded_tokens >= 0);
            EP_HOST_ASSERT(metadata.dim() == 2 and metadata.size(0) == *cached_num_recv_tokens and
                           metadata.size(1) == num_topk + 2 and metadata.is_contiguous());
            EP_HOST_ASSERT(metadata.scalar_type() == torch::kInt and metadata.device() == x.device());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.dim() == 1 and
                           psum_num_recv_tokens_per_rank.size(0) == context->num_ranks);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.dim() == 1 and
                           psum_num_recv_tokens_per_expert.size(0) == num_experts_per_rank);
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.dim() == 1 and
                           num_unaligned_recv_tokens_per_expert.size(0) == num_experts_per_rank);
            EP_HOST_ASSERT(dst_buffer_slot_idx.dim() == 2 and
                           dst_buffer_slot_idx.size(0) == num_tokens and
                           dst_buffer_slot_idx.size(1) == num_topk);
            EP_HOST_ASSERT(dst_gsge_idx.sizes() == dst_buffer_slot_idx.sizes());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(dst_buffer_slot_idx.is_contiguous());
            EP_HOST_ASSERT(dst_gsge_idx.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.scalar_type() == torch::kInt);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.scalar_type() == torch::kInt);
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.scalar_type() == torch::kInt);
            EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
            EP_HOST_ASSERT(dst_gsge_idx.scalar_type() == torch::kInt);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.device() == x.device());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.device() == x.device());
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.device() == x.device());
            EP_HOST_ASSERT(dst_buffer_slot_idx.device() == x.device());
            EP_HOST_ASSERT(dst_gsge_idx.device() == x.device());
        }

        const auto barrier_in_prologue = runtime->get_barrier_in_prologue();
        low_latency_state->prepare();
        const bool low_latency_transport = decode and low_latency_state->prepare_dispatch(
            context->num_ranks, num_max_tokens_per_rank, hidden, num_topk, num_experts,
            num_used_vec_cores, num_ub_bytes, num_sf_packs, sf.has_value());
        torch::Tensor metadata_send_buffer;
        const auto tokens_per_core = math::ceil_div<int64_t>(num_max_tokens_per_rank, num_used_vec_cores);
        if (not low_latency_transport or not get_dispatch_mte_stages(
                context->num_ranks, num_experts, tokens_per_core, num_hidden_bytes, num_sf_packs * sizeof(sf_pack_t),
                num_topk, num_ub_bytes, cached_mode).keeps_metadata_in_ub(num_sf_packs, tokens_per_core)) {
            metadata_send_buffer = torch::empty(
                {num_tokens, get_dispatch_token_layout(num_hidden_bytes, num_sf_packs, num_topk).get_num_bytes(true)},
                x.options().dtype(torch::kByte));
        }

        // Clean host workspace
        const auto host_workspace_layout = layout::EPWorkspaceLayout(host_workspace);
        if (not cached_mode) {
            std::fill_n(host_workspace_layout.get_host_rank_count_ptr(), context->num_ranks, 0);
            std::fill_n(host_workspace_layout.get_host_expert_count_ptr(), num_experts_per_rank, 0);
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }

        // Direct launches may read inputs allocated on another NPU stream.
        const auto issue_stream = (decode or defer_epilogue) ? std::optional{c10_npu::getCurrentNPUStream()} : std::nullopt;
        if (decode) {
            record_use(x, *issue_stream);
            record_use(topk_idx, *issue_stream);
            if (sf.has_value())
                record_use(*sf, *issue_stream);
            if (topk_weights.has_value())
                record_use(*topk_weights, *issue_stream);
            if (cumulative_local_expert_recv_stats.has_value())
                record_use(*cumulative_local_expert_recv_stats, *issue_stream);
            if (cached_mode) {
                record_use(dst_buffer_slot_idx, *issue_stream);
                record_use(dst_gsge_idx, *issue_stream);
                record_use(psum_num_recv_tokens_per_rank, *issue_stream);
            }
        }

        // Launch dispatch
        // A previous dispatch on this buffer may have produced the scales on another stream.
        const auto wire_scales = pack_scales ? std::optional{sf->contiguous()} : sf;
        const bool relay = low_latency_transport and low_latency_state->server_relay_enabled();
        auto& workspace = *static_cast<layout::EPSignals*>(context->workspace);
        auto submission = low_latency_state->submission_guard(x, metadata_send_buffer);
        launch_ep_dispatch(
            get_ep_kernel_mode(decode, low_latency_transport), [&] { submission.begin_submission(); },
            x.data_ptr(), wire_scales.has_value() ? wire_scales->data_ptr() : nullptr,
            topk_idx.data_ptr(), topk_weights_ptr,
            metadata_send_buffer.defined() ? metadata_send_buffer.data_ptr() : nullptr,
            cumulative_local_expert_recv_stats_ptr,
            psum_num_recv_tokens_per_rank.data_ptr<int>(),
            psum_num_recv_tokens_per_expert.data_ptr<int>(),
            num_unaligned_recv_tokens_per_expert.data_ptr<int>(),
            dst_buffer_slot_idx.data_ptr<int>(),
            dst_gsge_idx.data_ptr<int>(),
            context->buffer, context->workspace, mapped_host_workspace,
            context->hccl_context->jetty_device_ptrs,
            context->rank_idx, context->num_ranks,
            num_tokens, num_max_tokens_per_rank,
            num_experts, num_topk, expert_alignment,
            cached_mode, do_cpu_sync,
            num_hidden_bytes, num_sf_packs,
            sf_token_stride, sf_hidden_stride,
            num_ub_bytes, num_used_vec_cores,
            context->num_gpu_timeout_cycles, barrier_in_prologue,
            relay ? workspace.decode_server_ids : nullptr, relay ? workspace.decode_relay_slots : nullptr);

        // URMA still reads x and metadata_send_buffer after the issue kernel exits.
        // No NPU stream tracks that traffic; capture its sources until the epilogue queues the drain.
        auto epilogue = [=, x = x, metadata_send_buffer = metadata_send_buffer, context = context,
                         low_latency_state = low_latency_state]() {
            const auto ll_failure_guard = low_latency_state->failure_guard();
            if (issue_stream.has_value()) {
                const auto copy_stream = c10_npu::getCurrentNPUStream();
                const bool cross_stream = copy_stream != *issue_stream;
                if ((decode and cached_mode) or cross_stream) {
                    record_use(psum_num_recv_tokens_per_rank, copy_stream);
                    record_use(psum_num_recv_tokens_per_expert, copy_stream);
                    if (cached_mode)
                        record_use(*cached_recv_src_metadata, copy_stream);
                }
                if (not low_latency_transport and cross_stream) {
                    record_use(x, copy_stream);
                    record_use(metadata_send_buffer, copy_stream);
                }
            }

            // Wait GPU notify or reuse cached allocation sizes
            int num_recv_tokens = context->num_ranks * num_max_tokens_per_rank;
            int num_expanded_tokens = 0;
            std::vector<int> num_recv_tokens_per_expert_list;
            if (cached_mode) {
                num_recv_tokens = cached_num_recv_tokens.value();
                num_expanded_tokens = cached_num_expanded_tokens.value();
                num_recv_tokens_per_expert_list = cached_num_recv_tokens_per_expert_list.value();
            } else if (do_cpu_sync) {
                const auto cpu_sync_deadline =
                    std::chrono::steady_clock::now() + std::chrono::seconds(context->num_cpu_timeout_secs);

                // Wait rank status arrival
                num_recv_tokens = 0;
                for (int i = 0; i < context->num_ranks; ++ i) {
                    const auto status = reduction::host::wait_ready(
                        host_workspace_layout.get_host_rank_count_ptr(i), context->num_ranks, cpu_sync_deadline);
                    num_recv_tokens += reduction::get_reduction_value(status);
                }

                // Wait expert status arrival
                num_recv_tokens_per_expert_list.reserve(num_experts_per_rank);
                for (int i = 0; i < num_experts_per_rank; ++ i) {
                    const auto status = reduction::host::wait_ready(
                        host_workspace_layout.get_host_expert_count_ptr(i), context->num_ranks, cpu_sync_deadline);
                    const auto count = reduction::get_reduction_value(status);
                    num_expanded_tokens += count;
                    num_recv_tokens_per_expert_list.push_back(count);
                }
            } else {
                num_expanded_tokens = static_cast<int>(max_expanded_rows);
            }

            // Allocate output tensors
            auto recv_x = torch::empty({num_expanded_tokens, hidden}, x.options());
            auto recv_sf = std::optional<torch::Tensor>();
            int recv_sf_token_stride = 0;
            int recv_sf_hidden_stride = 0;
            void* recv_sf_ptr = nullptr;
            if (sf.has_value()) {
                recv_sf_token_stride = num_sf_packs;
                recv_sf_hidden_stride = 1;
                const auto columns = sf->size(1);
                recv_sf = torch::empty_strided({num_expanded_tokens, columns}, {columns, 1}, sf->options());
                recv_sf_ptr = recv_sf->data_ptr();
            }
            auto recv_topk_weights = std::optional<torch::Tensor>();
            if (topk_weights.has_value()) {
                // An empty sender can receive weights from peers on the EP device.
                const auto device = num_expanded_tokens > 0 ? x.device() : topk_weights->device();
                recv_topk_weights = torch::empty({num_expanded_tokens}, topk_weights->options().device(device));
            }
            auto recv_src_metadata = cached_mode ?
                cached_recv_src_metadata.value() :
                torch::empty({num_recv_tokens, num_topk + 2}, int_options);

            launch_dispatch_copy_epilogue(
                context->buffer,
                context->workspace,
                low_latency_transport ? nullptr : context->hccl_context->jetty_device_ptrs,
                psum_num_recv_tokens_per_rank.data_ptr<int>(),
                psum_num_recv_tokens_per_expert.data_ptr<int>(),
                recv_x.data_ptr(),
                recv_sf_ptr,
                recv_topk_weights.has_value() ? recv_topk_weights->data_ptr() : nullptr,
                recv_src_metadata.data_ptr<int>(),
                cached_mode,
                do_zero_padding,
                context->rank_idx, context->num_ranks,
                num_max_tokens_per_rank,
                num_experts, num_topk,
                expert_alignment,
                num_hidden_bytes,
                num_sf_packs,
                recv_sf_token_stride,
                recv_sf_hidden_stride,
                num_ub_bytes,
                num_used_vec_cores,
                context->num_gpu_timeout_cycles,
                not barrier_in_prologue
            );

            // Convert row-major epilogue output to column-major
            if (recv_sf.has_value() and use_cube_aligned_col_major_sf)
                recv_sf = recv_sf->transpose(0, 1).contiguous().transpose(0, 1);

            low_latency_state->record_completion(decode or async_with_compute_stream);

            return pybind11::make_tuple(
                recv_x, recv_sf, recv_topk_weights,
                num_recv_tokens,
                num_expanded_tokens,
                num_recv_tokens_per_expert_list,
                psum_num_recv_tokens_per_rank,
                psum_num_recv_tokens_per_expert,
                num_unaligned_recv_tokens_per_expert,
                recv_src_metadata,
                dst_buffer_slot_idx,
                dst_gsge_idx);
        };

        // Defer epilogue and host sync
        if (defer_epilogue)
            return defer(std::move(epilogue));

        // Do epilogue immediately
        const auto result = epilogue();
        const auto event = async_with_compute_stream ?
            std::optional<EventHandle>(low_latency_state->completion_event()) : std::nullopt;
        return pybind11::make_tuple(result, event, pybind11::none());
    }

    pybind11::tuple
    combine(const torch::Tensor& x,
            const std::optional<torch::Tensor>& topk_weights,
            const std::optional<torch::Tensor>& bias_0,
            const std::optional<torch::Tensor>& bias_1,
            const torch::Tensor& src_metadata,
            const torch::Tensor& combined_topk_idx,
            const torch::Tensor& psum_num_recv_tokens_per_rank,
            const int& num_experts,
            const int& num_max_tokens_per_rank,
            const int& num_sms,
            const std::optional<EventHandle>& previous_event,
            const bool& async_with_compute_stream,
            const bool& allocate_on_comm_stream,
            const bool& defer_epilogue) const {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(not defer_epilogue or async_with_compute_stream);
        const bool decode = use_decode_implementation(num_max_tokens_per_rank);
        EP_HOST_ASSERT(num_experts > 0 and num_experts % context->num_ranks == 0);

        // Check data tensor
        EP_HOST_ASSERT(x.dim() == 2 and x.device() == lb_storage.device() and x.is_contiguous());
        EP_HOST_ASSERT(x.size(0) <= std::numeric_limits<int>::max() and x.size(1) > 0 and
                       x.size(1) <= std::numeric_limits<int>::max() / x.element_size());
        const auto num_tokens = static_cast<int>(x.size(0));
        const auto hidden = static_cast<int>(x.size(1));
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16 or (decode and x.scalar_type() == torch::kFloat16));
        // Check the existing reduce-epilogue requirement before issuing the collective.
        EP_HOST_ASSERT(hidden % (kNumVectorRegisterBytes / static_cast<int>(sizeof(uint16_t))) == 0);

        // Check metadata from dispatch
        EP_HOST_ASSERT(combined_topk_idx.dim() == 2 and combined_topk_idx.size(0) <= num_max_tokens_per_rank and
                       combined_topk_idx.size(1) > 0 and combined_topk_idx.size(1) <= 32);
        EP_HOST_ASSERT(src_metadata.dim() == 2 and src_metadata.size(0) <= std::numeric_limits<int>::max());
        EP_HOST_ASSERT(combined_topk_idx.numel() == 0 or combined_topk_idx.device() == x.device());
        EP_HOST_ASSERT(src_metadata.numel() == 0 or src_metadata.device() == x.device());
        EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.device() == x.device());
        const auto num_combined_tokens = static_cast<int>(combined_topk_idx.size(0));
        const auto num_topk = static_cast<int>(combined_topk_idx.size(1));
        const auto num_reduced_tokens = static_cast<int>(src_metadata.size(0));
        EP_HOST_ASSERT(combined_topk_idx.is_contiguous());
        EP_HOST_ASSERT(combined_topk_idx.scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(src_metadata.size(1) == num_topk + 2);
        EP_HOST_ASSERT(src_metadata.size(0) <= static_cast<int64_t>(context->num_ranks) * num_max_tokens_per_rank);
        EP_HOST_ASSERT(src_metadata.is_contiguous());
        EP_HOST_ASSERT(src_metadata.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.dim() == 1);
        EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.size(0) == context->num_ranks);
        EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.is_contiguous());
        EP_HOST_ASSERT(psum_num_recv_tokens_per_rank.scalar_type() == torch::kInt);

        // Check optional tensors
        float* topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            EP_HOST_ASSERT(topk_weights->dim() == 1 and topk_weights->size(0) == num_tokens);
            EP_HOST_ASSERT(topk_weights->is_contiguous());
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat);
            EP_HOST_ASSERT((num_tokens == 0 and num_combined_tokens == 0) or topk_weights->device() == x.device());
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        const std::optional<torch::Tensor> bias_opts[2] = {bias_0, bias_1};
        for (int i = 0; i < 2; ++ i) {
            if (bias_opts[i].has_value()) {
                const auto& bias = bias_opts[i].value();
                EP_HOST_ASSERT(bias.dim() == 2);
                EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
                EP_HOST_ASSERT(bias.is_contiguous());
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
                EP_HOST_ASSERT(bias.numel() == 0 or bias.device() == x.device());
            }
        }

        // Resource configs
        const auto num_used_vec_cores = get_num_used_vec_cores(num_sms);
        const auto num_ub_bytes = static_cast<int>(runtime->get_num_ubuf_bytes_per_vec_core());
        EP_HOST_ASSERT(num_ub_bytes > 0);

        // Check buffer size
        EP_HOST_ASSERT(get_combine_buffer_size(
            context->num_ranks, num_max_tokens_per_rank, hidden, num_topk) <= num_buffer_bytes);

        // Push data into remote reduce buffers
        const auto barrier_in_prologue = runtime->get_barrier_in_prologue();
        low_latency_state->prepare();
        const bool low_latency_transport = decode and low_latency_state->prepare_combine(
            context->num_ranks, num_max_tokens_per_rank, hidden, num_topk, num_experts, num_used_vec_cores, num_ub_bytes);
        torch::Tensor send_buffer;
        if (not low_latency_transport) {
            const auto token_layout = get_combine_token_layout(hidden * static_cast<int>(sizeof(uint16_t)), num_topk);
            send_buffer = torch::empty({num_reduced_tokens, token_layout.get_num_bytes()}, x.options().dtype(torch::kByte));
        }
        // Direct JIT launches bypass Torch's allocator stream tracking.
        const auto issue_stream = (decode or defer_epilogue) ? std::optional{c10_npu::getCurrentNPUStream()} : std::nullopt;
        if (decode) {
            record_use(x, *issue_stream);
            record_use(src_metadata, *issue_stream);
            record_use(psum_num_recv_tokens_per_rank, *issue_stream);
            if (topk_weights.has_value())
                record_use(*topk_weights, *issue_stream);
        }
        auto submission = low_latency_state->submission_guard(x, send_buffer);
        const auto mode = get_ep_kernel_mode(decode, low_latency_transport);
        const auto reduce_buffer = launch_ep_combine(
            mode, [&] { submission.begin_submission(); }, x.data_ptr(),
            topk_weights_ptr, topk_weights.has_value(),
            src_metadata.data_ptr<int>(),
            psum_num_recv_tokens_per_rank.data_ptr<int>(),
            context->buffer,
            send_buffer.defined() ? send_buffer.data_ptr() : nullptr,
            context->workspace,
            context->hccl_context->jetty_device_ptrs,
            context->rank_idx, context->num_ranks,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            num_ub_bytes,
            num_used_vec_cores,
            context->num_gpu_timeout_cycles,
            barrier_in_prologue, x.scalar_type() == torch::kFloat16
        );

        // URMA still reads x (direct sends) and send_buffer after the issue kernel exits.
        // No NPU stream tracks that traffic; capture its sources until the epilogue queues the drain.
        auto epilogue = [=, x = x, send_buffer = send_buffer, context = context,
                         low_latency_state = low_latency_state]() {
            const auto ll_failure_guard = low_latency_state->failure_guard();
            if (issue_stream.has_value()) {
                const auto reduce_stream = c10_npu::getCurrentNPUStream();
                const bool cross_stream = reduce_stream != *issue_stream;
                if (decode or cross_stream) {
                    record_use(combined_topk_idx, reduce_stream);
                    if (bias_0.has_value())
                        record_use(*bias_0, reduce_stream);
                    if (bias_1.has_value())
                        record_use(*bias_1, reduce_stream);
                }
                if (send_buffer.defined() and cross_stream) {
                    record_use(x, reduce_stream);
                    record_use(send_buffer, reduce_stream);
                }
            }
            // Allocate output tensors
            auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
            auto combined_topk_weights = std::optional<torch::Tensor>();
            float* combined_topk_weights_ptr = nullptr;
            if (topk_weights.has_value()) {
                combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
                combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
            }

            // Reduce pushed data
            launch_ep_combine_reduce_epilogue(
                mode, combined_x.data_ptr(),
                combined_topk_weights_ptr,
                combined_topk_idx.data_ptr<int64_t>(),
                reduce_buffer,
                bias_0.has_value() ? bias_0->data_ptr() : nullptr,
                bias_1.has_value() ? bias_1->data_ptr() : nullptr,
                context->workspace,
                context->hccl_context->jetty_device_ptrs,
                context->num_ranks,
                num_combined_tokens,
                num_max_tokens_per_rank,
                num_experts, num_topk,
                hidden,
                num_ub_bytes,
                num_used_vec_cores,
                context->num_gpu_timeout_cycles,
                not barrier_in_prologue, x.scalar_type() == torch::kFloat16
            );

            low_latency_state->record_completion(decode or async_with_compute_stream);
            return pybind11::make_tuple(combined_x, combined_topk_weights);
        };

        // Defer epilogue
        if (defer_epilogue)
            return defer(std::move(epilogue));

        // Do epilogue immediately
        const auto result = epilogue();
        const auto event = async_with_compute_stream ?
            std::optional<EventHandle>(low_latency_state->completion_event()) : std::nullopt;
        return pybind11::make_tuple(result, event, pybind11::none());
    }

    static int64_t get_dispatch_buffer_size(const int& num_max_tokens_per_rank,
                                            const int& hidden, const int& num_sf_packs,
                                            const int& num_topk, const int& element_size,
                                            const int& num_ranks) {
        const auto token_layout = get_dispatch_token_layout(
            hidden * element_size, num_sf_packs, num_topk);
        const auto recv_buffer_layout = layout::BufferLayout(
            token_layout, num_ranks, num_max_tokens_per_rank);
        return recv_buffer_layout.get_num_bytes();
    }

    static int64_t get_combine_buffer_size(const int& num_ranks,
                                           const int& num_max_tokens_per_rank,
                                           const int& hidden, const int& num_topk) {
        const auto token_layout = get_combine_token_layout(
            hidden * static_cast<int>(sizeof(uint16_t)), num_topk);
        const auto reduce_buffer_layout = layout::BufferLayout(
            token_layout, std::min(num_topk, num_ranks), num_max_tokens_per_rank);
        return reduce_buffer_layout.get_num_bytes();
    }

    static int64_t calculate_buffer_size(const int& num_ranks,
                                         const int& num_max_tokens_per_rank, const int& hidden,
                                         int num_topk, const bool& use_fp8_dispatch) {
        EP_HOST_ASSERT(num_ranks > 0);
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
        EP_HOST_ASSERT(hidden > 0);
        EP_HOST_ASSERT(num_topk >= 0);

        // NOTES: dispatch requires `kNumTopk <= 32`, so reserve the maximum when top-k is unspecified
        num_topk = num_topk == 0 ? 32 : num_topk;

        // Dispatch layout
        // NOTES: approximate the SF pack count because the size-hint API does not receive an SF tensor
        const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0;
        const auto elem_size = use_fp8_dispatch ? sizeof(c10::Float8_e4m3fn) : sizeof(c10::BFloat16);
        const auto num_dispatch_bytes = get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, elem_size, num_ranks);

        // Combine layout
        const auto num_combine_bytes = get_combine_buffer_size(
            num_ranks, num_max_tokens_per_rank, hidden, num_topk);

        // Return the maximum of those layouts, aligned to 2 MB
        return math::align(std::max(num_dispatch_bytes, num_combine_bytes), kNumAllocationAlignmentBytes);
    }
};

}  // namespace deep_ep
