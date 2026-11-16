#pragma once

#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "../comm/decode_agreement.hpp"
#include "../kernels/comm/api.hpp"
#include "../kernels/ep/dispatch_low_latency_policy.hpp"
#include "../utils/event.hpp"

namespace deep_ep {

class EPLowLatencyState {
    struct Completion {
        c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
        EventHandle event{stream};
    };

    int64_t num_buffer_bytes;
    std::optional<Completion> last_completion;
    bool pending = false;
    bool failed = false;
    bool group_enabled = false;
    comm::DecodeAgreement agreement;

    void wait_for_workspace() {
        if (last_completion and last_completion->stream != c10_npu::getCurrentNPUStream())
            last_completion->event.current_stream_wait();
    }

public:
    explicit EPLowLatencyState(int64_t num_buffer_bytes): num_buffer_bytes(num_buffer_bytes) {}

    struct FailureGuard {
        EPLowLatencyState* owner;
        int exceptions = std::uncaught_exceptions();

        ~FailureGuard() {
            if (std::uncaught_exceptions() > exceptions)
                owner->fail();
        }
    };

    FailureGuard failure_guard() {
        return {this};
    }

    void initialize(const comm::Context& context) {
        if (not is_decode_capacity(context.num_ranks, 1))
            return;
        const auto* soc = aclrtGetSocName();
        const bool local = soc and std::string_view(soc) == "Ascend950DT_9582" and
            runtime->get_npu_arch() == 3510 and
            context.num_gpu_timeout_cycles == 100 * comm::kNumSystemCyclesPerSecond and
            runtime->get_num_vec_cores() >= 64 and runtime->get_num_vec_cores_per_ai_core() == 2 and
            runtime->get_num_ubuf_bytes_per_vec_core() == 216 * 1024 and agreement.initialize(context.num_ranks);
        group_enabled = comm::DecodeAgreement::agree_capability(
            context.hccl_context->comm, context.num_ranks, context.num_cpu_timeout_secs, local);
    }

    void require_idle() const {
        if (failed)
            throw std::runtime_error("Low-latency EP operation failed; restart the distributed job");
        if (pending)
            throw std::runtime_error("Complete the outstanding EP epilogue before reusing or destroying the EPBuffer");
    }

    bool prepare_dispatch(
            const comm::Context& context, int capacity, int hidden,
            int num_topk, int num_experts, int num_vec_cores, int num_ub_bytes, bool use_fp8,
            bool cached_mode, bool barrier_in_prologue, bool matched_options) {
        require_idle();
        const auto guard = failure_guard();
        wait_for_workspace();
        const auto world = context.num_ranks;
        // Only shared wire geometry may skip the collective. Rank-local options vote below.
        if (not group_enabled or not is_decode_geometry(world, capacity, hidden, num_topk, num_experts) or
            comm::DecodeAgreement::is_capturing())
            return false;

        const auto plan = get_dispatch_low_latency_plan(runtime->get_npu_arch(), world, capacity,
            hidden, num_topk, num_experts, num_vec_cores, num_ub_bytes, use_fp8);
        const bool match = plan and num_buffer_bytes >= plan->buffer_bytes and
            low_latency_transport_enabled() and matched_options;
        const uint16_t proposal = match ? 1 + 2 * cached_mode + barrier_in_prologue : 0;
        return agreement.matches(context.hccl_context->comm, world, context.num_cpu_timeout_secs, proposal);
    }

    bool prepare_combine(
            const comm::Context& context, int capacity, int hidden, int num_topk,
            int num_experts, int num_vec_cores, int num_ub_bytes, int batch, bool matched_options,
            bool barrier_in_prologue) {
        require_idle();
        const auto guard = failure_guard();
        wait_for_workspace();
        const auto world = context.num_ranks;
        if (not group_enabled or not is_decode_geometry(world, capacity, hidden, num_topk, num_experts) or
            comm::DecodeAgreement::is_capturing())
            return false;

        const auto threshold = mte_bs_threshold_override();
        const bool match = (threshold < 0 or capacity <= threshold) and
            low_latency_transport_enabled() and matched_options and is_decode_batch(batch, capacity) and
            is_decode_shape(world, capacity, hidden, num_topk, num_experts, num_vec_cores, num_ub_bytes, false);
        const uint16_t proposal = match ? 1 + barrier_in_prologue : 0;
        return agreement.matches(context.hccl_context->comm, world, context.num_cpu_timeout_secs, proposal);
    }

    void begin() {
        pending = group_enabled;
    }

    void fail() noexcept {
        if (group_enabled)
            failed = true;
    }

    void record_completion() {
        if (not group_enabled)
            return;
        // Both transports use the native EP buffer and workspace across streams.
        last_completion.emplace();
        pending = false;
    }

    void close() {
        require_idle();
        last_completion.reset();
    }
};

}  // namespace deep_ep
