#pragma once

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../comm/decode_agreement.hpp"
#include "../comm/shmem.hpp"
#include "../kernels/comm/api.hpp"
#include "../kernels/ep/dispatch_low_latency_policy.hpp"
#include "../utils/event.hpp"
#include "../utils/tensor.hpp"

namespace deep_ep {

struct EPLowLatencyLaunch {
    std::shared_ptr<comm::ShmemArena> arena;
    torch::Tensor peer_buffers;

    void* buffer() const {
        return arena->data();
    }
};

class EPLowLatencyState {
    std::string group_name;
    int64_t num_buffer_bytes;
    EPLowLatencyLaunch launch;
    std::optional<EventHandle> last_event;
    bool attempted = false;
    bool pending = false;
    bool failed = false;
    bool group_enabled = false;
    comm::DecodeAgreement agreement;

    void initialize_arena(const comm::Context& context, const torch::Tensor& x) {
        attempted = true;
        try {
            launch.arena = comm::ShmemArena::try_create(group_name, context.rank_idx, context.num_ranks,
                                                        num_buffer_bytes, context.num_cpu_timeout_secs);
        } catch (...) {
            failed = true;
            throw;
        }
        if (not launch.arena)
            return;

        std::vector<uint64_t> peers(context.num_ranks);
        for (int peer = 0; peer < context.num_ranks; ++ peer) {
            peers[peer] = reinterpret_cast<uint64_t>(launch.arena->peer_ptr(peer));
            EP_HOST_ASSERT(peers[peer] != 0 and peers[peer] % 32 == 0);
        }
        const auto peer_bytes = peers.size() * sizeof(uint64_t);
        launch.peer_buffers = torch::empty({context.num_ranks}, x.options().dtype(torch::kLong));
        ACL_CHECK(aclrtMemcpy(launch.peer_buffers.data_ptr(), peer_bytes, peers.data(), peer_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    const EPLowLatencyLaunch* prepare_launch(const comm::Context& context, const torch::Tensor& x,
                                             bool eligible) {
        const auto guard = failure_guard();
        if (launch.arena)
            launch.arena->check_available();
        if (last_event)
            last_event->current_stream_wait();
        if (not eligible)
            return nullptr;
        if (not attempted)
            initialize_arena(context, x);
        return launch.arena ? &launch : nullptr;
    }

public:
    EPLowLatencyState(std::string group_name, int64_t num_buffer_bytes):
        group_name(std::move(group_name)), num_buffer_bytes(num_buffer_bytes) {}

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
            comm::ShmemArena::compiled() and runtime->get_npu_arch() == 3510 and
            context.num_gpu_timeout_cycles == 100 * comm::kNumSystemCyclesPerSecond and
            runtime->get_num_vec_cores() >= 64 and runtime->get_num_vec_cores_per_ai_core() == 2 and
            runtime->get_num_ubuf_bytes_per_vec_core() == 216 * 1024 and agreement.initialize(context.num_ranks);
        group_enabled = comm::ShmemArena::agree_capabilities(
            context.hccl_context->comm, context.num_ranks, context.num_cpu_timeout_secs, local);
    }

    void require_idle() const {
        if (failed)
            throw std::runtime_error("Low-latency EP operation failed; restart the distributed job");
        if (pending)
            throw std::runtime_error("Complete the outstanding EP epilogue before reusing or destroying the EPBuffer");
    }

    const EPLowLatencyLaunch* prepare_dispatch(
            const comm::Context& context, const torch::Tensor& x, int capacity, int hidden,
            int num_topk, int num_experts, int num_vec_cores, int num_ub_bytes, bool use_fp8,
            bool cached_mode, bool barrier_in_prologue, bool matched_options) {
        require_idle();
        bool eligible = false;
        // Capacity and payload geometry are shared EP wire-layout arguments. B and the
        // optional tensors are rank-local, so they only affect the vote, never participation.
        if (group_enabled and is_decode_capacity(context.num_ranks, capacity) and
            hidden == 7168 and num_topk == 8 and num_experts == 128 and
            capacity <= get_dispatch_low_latency_threshold(context.num_ranks, false, true) and
            not comm::DecodeAgreement::is_capturing()) {
            const auto guard = failure_guard();
            const auto plan = get_dispatch_low_latency_plan(runtime->get_npu_arch(), context.num_ranks, capacity,
                hidden, num_topk, num_experts, num_vec_cores, num_ub_bytes, use_fp8);
            const bool match = plan and num_buffer_bytes >= plan->buffer_bytes and low_latency_transport_enabled() and
                matched_options and x.size(0) == std::max(1, capacity - context.rank_idx) and
                capacity <= get_dispatch_low_latency_threshold(context.num_ranks, cached_mode, barrier_in_prologue);
            const uint16_t proposal = match ? 1 + 2 * cached_mode + barrier_in_prologue : 0;
            eligible = agreement.matches(context.hccl_context->comm, context.num_ranks, context.num_cpu_timeout_secs, proposal);
        }
        return prepare_launch(context, x, eligible);
    }

    const EPLowLatencyLaunch* prepare_combine(
            const comm::Context& context, const torch::Tensor& x, int capacity, int hidden, int num_topk,
            int num_experts, int num_vec_cores, int num_ub_bytes, int batch, bool matched_options,
            bool barrier_in_prologue, bool& native_mte) {
        require_idle();
        const auto world = context.num_ranks;
        const bool mixed = (world == 8 and capacity == 64) or (world == 16 and capacity == 32);
        const bool shmem = combine_shmem_capacity(world, capacity, -1);
        bool eligible = false;
        native_mte = false;
        if (group_enabled and (shmem or mixed) and
            hidden == 7168 and num_topk == 8 and num_experts == 128 and not comm::DecodeAgreement::is_capturing()) {
            const auto guard = failure_guard();
            const auto threshold = mte_bs_threshold_override();
            const bool enabled = shmem ? combine_shmem_capacity(world, capacity, threshold) :
                (threshold < 0 or threshold >= (world == 8 ? 60 : 17));
            const bool match = enabled and low_latency_transport_enabled() and matched_options and
                is_decode_shape(world, capacity, hidden, num_topk, num_experts, num_vec_cores, num_ub_bytes, false) and
                batch == std::max(1, capacity - context.rank_idx);
            const uint16_t proposal = match ? 1 + barrier_in_prologue : 0;
            eligible = agreement.matches(context.hccl_context->comm, world, context.num_cpu_timeout_secs, proposal);
            native_mte = eligible and mixed and combine_native_mte_rank(world, capacity, context.rank_idx, batch, threshold);
        }
        return prepare_launch(context, x, eligible and shmem);
    }

    void begin() {
        pending = group_enabled;
    }

    void fail() noexcept {
        if (launch.arena)
            launch.arena->mark_failed();
        if (launch.arena or pending or last_event)
            failed = true;
    }

    void record_completion() {
        if (not group_enabled)
            return;
        // Native and SHMEM kernels share the EP workspace, even before the first arena allocation.
        if (launch.arena or not attempted)
            last_event.emplace();
        else
            last_event.reset();
        pending = false;
    }

    void close() {
        require_idle();
        if (launch.arena) {
            launch.arena->close();
            launch.arena.reset();
        }
        launch.peer_buffers = torch::Tensor();
        last_event.reset();
    }
};

}  // namespace deep_ep
