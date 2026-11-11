#pragma once

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
    static constexpr uint16_t kDispatch = 1, kCombine = 2;

    std::string group_name;
    int64_t num_buffer_bytes;
    std::shared_ptr<comm::ShmemArena> arena;
    torch::Tensor peer_buffers;
    std::optional<EventHandle> last_event;
    std::shared_ptr<EPLowLatencyLaunch> retained_launch;
    bool attempted = false;
    bool pending = false;
    bool failed = false;
    uint16_t group_capabilities = 0;

    void initialize_arena(const comm::Context& context, const torch::Tensor& x) {
        attempted = true;
        try {
            arena = comm::ShmemArena::try_create(group_name, context.rank_idx, context.num_ranks,
                                                 num_buffer_bytes, context.num_cpu_timeout_secs);
        } catch (...) {
            failed = true;
            throw;
        }
        if (not arena)
            return;

        std::vector<uint64_t> peers(context.num_ranks);
        for (int peer = 0; peer < context.num_ranks; ++ peer) {
            peers[peer] = reinterpret_cast<uint64_t>(arena->peer_ptr(peer));
            EP_HOST_ASSERT(peers[peer] != 0 and peers[peer] % 32 == 0);
        }
        const auto peer_bytes = peers.size() * sizeof(uint64_t);
        peer_buffers = torch::empty({context.num_ranks}, x.options().dtype(torch::kLong));
        ACL_CHECK(aclrtMemcpy(peer_buffers.data_ptr(), peer_bytes, peers.data(), peer_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    std::shared_ptr<EPLowLatencyLaunch> prepare_launch(const comm::Context& context, const torch::Tensor& x,
                                                       bool eligible) {
        require_idle();
        const auto guard = failure_guard();
        if (arena)
            arena->check_available();
        if (last_event)
            last_event->current_stream_wait();
        if (not eligible)
            return nullptr;
        if (not attempted)
            initialize_arena(context, x);
        if (not arena)
            return nullptr;
        return std::make_shared<EPLowLatencyLaunch>(arena, peer_buffers);
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
        uint16_t local = 0;
        if (comm::ShmemArena::compiled() and runtime->get_npu_arch() == 3510) {
            const auto cores = runtime->get_num_vec_cores();
            const auto ub_bytes = runtime->get_num_ubuf_bytes_per_vec_core();
            if (cores >= 16 and ub_bytes > 0)
                local |= kDispatch;
            if (cores >= 64 and runtime->get_num_vec_cores_per_ai_core() == 2 and ub_bytes >= 190 * 1024)
                local |= kCombine;
        }
        group_capabilities = comm::ShmemArena::agree_capabilities(
            context.hccl_context->comm, context.num_ranks, context.num_cpu_timeout_secs, local);
    }

    void require_idle() const {
        if (failed)
            throw std::runtime_error("Low-latency EP operation failed; restart the distributed job");
        if (pending)
            throw std::runtime_error("Complete the outstanding EP epilogue before reusing or destroying the EPBuffer");
    }

    std::shared_ptr<EPLowLatencyLaunch> prepare_dispatch(
            const comm::Context& context, const torch::Tensor& x, int capacity, int hidden,
            int num_topk, int num_experts, int num_vec_cores, int num_ub_bytes, bool use_fp8,
            bool cached_mode, bool barrier_in_prologue) {
        bool eligible = false;
        if ((group_capabilities & kDispatch) and
            capacity <= get_dispatch_low_latency_threshold(context.num_ranks, cached_mode, barrier_in_prologue)) {
            const auto plan = get_dispatch_low_latency_plan(
                runtime->get_npu_arch(), context.num_ranks, capacity, hidden, num_topk, num_experts, num_vec_cores, num_ub_bytes, use_fp8);
            eligible = plan and num_buffer_bytes >= plan->buffer_bytes;
        }
        return prepare_launch(context, x, eligible);
    }

    // group_eligible may only depend on shape and options shared by all ranks.
    std::shared_ptr<EPLowLatencyLaunch> prepare_combine(
            const comm::Context& context, const torch::Tensor& x, bool group_eligible) {
        return prepare_launch(context, x, group_eligible and (group_capabilities & kCombine));
    }

    void begin(const std::shared_ptr<EPLowLatencyLaunch>& state) {
        pending = true;
        retained_launch = state;
    }

    void fail() noexcept {
        if (arena)
            arena->mark_failed();
        if (arena or pending or last_event)
            failed = true;
    }

    void record_completion() {
        // Native and SHMEM kernels share the EP workspace, even before the first arena allocation.
        if (arena or (group_capabilities and not attempted))
            last_event.emplace();
        else
            last_event.reset();
        pending = false;
        retained_launch.reset();
    }

    void close() {
        require_idle();
        if (arena) {
            arena->close();
            arena.reset();
        }
        peer_buffers = torch::Tensor();
        last_event.reset();
    }
};

}  // namespace deep_ep
