#pragma once

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../comm/shmem.hpp"
#include "../kernels/ep/dispatch_low_latency_policy.hpp"
#include "../utils/event.hpp"

namespace deep_ep {

struct DispatchLowLatencyLaunch {
    std::shared_ptr<comm::ShmemArena> arena;
    torch::Tensor peer_buffers;

    void* buffer() const { return arena->data(); }
};

class DispatchLowLatencyState {
    std::string group_name;
    int64_t num_buffer_bytes;
    std::shared_ptr<comm::ShmemArena> arena;
    torch::Tensor peer_buffers;
    std::optional<EventHandle> last_event;
    std::shared_ptr<DispatchLowLatencyLaunch> retained_launch;
    bool attempted = false;
    bool pending = false;
    bool failed = false;
    bool group_capable = false;

public:
    DispatchLowLatencyState(std::string group_name, int64_t num_buffer_bytes):
        group_name(std::move(group_name)), num_buffer_bytes(num_buffer_bytes) {}

    struct FailureGuard {
        DispatchLowLatencyState* owner;
        int exceptions = std::uncaught_exceptions();
        ~FailureGuard() {
            if (std::uncaught_exceptions() > exceptions)
                owner->fail();
        }
    };

    FailureGuard failure_guard() { return {this}; }

    void initialize(const comm::Context& context) {
        const bool supported = comm::ShmemArena::compiled() and runtime->get_npu_arch() == 3510 and
            runtime->get_num_vec_cores() >= 16 and runtime->get_num_ubuf_bytes_per_vec_core() > 0;
        group_capable = comm::ShmemArena::agree_capability(context.hccl_context->comm,
            context.num_ranks, context.num_cpu_timeout_secs, supported);
    }

    std::shared_ptr<DispatchLowLatencyLaunch> prepare(
            const comm::Context& context, const torch::Tensor& x, int capacity, int hidden,
            int num_topk, int num_experts, int num_vec_cores, int num_ub_bytes, bool use_fp8) {
        if (failed)
            throw std::runtime_error("Low-latency dispatch failed; restart the distributed job");
        if (pending)
            throw std::runtime_error("Complete the outstanding dispatch epilogue before reusing the EPBuffer");
        const auto guard = failure_guard();
        if (arena)
            arena->check_available();
        if (last_event)
            last_event->current_stream_wait();
        const auto plan = get_dispatch_low_latency_plan(runtime->get_npu_arch(), context.num_ranks, capacity,
            hidden, num_topk, num_experts, num_vec_cores, num_ub_bytes, use_fp8);
        if (not plan or not group_capable or num_buffer_bytes < plan->buffer_bytes or
            not prefer_dispatch_low_latency(context.num_ranks, capacity, hidden, num_topk, num_experts))
            return nullptr;
        if (not attempted) {
            attempted = true;
            try {
                arena = comm::ShmemArena::try_create(group_name, context.rank_idx, context.num_ranks,
                    num_buffer_bytes, context.num_cpu_timeout_secs);
            } catch (...) {
                failed = true;
                throw;
            }
            if (arena) {
                std::vector<uint64_t> peers(context.num_ranks);
                for (int peer = 0; peer < context.num_ranks; ++peer) {
                    peers[peer] = reinterpret_cast<uint64_t>(arena->peer_ptr(peer));
                    EP_HOST_ASSERT(peers[peer] != 0 and peers[peer] % 32 == 0);
                }
                peer_buffers = torch::empty({context.num_ranks}, x.options().dtype(torch::kLong));
                ACL_CHECK(aclrtMemcpy(peer_buffers.data_ptr(), peers.size() * sizeof(uint64_t),
                    peers.data(), peers.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
            }
        }
        if (not arena)
            return nullptr;
        arena->check_available();
        auto state = std::make_shared<DispatchLowLatencyLaunch>();
        state->arena = arena;
        state->peer_buffers = peer_buffers;
        return state;
    }

    void begin(const std::shared_ptr<DispatchLowLatencyLaunch>& state) {
        pending = arena != nullptr;
        retained_launch = state;
    }

    void fail() noexcept {
        if (arena) {
            arena->mark_failed();
            failed = true;
        }
    }

    void record_completion() {
        if (arena)
            last_event.emplace();
        pending = false;
        retained_launch.reset();
    }

    void close() {
        if (arena) {
            arena->close();
            arena.reset();
        }
        peer_buffers = torch::Tensor();
        last_event.reset();
    }
};

}  // namespace deep_ep
