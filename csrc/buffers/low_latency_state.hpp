#pragma once

#include <atomic>
#include <exception>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "../kernels/comm/api.hpp"
#include "../kernels/ep/low_latency_policy.hpp"
#include "../utils/event.hpp"

namespace deep_ep {

class EPLowLatencyState {
    struct Completion {
        c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
        EventHandle event{stream};
    };

    std::optional<Completion> last_completion;
    // The queue can still read these allocations after the issuing NPU kernel exits.
    struct Resources {
        std::shared_ptr<comm::Context> context;
        torch::Tensor source, staging;
    };
    std::unique_ptr<Resources> resources = std::make_unique<Resources>();
    bool pending = false;
    bool failed = false;
    bool mte_enabled = false;
    bool server_relay = false;
    bool used_mte = false;

    static bool stream_is_captured() {
        aclmdlRICaptureStatus status{};
        aclmdlRI model = nullptr;
        ACL_CHECK(aclmdlRICaptureGetInfo(c10_npu::getCurrentNPUStream(), &status, &model));
        return status != ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    }

    void release_sources() {
        resources->source = torch::Tensor();
        resources->staging = torch::Tensor();
    }

public:
    static comm::DecodeTopology local_topology(int world) {
        if (not is_decode_rank_count(world))
            return comm::DecodeTopology(0);
        const auto* soc = aclrtGetSocName();
        const bool capable = soc and std::string_view(soc) == "Ascend950DT_9582" and
            runtime->get_npu_arch() == 3510 and
            runtime->get_num_vec_cores() >= 64 and runtime->get_num_vec_cores_per_ai_core() == 2 and
            runtime->get_num_ubuf_bytes_per_vec_core() == 216 * 1024;
        return comm::DecodeTopology(capable ? runtime->get_num_vec_cores() : 0);
    }

    void attach_context(const std::shared_ptr<comm::Context>& context, const comm::DecodeTopology& topology) {
        resources->context = context;
        if (not topology.server_ids.empty()) {
            auto& workspace = *static_cast<layout::EPSignals*>(context->workspace);
            ACL_CHECK(aclrtMemcpy(workspace.decode_server_ids, sizeof(workspace.decode_server_ids),
                                  topology.server_ids.data(), topology.server_ids.size() * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE));
            server_relay = true;
        }
        mte_enabled = topology.mte_enabled;
    }

    ~EPLowLatencyState() {
        // An unconfirmed collective may still access the registered window and URMA
        // sources. Only process teardown can reclaim them safely in this state.
        if (pending or failed)
            resources.release();
    }

    struct FailureGuard {
        EPLowLatencyState* owner;
        bool armed;
        int exceptions = std::uncaught_exceptions();

        void begin_submission() {
            owner->pending = true;
            armed = true;
        }

        ~FailureGuard() {
            if (std::uncaught_exceptions() > exceptions) {
                if (armed)
                    owner->fail();
                else
                    owner->release_sources();
            }
        }
    };

    static auto make_deferred_epilogue(auto epilogue) {
        // Copies of the returned hook belong to the same submission.
        return [epilogue = std::move(epilogue), consumed = std::make_shared<std::atomic_bool>(false)] {
            const bool already_called = consumed->exchange(true);
            EP_HOST_ASSERT(not already_called and "Deferred EP epilogue was already called");
            return epilogue();
        };
    }

    FailureGuard epilogue_guard(bool deferred) {
        // The caller can hold the callback beyond the lifetime of the buffer.
        if (deferred) {
            require_healthy();
            resources->context->check_available();
        }
        return {this, true};
    }

    FailureGuard submission_guard(const torch::Tensor& source, const torch::Tensor& staging) {
        if (staging.defined()) {
            resources->source = source;
            resources->staging = staging;
        }
        return {this, false};
    }

    bool server_relay_enabled() const { return server_relay; }

    void require_healthy() const {
        EP_HOST_ASSERT(not failed and "EP operation failed; restart the distributed job");
    }

    void prepare() {
        require_healthy();
        EP_HOST_ASSERT(not pending and "Complete the outstanding EP epilogue before reusing the EPBuffer");
        if (last_completion and last_completion->stream != c10_npu::getCurrentNPUStream()) {
            const FailureGuard guard{this, true};
            last_completion->event.current_stream_wait();
        }
    }

    bool prepare_dispatch(int world, int capacity, int hidden, int num_topk, int num_experts,
                          int num_vec_cores, int num_ub_bytes, bool cached_mode) const {
        return mte_enabled and supports_dispatch_mte(world, capacity, hidden, num_topk, num_experts,
            num_vec_cores, num_ub_bytes, cached_mode) and not stream_is_captured();
    }

    bool prepare_combine(int world, int capacity, int hidden, int num_topk, int num_experts,
                         int num_vec_cores, int num_ub_bytes) const {
        return mte_enabled and supports_combine_mte(world, capacity, hidden, num_topk, num_experts,
            num_vec_cores, num_ub_bytes) and not stream_is_captured();
    }

    void fail() noexcept { failed = true; }

    // Once MTE uses this buffer, follow the last writer even across native epilogues.
    void record_completion(bool mte, bool async_with_compute_stream) {
        used_mte = used_mte or mte;
        if (used_mte or async_with_compute_stream)
            last_completion.emplace();
        else
            last_completion.reset();
        pending = false;
        if (resources->staging.defined())
            release_sources();
    }

    void clear_after_drain() {
        used_mte = false;
        record_completion(false, false);
    }

    EventHandle completion_event() const { return last_completion.value().event; }
};

}  // namespace deep_ep
