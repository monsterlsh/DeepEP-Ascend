#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <acl/acl.h>
#include <hccl/hccl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <deep_ep/common/exception.hpp>

namespace deep_ep::comm {

// This host decision needs only shape/options metadata. No device data or producer event is an input.
class DecodeAgreement {
    // Zero vetoes the fast path. A dispatch encodes fresh/cached and barrier
    // placement as 1..4; combine uses 1..2. Keep each HCCL input aligned.
    struct alignas(64) Vote { uint16_t value; };
    static constexpr std::array<Vote, 5> votes{{{0}, {1}, {2}, {3}, {4}}};
    aclrtStream stream = nullptr;
    void* host = nullptr;
    void* device = nullptr;
    bool in_flight = false;

public:
    DecodeAgreement() = default;
    DecodeAgreement(const DecodeAgreement&) = delete;
    DecodeAgreement& operator=(const DecodeAgreement&) = delete;

    bool initialize(int world) {
        const size_t recv_bytes = world * sizeof(uint16_t);
        return aclrtCreateStream(&stream) == ACL_SUCCESS and
               aclrtMallocHost(&host, recv_bytes) == ACL_SUCCESS and
               aclrtMalloc(&device, sizeof(votes) + recv_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) == ACL_SUCCESS and
               aclrtMemcpy(device, sizeof(votes), votes.data(), sizeof(votes), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    }

    // Even a rank that could not allocate the per-call vote resources must
    // participate in the initial capability vote so the group can use URMA.
    static bool agree_capability(HcclComm comm, int world, int timeout, bool capable) {
        constexpr int max_timeout_secs = 3600;
        const uint16_t local = capable and timeout <= max_timeout_secs;
        const auto bytes = static_cast<size_t>(world) * sizeof(local);
        std::vector<uint16_t> peers(world);
        void* raw = nullptr;
        ACL_CHECK(aclrtMalloc(&raw, bytes + sizeof(local), ACL_MEM_MALLOC_NORMAL_ONLY));
        std::unique_ptr<void, decltype(&aclrtFree)> owned(raw, &aclrtFree);
        auto* send = static_cast<char*>(raw) + bytes;
        ACL_CHECK(aclrtMemcpy(send, sizeof(local), &local, sizeof(local), ACL_MEMCPY_HOST_TO_DEVICE));
        const auto stream = c10_npu::getCurrentNPUStream();
        const auto status = HcclAllGather(send, raw, 1, HCCL_DATA_TYPE_UINT16, comm, stream);
        // A timed-out collective can still access its device allocation.
        owned.release();
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(stream, std::min(timeout, max_timeout_secs) * 1000));
        owned.reset(raw);
        HCCL_CHECK(status);
        ACL_CHECK(aclrtMemcpy(peers.data(), bytes, raw, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        return std::all_of(peers.begin(), peers.end(), [](uint16_t vote) { return vote != 0; });
    }

    static bool is_capturing() {
        aclmdlRICaptureStatus status{};
        aclmdlRI model = nullptr;
        ACL_CHECK(aclmdlRICaptureGetInfo(c10_npu::getCurrentNPUStream(), &status, &model));
        return status != ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    }

    bool matches(HcclComm comm, int world, int timeout, uint16_t proposal) {
        if (in_flight)
            throw std::runtime_error("Previous EP transport agreement failed; restart the distributed job");
        const size_t bytes = world * sizeof(proposal);
        auto* host_recv = static_cast<uint16_t*>(host);
        auto* device_recv = static_cast<char*>(device) + sizeof(votes);
        // Select an immutable device value. Every rank still votes on this call,
        // including a rank with an unsupported proposal or different local inputs.
        if (proposal >= votes.size())
            proposal = 0;
        auto* send = static_cast<char*>(device) + proposal * sizeof(Vote);
        in_flight = true;
        HCCL_CHECK(HcclAllGather(send, device_recv, 1, HCCL_DATA_TYPE_UINT16, comm, stream));
        if (proposal != 0)
            ACL_CHECK(aclrtMemcpyAsync(host_recv, bytes, device_recv, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream));
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(stream, timeout * 1000));
        in_flight = false;
        return proposal != 0 and std::all_of(host_recv, host_recv + world,
                                              [proposal](uint16_t peer) { return peer == proposal; });
    }

    ~DecodeAgreement() {
        // A timed-out collective may still access staging. Its process must restart.
        if (not in_flight) {
            if (device) aclrtFree(device);
            if (host) aclrtFreeHost(host);
            if (stream) aclrtDestroyStream(stream);
        }
    }
};

}  // namespace deep_ep::comm
