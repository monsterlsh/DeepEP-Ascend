#pragma once

#include <algorithm>
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
    static constexpr size_t kRecvOffset = 64;
    aclrtStream stream = nullptr;
    void* host = nullptr;
    void* device = nullptr;
    bool in_flight = false;

public:
    DecodeAgreement() = default;
    DecodeAgreement(const DecodeAgreement&) = delete;
    DecodeAgreement& operator=(const DecodeAgreement&) = delete;

    bool initialize(int world) {
        const size_t bytes = kRecvOffset + world * sizeof(uint16_t);
        return aclrtCreateStream(&stream) == ACL_SUCCESS and
               aclrtMallocHost(&host, bytes) == ACL_SUCCESS and
               aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) == ACL_SUCCESS;
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
        auto host_recv = reinterpret_cast<uint16_t*>(static_cast<char*>(host) + kRecvOffset);
        auto device_recv = static_cast<char*>(device) + kRecvOffset;
        *static_cast<uint16_t*>(host) = proposal;
        in_flight = true;
        ACL_CHECK(aclrtMemcpyAsync(device, sizeof(proposal), host, sizeof(proposal), ACL_MEMCPY_HOST_TO_DEVICE, stream));
        HCCL_CHECK(HcclAllGather(device, device_recv, 1, HCCL_DATA_TYPE_UINT16, comm, stream));
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
