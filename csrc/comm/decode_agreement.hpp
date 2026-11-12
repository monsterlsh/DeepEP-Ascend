#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

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
