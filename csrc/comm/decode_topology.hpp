#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <acl/acl.h>
#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <deep_ep/common/exception.hpp>

namespace deep_ep::comm {

// EP rank numbers need not be contiguous on a physical server. Collect the
// launcher's local server ID once, in communicator rank order, before allocating
// the symmetric relay region. The default rank-direct buffer reserves no relay space.
struct DecodeTopology {
    bool group_enabled = false;
    std::vector<int> server_ids;
    int64_t scratch_bytes = 0;

    static DecodeTopology discover(const std::string& group, int world, int timeout,
                                   int64_t scratch_offset, int64_t scratch_hint, bool capable) {
        constexpr int max_timeout_secs = 3600;
        constexpr int64_t unspecified = -1, invalid = -2;
        int64_t id = unspecified;
        if (const char* value = std::getenv("EP_DECODE_SERVER_ID"); value and *value) {
            const auto end = value + std::strlen(value);
            int parsed;
            const auto result = std::from_chars(value, end, parsed);
            id = result.ec == std::errc{} and result.ptr == end and parsed >= 0 ? parsed : invalid;
        }
        // One setup collective even if this rank failed to reserve its per-call
        // vote resources. All peers must choose the same transport and arena.
        using Record = std::array<int64_t, 4>;
        const Record local{id, scratch_offset, scratch_hint, capable and timeout <= max_timeout_secs};
        std::vector<Record> peers(world);
        const size_t bytes = peers.size() * sizeof(Record);
        HcclComm comm;
        HCCL_CHECK(HcclCommGetHandleWithName(group.c_str(), &comm));
        void* raw = nullptr;
        ACL_CHECK(aclrtMalloc(&raw, bytes + sizeof(Record), ACL_MEM_MALLOC_NORMAL_ONLY));
        std::unique_ptr<void, decltype(&aclrtFree)> allocation(raw, &aclrtFree);
        auto* send = static_cast<char*>(raw) + bytes;
        ACL_CHECK(aclrtMemcpy(send, sizeof(local), local.data(), sizeof(local), ACL_MEMCPY_HOST_TO_DEVICE));
        auto stream = c10_npu::getCurrentNPUStream();
        const auto status = HcclAllGather(send, raw, local.size(), HCCL_DATA_TYPE_INT64, comm, stream);
        // A timed-out HCCL stream can still be using its buffers.
        allocation.release();
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(stream, std::min(timeout, max_timeout_secs) * 1000));
        allocation.reset(raw);
        HCCL_CHECK(status);
        ACL_CHECK(aclrtMemcpy(peers.data(), bytes, raw, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        DecodeTopology topology;
        topology.group_enabled = std::all_of(peers.begin(), peers.end(), [](const auto& p) { return p[3] != 0; });
        if (std::all_of(peers.begin(), peers.end(), [](const auto& p) { return p[0] == unspecified; }))
            return topology;
        for (const auto& p : peers) {
            if (p[0] < 0)
                throw std::invalid_argument("Set EP_DECODE_SERVER_ID to a nonnegative integer on every EP rank");
            if (p[1] != scratch_offset)
                throw std::invalid_argument("Decode relay requires the same EP+LB arena size on every rank");
            topology.server_ids.push_back(static_cast<int>(p[0]));
            topology.scratch_bytes = std::max(topology.scratch_bytes, p[2]);
        }
        auto distinct = topology.server_ids;
        std::sort(distinct.begin(), distinct.end());
        const int servers = static_cast<int>(std::unique(distinct.begin(), distinct.end()) - distinct.begin());
        if (not topology.group_enabled or servers == 1 or servers == world) {
            topology.server_ids.clear();
            topology.scratch_bytes = 0;
        }
        return topology;
    }
};

}  // namespace deep_ep::comm
