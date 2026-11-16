#include <cassert>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using aclrtStream = void*;
using HcclComm = void*;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 2;
constexpr int HCCL_DATA_TYPE_UINT16 = 0;
constexpr int HCCL_DATA_TYPE_INT64 = 1;
using TopologyRecord = std::array<int64_t, 4>;
static std::vector<TopologyRecord> peer_topology;
static bool fail_collective = false, fail_wait = false;
static uint16_t remote_capabilities = UINT16_MAX;
// DecodeAgreement uses a dedicated stream and pinned host staging for a rank-wide vote.
constexpr int ACL_SUCCESS = 0;
using aclmdlRI = void*;
enum aclmdlRICaptureStatus { ACL_MODEL_RI_CAPTURE_STATUS_NONE, ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE };
static bool capturing = false, fail_pinned_alloc = false, fail_device_alloc = false;
static int streams = 0, pinned = 0, devices = 0;
static bool fail_init_copy = false;
static int readbacks = 0, gathers = 0;
int aclrtCreateStream(aclrtStream* stream) { *stream = reinterpret_cast<void*>(4); ++streams; return 0; }
int aclrtDestroyStream(aclrtStream stream) { assert(stream == reinterpret_cast<void*>(4)); --streams; return 0; }
int aclrtMallocHost(void** ptr, size_t bytes) {
    if (fail_pinned_alloc) return 1;
    *ptr = std::malloc(bytes);
    if (*ptr) ++pinned;
    return *ptr ? 0 : 1;
}
int aclrtFreeHost(void* ptr) { std::free(ptr); --pinned; return 0; }
int aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus* status, aclmdlRI*) {
    assert(stream == reinterpret_cast<void*>(3));
    *status = capturing ? ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE : ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    return 0;
}

int aclrtMalloc(void** pointer, size_t bytes, int) {
    if (fail_device_alloc) return 1;
    *pointer = std::malloc(bytes);
    if (*pointer) ++devices;
    return *pointer ? 0 : 1;
}

int aclrtFree(void* pointer) {
    --devices;
    std::free(pointer);
    return 0;
}

int aclrtMemcpy(void* dst, size_t capacity, const void* src, size_t bytes, int kind) {
    assert(bytes <= capacity);
    if (fail_init_copy && kind == ACL_MEMCPY_HOST_TO_DEVICE) return 1;
    std::memcpy(dst, src, bytes);
    return 0;
}

int aclrtMemcpyAsync(void* dst, size_t capacity, const void* src, size_t bytes, int kind, aclrtStream stream) {
    assert(stream == reinterpret_cast<void*>(4));
    assert(kind == ACL_MEMCPY_DEVICE_TO_HOST);
    ++readbacks;
    return aclrtMemcpy(dst, capacity, src, bytes, kind);
}

int aclrtSynchronizeStreamWithTimeout(aclrtStream, int) {
    return fail_wait ? 1 : 0;
}

int HcclCommGetHandleWithName(const char* group, HcclComm* comm) {
    assert(std::string(group) == "ep");
    *comm = reinterpret_cast<HcclComm>(2);
    return 0;
}

int HcclAllGather(void* send, void* receive, uint64_t count, int dtype, HcclComm, aclrtStream) {
    ++gathers;
    if (dtype == HCCL_DATA_TYPE_INT64) {
        assert(count == 4);
        auto* output = static_cast<TopologyRecord*>(receive);
        std::memcpy(output, send, sizeof(TopologyRecord));
        std::memcpy(output + 1, peer_topology.data(), peer_topology.size() * sizeof(TopologyRecord));
        return fail_collective ? 1 : 0;
    }
    assert(count == 1);
    auto* peers = static_cast<uint16_t*>(receive);
    peers[0] = *static_cast<uint16_t*>(send);
    peers[1] = peers[0] & remote_capabilities;
    return fail_collective ? 1 : 0;
}
namespace c10_npu {

aclrtStream getCurrentNPUStream() {
    return reinterpret_cast<void*>(3);
}

}  // namespace c10_npu

#include "csrc/comm/decode_agreement.hpp"
#include "csrc/comm/decode_topology.hpp"

template <typename F>
void throws(F function) {
    bool caught = false;
    try {
        function();
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);
}

void check_topology() {
    using deep_ep::comm::DecodeTopology;
    const auto set_id = [](const char* id) {
#ifdef _WIN32
        assert(_putenv_s("EP_DECODE_SERVER_ID", id) == 0);
#else
        assert((*id ? setenv("EP_DECODE_SERVER_ID", id, 1) : unsetenv("EP_DECODE_SERVER_ID")) == 0);
#endif
    };
    const auto discover = [](const std::vector<int>& peers, bool same_offset = true,
                             bool capable = true, int timeout = 120, bool peers_capable = true) {
        peer_topology.clear();
        int64_t scratch = 128;
        for (int id : peers) {
            peer_topology.push_back({id, same_offset ? 1024 : 512, scratch, peers_capable});
            scratch *= 2;
        }
        const int before = gathers;
        auto topology = DecodeTopology::discover("ep", static_cast<int>(peers.size()) + 1, timeout, 1024, 16, capable);
        assert(gathers == before + 1);
        return topology;
    };
    set_id("");
    assert(discover({-1,-1,-1}).server_ids.empty());
    set_id("17");
    auto interleaved = discover({9,17,9});
    assert(interleaved.group_enabled && interleaved.server_ids == std::vector<int>({17,9,17,9})
           && interleaved.scratch_bytes == 512);
    assert(discover({17,17,17}).scratch_bytes == 0);
    assert(discover({9,13,21}).server_ids.empty());
    for (const char* local : {"", "unknown", "-2", "2147483648", "7x"}) {
        set_id(local);
        try { (void) discover({9,17,9}); assert(false); }
        catch (const std::invalid_argument&) {}
    }
    set_id("0");
    try { (void) discover({0,-1,17}); assert(false); }
    catch (const std::invalid_argument&) {}
    try { (void) discover({0,17,17}, false); assert(false); }
    catch (const std::invalid_argument&) {}
    set_id("");
    for (bool local : {true, false})
        for (bool peer : {true, false})
            assert(discover({-1}, true, local, 120, peer).group_enabled == (local && peer));
    assert(!discover({-1}, true, true, 7200).group_enabled);
    set_id("9");
    assert(discover({17,9}, true, true, 120, false).scratch_bytes == 0);
    fail_collective = true;
    throws([&] { discover({17,9}); });
    fail_collective = false;
    set_id("");
    assert(devices == 0);
}

int main() {
    check_topology();
    using deep_ep::comm::DecodeAgreement;
    const auto comm = reinterpret_cast<HcclComm>(2);

    assert(!DecodeAgreement::is_capturing());
    capturing = true;
    assert(DecodeAgreement::is_capturing());
    for (int failure : {0, 1, 2, 3}) {
        {
            DecodeAgreement agreement;
            fail_pinned_alloc = failure == 1;
            fail_device_alloc = failure == 2;
            fail_init_copy = failure == 3;
            assert(agreement.initialize(2) == (failure == 0));
            if (!failure) {
                for (uint16_t proposal : {0, 1, 2, 3, 4, 8}) {
                    const bool valid = proposal >= 1 && proposal <= 4;
                    const int reads_before = readbacks, gathers_before = gathers;
                    remote_capabilities = UINT16_MAX;
                    assert(agreement.matches(comm, 2, 120, proposal) == valid);
                    remote_capabilities = 0;
                    assert(!agreement.matches(comm, 2, 120, proposal));
                    assert(gathers == gathers_before + 2 && readbacks == reads_before + (valid ? 2 : 0));
                }
            }
        }
        assert(streams == 0 && pinned == 0 && devices == 0);
    }
    fail_init_copy = false;
    fail_wait = true;
    {
        DecodeAgreement agreement;
        assert(agreement.initialize(2));
        throws([&] { agreement.matches(comm, 2, 120, 1); });
        throws([&] { agreement.matches(comm, 2, 120, 1); });
    }
    // A timed-out device request may still reference its buffers.
    assert(streams == 1 && pinned == 1 && devices == 1);
}
