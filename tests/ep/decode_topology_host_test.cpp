#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "csrc/comm/decode_topology.hpp"

using deep_ep::comm::DecodeTopology;
using aclrtStream = int;
using HcclComm = int;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 0, ACL_MEMCPY_HOST_TO_DEVICE = 1, ACL_MEMCPY_DEVICE_TO_HOST = 2;
constexpr int HCCL_DATA_TYPE_UINT16 = 1;
constexpr int ACL_SUCCESS = 0, HCCL_SUCCESS = 0;

static std::vector<DecodeTopology::Record> other;
static int allocations = 0, gathers = 0, waits = 0;
static int32_t last_timeout_ms = 0;
enum Failure { none, malloc_failure, h2d_failure, collective_failure, d2h_failure, timeout_failure, collective_and_timeout_failure };
static Failure fail = none;
static void* last_allocation = nullptr;

int aclrtMalloc(void** pointer, size_t bytes, int) {
    if (fail == malloc_failure) return 1;
    *pointer = std::malloc(bytes);
    last_allocation = *pointer;
    if (*pointer) ++allocations;
    return *pointer ? 0 : 1;
}
int aclrtFree(void* pointer) {
    --allocations;
    std::free(pointer);
    return 0;
}
int aclrtMemcpy(void* dst, size_t capacity, const void* src, size_t bytes, int kind) {
    if ((fail == h2d_failure && kind == ACL_MEMCPY_HOST_TO_DEVICE)
        || (fail == d2h_failure && kind == ACL_MEMCPY_DEVICE_TO_HOST)) return 1;
    assert(bytes <= capacity);
    std::memcpy(dst, src, bytes);
    return 0;
}
int aclrtSynchronizeStreamWithTimeout(aclrtStream, int32_t timeout_ms) {
    ++waits;
    last_timeout_ms = timeout_ms;
    return fail == timeout_failure || fail == collective_and_timeout_failure ? 1 : 0;
}
int HcclAllGather(void* send, void* receive, uint64_t count, int dtype, HcclComm, aclrtStream) {
    ++gathers;
    assert(dtype == HCCL_DATA_TYPE_UINT16);
    const size_t bytes = count * sizeof(uint16_t);
    std::memcpy(receive, send, bytes);
    if (count == 1) {
        auto ports = static_cast<uint16_t*>(receive);
        for (size_t i = 0; i < other.size(); ++i)
            ports[i + 1] = static_cast<uint16_t>(other[i].port);
    } else {
        assert(bytes == sizeof(DecodeTopology::Record));
        std::memcpy(static_cast<char*>(receive) + bytes, other.data(), bytes * other.size());
    }
    return fail == collective_failure || fail == collective_and_timeout_failure ? 1 : 0;
}
namespace c10_npu {
aclrtStream getCurrentNPUStream() { return 2; }
}

#define ACL_CHECK(op) do { if (op) throw std::runtime_error("ACL"); } while (false)
#define HCCL_CHECK(op) do { if (op) throw std::runtime_error("HCCL"); } while (false)
#define EP_HOST_ASSERT(op) assert(op)

namespace deep_ep {
struct TestContext {
    int num_ranks;
    HcclComm comm;
#include "gather_listen_ports.inc"
};

struct TestMemory {
    void close() { assert(!closed); closed = true; }
    bool closed = false;
};

struct TestHCCLContext {
    std::shared_ptr<TestMemory> memory;
    bool close_fails = false;
    void close() {
        if (close_fails)
            throw std::runtime_error("close");
        memory.reset();
    }
};

struct SharedContext {
    bool finalized = false;
    std::shared_ptr<TestMemory> symmetric_memory;
    std::shared_ptr<TestHCCLContext> hccl_context;
#include "finalize.inc"
};
}

int main() {
    {
        deep_ep::SharedContext first, second;
        first.symmetric_memory = second.symmetric_memory = std::make_shared<deep_ep::TestMemory>();
        first.hccl_context = std::make_shared<deep_ep::TestHCCLContext>(first.symmetric_memory);
        second.hccl_context = std::make_shared<deep_ep::TestHCCLContext>(second.symmetric_memory);
        auto retained_registration = first.hccl_context;
        std::weak_ptr<deep_ep::TestMemory> window = first.symmetric_memory;
        first.finalize_ep();
        second.finalize_ep();
        assert(first.finalized && second.finalized);
        assert(retained_registration->memory && !retained_registration->memory->closed);
        retained_registration.reset();
        assert(window.expired());
    }
    for (bool ep : {false, true}) {
        deep_ep::SharedContext context;
        context.symmetric_memory = std::make_shared<deep_ep::TestMemory>();
        context.hccl_context = std::make_shared<deep_ep::TestHCCLContext>(context.symmetric_memory, true);
        if (ep) {
            try { context.finalize_ep(); assert(false); }
            catch (const std::runtime_error& e) { assert(std::strcmp(e.what(), "close") == 0); }
            assert(context.finalized && context.hccl_context && context.symmetric_memory);
        } else {
            context.finalize();
            assert(context.finalized && !context.hccl_context && !context.symmetric_memory);
        }
    }
    using Location = DecodeTopology::ServerLocation;
    enum LocType { ENDPOINT_LOC_TYPE_DEVICE, ENDPOINT_LOC_TYPE_HOST };
    struct EndpointLoc {
        LocType locType;
        struct { uint32_t serverIdx, superPodIdx; } device;
    };
    struct OlderEndpointLoc {
        LocType locType;
        struct { uint32_t devPhyId; } device;
    };
    const Location local{17, 0}, remote{17, 1};
    assert(DecodeTopology::get_server_location(EndpointLoc{ENDPOINT_LOC_TYPE_DEVICE, {17, 1}}) == remote);
    assert(DecodeTopology::get_server_location(EndpointLoc{ENDPOINT_LOC_TYPE_HOST, {17, 1}}) == Location{});
    assert(DecodeTopology::get_server_location(OlderEndpointLoc{ENDPOINT_LOC_TYPE_DEVICE, {3}}) == Location{});

    const auto setup = [](const std::vector<Location>& locations, int local_cores = 64, int peer_cores = 64,
                          int timeout = 120, Location local_location = {17, 0}) {
        DecodeTopology topology(local_cores);
        topology.local_server = local_location;
        other.clear();
        std::vector<uint16_t> expected{3000};
        for (auto location : locations) {
            const int port = 3000 + static_cast<int>(expected.size());
            other.push_back({port, peer_cores, location});
            expected.push_back(static_cast<uint16_t>(port));
        }
        deep_ep::TestContext ctx{static_cast<int>(locations.size()) + 1, 1};
        int before = gathers, before_wait = waits;
        assert(ctx.gather_listen_ports(3000, timeout, &topology) == expected);
        assert(gathers == before + 1 && waits == before_wait + 1);
        assert(ctx.gather_listen_ports(3000, timeout, nullptr) == expected);
        assert(gathers == before + 2 && waits == before_wait + 2);
        return topology;
    };
    // Servers can reuse a local index across different superpods.
    const auto interleaved = setup({remote, local, remote});
    assert(interleaved.mte_enabled && interleaved.server_ids == std::vector<int>({0, 1, 0, 1}));
    assert(setup({{18, 0}, local, {18, 0}}).server_ids == interleaved.server_ids);
    assert(setup({local, local, local}).server_ids.empty());
    assert(setup({{9, 0}, {13, 0}, {21, 0}}).server_ids.empty());
    for (Location unknown : {Location{}, Location{UINT32_MAX, 0}, Location{17, UINT32_MAX}}) {
        const auto missing_peer = setup({local, unknown, remote});
        const auto missing_local = setup({local, local, remote}, 64, 64, 120, unknown);
        assert(missing_peer.mte_enabled && missing_peer.server_ids.empty());
        assert(missing_local.mte_enabled && missing_local.server_ids.empty());
    }
    for (int cores : {0, 64, 80})
        for (int peer : {0, 64, 80}) {
            const auto topology = setup({remote, local, remote}, cores, peer);
            assert(topology.mte_enabled == (cores > 0 && cores == peer));
            assert(topology.server_ids.empty() == !topology.mte_enabled);
        }
    setup({{}}, 64, 64, 7200);
    assert(last_timeout_ms == 7200000);
    constexpr int32_t max_ms = std::numeric_limits<int32_t>::max();
    setup({{}}, 64, 64, max_ms / 1000);
    assert(last_timeout_ms == (max_ms / 1000) * 1000);
    setup({{}}, 64, 64, max_ms / 1000 + 1);
    assert(last_timeout_ms == max_ms);
    for (Failure failure : {malloc_failure, h2d_failure, collective_failure, d2h_failure}) {
        const int before = gathers;
        fail = failure;
        try { setup({local}); assert(false); } catch (const std::runtime_error&) {}
        assert(allocations == 0 && gathers == before + (failure >= collective_failure));
        fail = none;
        assert(setup({local}).mte_enabled);
    }
    for (Failure failure : {timeout_failure, collective_and_timeout_failure}) {
        fail = failure;
        try { setup({local}); assert(false); }
        catch (const std::runtime_error& e) {
            assert(std::strcmp(e.what(), failure == timeout_failure ? "ACL" : "HCCL") == 0);
        }
        // A timed-out stream might still use the receive buffer; runtime teardown reclaims it.
        assert(allocations == 1);
        aclrtFree(last_allocation);
        assert(allocations == 0);
    }
}
