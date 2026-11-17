#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "csrc/comm/acl_timeout.hpp"
#include "csrc/comm/decode_topology.hpp"

using deep_ep::comm::DecodeTopology;
using aclrtStream = int;
using HcclComm = int;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 0, ACL_MEMCPY_HOST_TO_DEVICE = 1, ACL_MEMCPY_DEVICE_TO_HOST = 2;
constexpr int HCCL_DATA_TYPE_UINT16 = 1;

static std::vector<DecodeTopology::Record> other;
static int allocations = 0, gathers = 0, waits = 0;
static int32_t last_timeout_ms = 0;
enum Failure { none, malloc_failure, h2d_failure, collective_failure, d2h_failure, timeout_failure };
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
    return fail == timeout_failure ? 1 : 0;
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
    return fail == collective_failure ? 1 : 0;
}
namespace c10_npu {
aclrtStream getCurrentNPUStream() { return 2; }
}

#define ACL_CHECK(op) do { if (op) throw std::runtime_error("ACL"); } while (false)
#define HCCL_CHECK(op) ACL_CHECK(op)
#define EP_HOST_ASSERT(op) assert(op)

namespace deep_ep {
struct TestContext {
    int num_ranks;
    HcclComm comm;
#include "gather_listen_ports.inc"
};
}

int main() {
    const auto set_id = [](const char* id) {
#ifdef _WIN32
        assert(_putenv_s("EP_DECODE_SERVER_ID", id) == 0);
#else
        assert((*id ? setenv("EP_DECODE_SERVER_ID", id, 1) : unsetenv("EP_DECODE_SERVER_ID")) == 0);
#endif
    };
    const auto setup = [](const std::vector<int>& ids, int local_cores = 64, int peer_cores = 64, int timeout = 120) {
        DecodeTopology topology(local_cores);
        other.clear();
        std::vector<uint16_t> expected{3000};
        for (auto id : ids) {
            const int port = 3000 + static_cast<int>(expected.size());
            other.push_back({port, id, peer_cores});
            expected.push_back(static_cast<uint16_t>(port));
        }
        deep_ep::TestContext ctx{static_cast<int>(ids.size()) + 1, 1};
        int before = gathers, before_wait = waits;
        assert(ctx.gather_listen_ports(3000, timeout, &topology) == expected);
        assert(gathers == before + 1 && waits == before_wait + 1);
        assert(ctx.gather_listen_ports(3000, timeout, nullptr) == expected);
        assert(gathers == before + 2 && waits == before_wait + 2);
        return topology;
    };
    set_id("17");
    const auto interleaved = setup({9,17,9});
    assert(interleaved.mte_enabled && interleaved.server_ids == std::vector<int>({17,9,17,9}));
    assert(setup({17,17,17}).server_ids.empty());
    assert(setup({9,13,21}).server_ids.empty());
    for (const char* id : {"", "unknown", "-2", "2147483648", "7x"}) {
        set_id(id);
        const auto topology = setup({9,17,9});
        assert(topology.mte_enabled && topology.server_ids.empty());
    }
    set_id("0");
    assert(setup({0,-1,17}).server_ids.empty());
    for (int local : {0, 64, 80})
        for (int peer : {0, 64, 80}) {
            const auto topology = setup({17, 0, 17}, local, peer);
            assert(topology.mte_enabled == (local > 0 && local == peer));
            assert(topology.server_ids.empty() == !topology.mte_enabled);
        }
    setup({-1}, 64, 64, 7200);
    assert(last_timeout_ms == 7200000);
    constexpr int32_t max_ms = std::numeric_limits<int32_t>::max();
    setup({-1}, 64, 64, max_ms / 1000);
    assert(last_timeout_ms == (max_ms / 1000) * 1000);
    setup({-1}, 64, 64, max_ms / 1000 + 1);
    assert(last_timeout_ms == max_ms);
    for (Failure failure : {malloc_failure, h2d_failure, collective_failure, d2h_failure}) {
        const int before = gathers;
        fail = failure;
        try { setup({17}); assert(false); } catch (const std::runtime_error&) {}
        assert(allocations == 0 && gathers == before + (failure >= collective_failure));
        fail = none;
        assert(setup({17}).mte_enabled);
    }
    fail = timeout_failure;
    try { setup({17}); assert(false); } catch (const std::runtime_error&) {}
    // A timed-out stream might still use the receive buffer; runtime teardown reclaims it.
    assert(allocations == 1);
    aclrtFree(last_allocation);
    set_id("");
    assert(allocations == 0);
}
