#pragma once

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>
#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#ifdef DEEPEP_WITH_SHMEM
#include <host/init/shmem_host_init.h>
#include <host/mem/shmem_host_heap.h>
#include <host/data_plane/shmem_host_cc.h>
#include <host/data_plane/shmem_host_rma.h>
#endif

namespace deep_ep::comm {

class ShmemArena {
#ifdef DEEPEP_WITH_SHMEM
    static constexpr size_t kHeapBytes = size_t{32} << 30;
    static constexpr size_t kMemoryReserveBytes = size_t{1} << 30;
    static constexpr size_t kAllocationReserveBytes = size_t{2} << 20;

    struct Session {
        std::mutex mutex;
        std::string group;
        aclrtContext context = nullptr;
        size_t used = 0;
        uint64_t sequence = 0, live = 0;
        int rank = -1, world = 0;
        bool initialized = false, poisoned = false;
        std::atomic<bool> failed{false};
    };

    struct Admission {
        uint64_t bytes, sequence, live, used, world, timeout;
        uint64_t initialized, ready, poisoned;
    };

    struct Status {
        uint64_t phase, ready;
    };

    static Session& session() {
        static auto* value = new Session();
        return *value;
    }
#endif

    static void check(int status, const char* operation) {
        if (status != 0) {
#ifdef DEEPEP_WITH_SHMEM
            session().poisoned = true;
#endif
            throw std::runtime_error(std::string("SHMEM: ") + operation + " failed (" + std::to_string(status) + ")");
        }
    }

    template <typename T>
    static std::vector<T> gather(const T& value, HcclComm comm, int world, int timeout, aclrtStream stream) {
        static_assert(sizeof(T) % sizeof(uint16_t) == 0);
        const auto recv_bytes = sizeof(T) * static_cast<size_t>(world);
        std::vector<T> result(world);
        void* device = nullptr;
        check(aclrtMalloc(&device, recv_bytes + sizeof(T), ACL_MEM_MALLOC_NORMAL_ONLY), "bootstrap allocation");
        auto* send = static_cast<char*>(device) + recv_bytes;
        const auto copied = aclrtMemcpy(send, sizeof(T), &value, sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
        if (copied != 0) {
            aclrtFree(device);
            check(copied, "bootstrap host copy");
        }
        const auto launched = HcclAllGather(send, device, sizeof(T) / sizeof(uint16_t),
                                          HCCL_DATA_TYPE_UINT16, comm, stream);
        const auto waited = aclrtSynchronizeStreamWithTimeout(stream, timeout * 1000);
        if (waited != 0)
            check(waited, "bootstrap stream wait; allocation retained until process exit");
        if (launched != 0) {
            aclrtFree(device);
            check(launched, "bootstrap all-gather");
        }
        const auto received = aclrtMemcpy(result.data(), recv_bytes, device, recv_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(device);
        check(received, "bootstrap device copy");
        return result;
    }

#ifdef DEEPEP_WITH_SHMEM
    static void agree(bool ready, uint64_t phase, HcclComm comm, int world, int timeout, aclrtStream stream) {
        for (const auto& item: gather(Status{phase, ready}, comm, world, timeout, stream))
            if (item.phase != phase || !item.ready) {
                session().poisoned = true;
                throw std::runtime_error("SHMEM collective initialization/cleanup failed; restart the distributed job");
            }
    }

    void check_context() const {
        auto& state = session();
        aclrtContext current = nullptr;
        check(aclrtGetCurrentContext(&current), "current context");
        if (closed || state.poisoned || state.failed.load() || !state.initialized || current != state.context)
            throw std::runtime_error("SHMEM arena is closed, unavailable or used on a different NPU context");
    }

    HcclComm comm = nullptr;
    void* pointer = nullptr;
    std::vector<void*> peers;
    size_t bytes = 0, charge = 0;
    uint64_t id = 0;
    int world = 0, timeout = 0;
    bool closed = true;
#endif

    ShmemArena() = default;

public:
    ShmemArena(const ShmemArena&) = delete;
    ShmemArena& operator=(const ShmemArena&) = delete;

    static constexpr bool compiled() {
#ifdef DEEPEP_WITH_SHMEM
        return true;
#else
        return false;
#endif
    }

    static bool agree_capability(HcclComm comm, int world, int timeout, bool local_ready) {
        if (world <= 0 || timeout <= 0)
            throw std::invalid_argument("SHMEM capability check requires a valid group and timeout");
        local_ready = local_ready && timeout <= 3600;
        const auto capabilities = gather(static_cast<uint16_t>(local_ready), comm, world, std::min(timeout, 3600),
                                         c10_npu::getCurrentNPUStream());
        return std::all_of(capabilities.begin(), capabilities.end(), [](const auto ready) { return ready == 1; });
    }

    static std::shared_ptr<ShmemArena> try_create(const std::string& group, int rank, int world,
                                                size_t bytes, int timeout) {
#ifdef DEEPEP_WITH_SHMEM
        auto& state = session();
        std::lock_guard<std::mutex> guard(state.mutex);
        HcclComm comm = nullptr;
        check(HcclCommGetHandleWithName(group.c_str(), &comm), "HCCL group");
        uint32_t comm_rank = 0, comm_world = 0;
        check(HcclGetRankId(comm, &comm_rank), "HCCL rank");
        check(HcclGetRankSize(comm, &comm_world), "HCCL size");
        if (comm_world == 0)
            throw std::runtime_error("SHMEM bootstrap requires a nonempty HCCL group");
        auto stream = c10_npu::getCurrentNPUStream();
        aclrtContext context = nullptr;
        const auto context_status = aclrtGetCurrentContext(&context);
        const auto* soc = aclrtGetSocName();
        const auto valid_timeout = timeout > 0 && timeout <= 3600;
        const auto wait_timeout = valid_timeout ? timeout : 120;
        const auto valid_size = bytes > 0 && bytes <= kHeapBytes - kAllocationReserveBytes - 511;
        const auto charge = valid_size ? ((bytes + 511) & ~size_t{511}) + kAllocationReserveBytes : kHeapBytes;
        bool ready = context_status == 0 && context != nullptr && valid_size && valid_timeout &&
                     rank == static_cast<int>(comm_rank) && world == static_cast<int>(comm_world) &&
                     soc != nullptr && std::string(soc).rfind("Ascend950", 0) == 0 && !state.poisoned && !state.failed.load();
        if (state.initialized) {
            ready = ready && state.group == group && state.rank == rank && state.world == world &&
                    state.context == context && charge <= kHeapBytes - state.used;
        } else {
            size_t free_bytes = 0, total_bytes = 0;
            ready = ready && aclshmemx_init_status() == ACLSHMEM_STATUS_NOT_INITIALIZED &&
                    aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes) == 0 &&
                    free_bytes >= kHeapBytes + kMemoryReserveBytes;
        }
        const Admission request{bytes, state.sequence, state.live, state.used, static_cast<uint64_t>(world),
                                static_cast<uint64_t>(timeout), state.initialized, ready, state.poisoned || state.failed.load()};
        const auto admissions = gather(request, comm, comm_world, wait_timeout, stream);
        for (const auto& item: admissions)
            if (item.poisoned)
                throw std::runtime_error("SHMEM session was left incomplete; restart the distributed job");
        for (const auto& item: admissions)
            if (!item.ready || item.bytes != request.bytes || item.sequence != request.sequence ||
                item.live != request.live || item.used != request.used || item.world != request.world ||
                item.timeout != request.timeout || item.initialized != request.initialized)
                return nullptr;

        auto arena = std::shared_ptr<ShmemArena>(new ShmemArena());
        arena->peers.resize(world);
        if (!state.initialized) {
            struct Bootstrap {
                aclshmemx_uniqueid_t uid{};
                uint64_t ready = 1;
            } bootstrap;
            if (rank == 0)
                bootstrap.ready = aclshmemx_get_uniqueid(&bootstrap.uid) == 0;
            const auto ids = gather(bootstrap, comm, world, timeout, stream);
            if (!ids[0].ready)
                return nullptr;
            auto uid = ids[0].uid;
            aclshmemx_init_attr_t attr{};
            const auto attr_status = aclshmemx_set_attr_uniqueid_args(rank, world, kHeapBytes, &uid, &attr);
            const auto attrs = gather(Status{0, attr_status == 0}, comm, world, timeout, stream);
            if (std::any_of(attrs.begin(), attrs.end(), [](const auto& item) { return !item.ready; }))
                return nullptr;
            attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
            attr.option_attr.shm_init_timeout = timeout;
            attr.option_attr.shm_create_timeout = timeout;
            attr.option_attr.control_operation_timeout = timeout;
            state.group = group;
            state.context = context;
            state.rank = rank;
            state.world = world;
            state.poisoned = true;
            const auto initialized = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr);
            agree(initialized == 0, 1, comm, world, timeout, stream);
            state.initialized = true;
        }
        state.poisoned = true;
        arena->pointer = aclshmem_align(512, bytes);
        bool allocated = arena->pointer != nullptr;
        if (allocated) {
            allocated = aclrtMemset(arena->pointer, bytes, 0, bytes) == 0;
            for (int peer = 0; peer < world; ++peer) {
                arena->peers[peer] = aclshmem_ptr(arena->pointer, peer);
                allocated = allocated && arena->peers[peer] != nullptr;
            }
        }
        agree(allocated, 2, comm, world, timeout, stream);
        arena->comm = comm;
        arena->world = world;
        arena->timeout = timeout;
        arena->bytes = bytes;
        arena->charge = charge;
        arena->id = state.sequence++;
        arena->closed = false;
        state.used += charge;
        ++state.live;
        state.poisoned = false;
        return arena;
#else
        (void) group;
        (void) rank;
        (void) world;
        (void) bytes;
        (void) timeout;
        return nullptr;
#endif
    }

    void* data() const {
#ifdef DEEPEP_WITH_SHMEM
        return pointer;
#else
        return nullptr;
#endif
    }

    void check_available() const {
#ifdef DEEPEP_WITH_SHMEM
        auto& state = session();
        std::lock_guard<std::mutex> guard(state.mutex);
        check_context();
#endif
    }

    void mark_failed() noexcept {
#ifdef DEEPEP_WITH_SHMEM
        session().failed.store(true);
#endif
    }

    void* peer_ptr(int peer) const {
#ifdef DEEPEP_WITH_SHMEM
        if (closed || peer < 0 || peer >= world)
            throw std::out_of_range("SHMEM peer index or closed arena");
        return peers[peer];
#else
        (void) peer;
        return nullptr;
#endif
    }

    size_t size() const {
#ifdef DEEPEP_WITH_SHMEM
        return bytes;
#else
        return 0;
#endif
    }

    void barrier(aclrtStream stream) {
#ifdef DEEPEP_WITH_SHMEM
        auto& state = session();
        std::lock_guard<std::mutex> guard(state.mutex);
        check_context();
        state.poisoned = true;
        const auto waited = aclrtSynchronizeStreamWithTimeout(stream, timeout * 1000);
        agree(waited == 0, 3, comm, world, timeout, stream);
        aclshmem_barrier_all();
        state.poisoned = false;
#else
        (void) stream;
#endif
    }

    void close() {
#ifdef DEEPEP_WITH_SHMEM
        auto& state = session();
        std::lock_guard<std::mutex> guard(state.mutex);
        if (closed)
            return;
        check_context();
        auto stream = c10_npu::getCurrentNPUStream();
        const Admission request{id, state.sequence, state.live, state.used, static_cast<uint64_t>(world),
                                static_cast<uint64_t>(timeout), state.initialized, 1, state.poisoned};
        state.poisoned = true;
        for (const auto& item: gather(request, comm, world, timeout, stream))
            if (std::memcmp(&item, &request, sizeof(request)) != 0)
                throw std::runtime_error("SHMEM arenas must be closed in the same order on every rank");
        const auto waited = aclrtSynchronizeDevice();
        agree(waited == 0, 4, comm, world, timeout, stream);
        aclshmem_barrier_all();
        aclshmem_free(pointer);
        pointer = nullptr;
        closed = true;
        --state.live;
        state.used -= charge;
        if (state.live == 0) {
            const auto finalized = aclshmem_finalize();
            agree(finalized == 0, 5, comm, world, timeout, stream);
            state.initialized = false;
            state.group.clear();
            state.context = nullptr;
            state.rank = -1;
            state.world = 0;
        }
        state.poisoned = false;
#endif
    }

    ~ShmemArena() {
#ifdef DEEPEP_WITH_SHMEM
        if (!closed) {
            auto& state = session();
            std::lock_guard<std::mutex> guard(state.mutex);
            state.poisoned = true;
        }
#endif
    }
};

}
