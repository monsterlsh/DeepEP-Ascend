#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <acl/acl.h>
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
    static constexpr int kMaxTimeoutSecs = 3600;

#ifdef DEEPEP_WITH_SHMEM
    static constexpr size_t kHeapBytes = size_t{32} << 30;
    static constexpr size_t kMemoryReserveBytes = size_t{1} << 30;
    static constexpr size_t kAllocationReserveBytes = size_t{2} << 20;
    static constexpr size_t kAlignmentBytes = 512;

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

        bool operator==(const Admission&) const = default;
    };

    enum class Phase: uint64_t { Attributes, Initialize, Allocate, Close, Finalize };

    struct Status {
        Phase phase;
        uint64_t ready;
    };

    static Session& session() {
        static auto* value = new Session();
        return *value;
    }
#endif

    static void check(int status, const char* operation) {
        if (status != 0) {
#ifdef DEEPEP_WITH_SHMEM
            session().failed.store(true);
#endif
            throw std::runtime_error(std::string("SHMEM: ") + operation + " failed (" + std::to_string(status) + ")");
        }
    }

    template <typename T>
    static std::vector<T> gather(const T& value, HcclComm comm, int world, int timeout, aclrtStream stream) {
        static_assert(std::is_trivially_copyable_v<T> and sizeof(T) % sizeof(uint16_t) == 0);
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
    static void agree(bool ready, Phase phase, HcclComm comm, int world, int timeout, aclrtStream stream) {
        for (const auto& item: gather(Status{phase, ready}, comm, world, timeout, stream)) {
            if (item.phase != phase or not item.ready) {
                session().poisoned = true;
                throw std::runtime_error("SHMEM collective initialization/cleanup failed; restart the distributed job");
            }
        }
    }

    static bool initialize_session(const std::string& group, int rank, int world, int timeout,
                                   aclrtContext context, HcclComm comm, aclrtStream stream) {
        auto& state = session();
        struct Bootstrap {
            aclshmemx_uniqueid_t uid{};
            uint64_t ready = 1;
        } bootstrap;
        if (rank == 0)
            bootstrap.ready = aclshmemx_get_uniqueid(&bootstrap.uid) == 0;
        const auto ids = gather(bootstrap, comm, world, timeout, stream);
        if (not ids[0].ready)
            return false;
        auto uid = ids[0].uid;
        aclshmemx_init_attr_t attr{};
        const auto attr_status = aclshmemx_set_attr_uniqueid_args(rank, world, kHeapBytes, &uid, &attr);
        const auto attrs = gather(Status{Phase::Attributes, attr_status == 0}, comm, world, timeout, stream);
        if (std::any_of(attrs.begin(), attrs.end(), [](const auto& item) { return not item.ready; }))
            return false;
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
        agree(initialized == 0, Phase::Initialize, comm, world, timeout, stream);
        state.initialized = true;
        return true;
    }

    void check_context() const {
        auto& state = session();
        aclrtContext current = nullptr;
        check(aclrtGetCurrentContext(&current), "current context");
        if (closed or state.poisoned or state.failed.load() or not state.initialized or current != state.context)
            throw std::runtime_error("SHMEM arena is closed, unavailable or used on a different NPU context");
    }

    HcclComm comm = nullptr;
    void* pointer = nullptr;
    std::vector<void*> peers;
    size_t charge = 0;
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

    static uint16_t agree_capabilities(HcclComm comm, int world, int timeout, uint16_t local) {
        if (world <= 0 or timeout <= 0)
            throw std::invalid_argument("SHMEM capability check requires a valid group and timeout");
        if (timeout > kMaxTimeoutSecs)
            local = 0;
        const auto capabilities = gather(local, comm, world, std::min(timeout, kMaxTimeoutSecs),
                                         c10_npu::getCurrentNPUStream());
        for (const auto peer: capabilities)
            local &= peer;
        return local;
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
        const auto stream = c10_npu::getCurrentNPUStream();
        aclrtContext context = nullptr;
        const auto context_status = aclrtGetCurrentContext(&context);
        const auto* soc = aclrtGetSocName();
        const auto valid_timeout = timeout > 0 and timeout <= kMaxTimeoutSecs;
        const auto wait_timeout = valid_timeout ? timeout : 120;
        const auto valid_size = bytes > 0 and bytes <= kHeapBytes - kAllocationReserveBytes - (kAlignmentBytes - 1);
        const auto charge = valid_size ? (bytes + kAlignmentBytes - 1) / kAlignmentBytes * kAlignmentBytes + kAllocationReserveBytes
                                       : kHeapBytes;
        bool ready = context_status == 0 and context != nullptr and valid_size and valid_timeout and
                     rank == static_cast<int>(comm_rank) and world == static_cast<int>(comm_world) and
                     soc != nullptr and std::string_view(soc).starts_with("Ascend950") and not state.poisoned and not state.failed.load();
        if (state.initialized) {
            ready = ready and state.group == group and state.rank == rank and state.world == world and
                    state.context == context and charge <= kHeapBytes - state.used;
        } else {
            size_t free_bytes = 0, total_bytes = 0;
            ready = ready and aclshmemx_init_status() == ACLSHMEM_STATUS_NOT_INITIALIZED and
                    aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes) == 0 and
                    free_bytes >= kHeapBytes + kMemoryReserveBytes;
        }
        const Admission request{bytes, state.sequence, state.live, state.used, static_cast<uint64_t>(world),
                                static_cast<uint64_t>(timeout), state.initialized, ready, state.poisoned or state.failed.load()};
        const auto admissions = gather(request, comm, comm_world, wait_timeout, stream);
        for (const auto& item: admissions) {
            if (item.poisoned)
                throw std::runtime_error("SHMEM session was left incomplete; restart the distributed job");
        }
        for (const auto& item: admissions) {
            if (not item.ready or item != request)
                return nullptr;
        }

        auto arena = std::shared_ptr<ShmemArena>(new ShmemArena());
        arena->peers.resize(world);
        if (not state.initialized and not initialize_session(group, rank, world, timeout, context, comm, stream))
            return nullptr;

        state.poisoned = true;
        arena->pointer = aclshmem_align(kAlignmentBytes, bytes);
        bool allocated = arena->pointer != nullptr;
        if (allocated) {
            allocated = aclrtMemset(arena->pointer, bytes, 0, bytes) == 0;
            for (int peer = 0; peer < world; ++ peer) {
                arena->peers[peer] = aclshmem_ptr(arena->pointer, peer);
                allocated = allocated and arena->peers[peer] != nullptr;
            }
        }
        agree(allocated, Phase::Allocate, comm, world, timeout, stream);
        arena->comm = comm;
        arena->world = world;
        arena->timeout = timeout;
        arena->charge = charge;
        arena->id = state.sequence++;
        arena->closed = false;
        state.used += charge;
        ++ state.live;
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
        if (closed or peer < 0 or peer >= world)
            throw std::out_of_range("SHMEM peer index or closed arena");
        return peers[peer];
#else
        (void) peer;
        return nullptr;
#endif
    }

    void close() {
#ifdef DEEPEP_WITH_SHMEM
        auto& state = session();
        std::lock_guard<std::mutex> guard(state.mutex);
        if (closed)
            return;
        check_context();
        const auto stream = c10_npu::getCurrentNPUStream();
        const Admission request{id, state.sequence, state.live, state.used, static_cast<uint64_t>(world),
                                static_cast<uint64_t>(timeout), state.initialized, 1, state.poisoned};
        state.poisoned = true;
        for (const auto& item: gather(request, comm, world, timeout, stream)) {
            if (item != request)
                throw std::runtime_error("SHMEM arenas must be closed in the same order on every rank");
        }
        const auto waited = aclrtSynchronizeDevice();
        agree(waited == 0, Phase::Close, comm, world, timeout, stream);
        aclshmem_barrier_all();
        aclshmem_free(pointer);
        pointer = nullptr;
        closed = true;
        -- state.live;
        state.used -= charge;
        if (state.live == 0) {
            const auto finalized = aclshmem_finalize();
            agree(finalized == 0, Phase::Finalize, comm, world, timeout, stream);
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
        if (not closed) {
            auto& state = session();
            std::lock_guard<std::mutex> guard(state.mutex);
            state.poisoned = true;
        }
#endif
    }
};

}  // namespace deep_ep::comm
