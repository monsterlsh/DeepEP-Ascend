#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using aclrtContext = void*;
using aclrtStream = void*;
using HcclComm = void*;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 2;
constexpr int ACL_HBM_MEM = 0;
constexpr int HCCL_DATA_TYPE_UINT16 = 0;
constexpr int ACLSHMEM_STATUS_NOT_INITIALIZED = 0;
constexpr int ACLSHMEM_DATA_OP_MTE = 1;
constexpr int ACLSHMEMX_INIT_WITH_UNIQUEID = 2;

struct aclshmemx_uniqueid_t {
    char bytes[128]{};
};
struct aclshmemx_init_attr_t {
    struct {
        int data_op_engine_type = 0;
        int shm_init_timeout = 0, shm_create_timeout = 0, control_operation_timeout = 0;
    } option_attr;
};

static int sdk_status = 0, init_count = 0, free_count = 0, finalize_count = 0;
static bool reject_peer = false, mismatch_size = false, fail_init = false, fail_peer = false;
static bool fail_uid = false, fail_attr = false, low_memory = false, fail_collective = false;
static bool fail_finalize = false, fail_wait = false, mismatch_close = false;
static bool remote_init_failure = false;
static uint16_t remote_capabilities = UINT16_MAX;
static int allocation_count = 0, barrier_count = 0;
static size_t configured_heap = 0;
static std::vector<void*> allocations;

// DecodeAgreement uses a dedicated stream and pinned host staging for a rank-wide vote.
constexpr int ACL_SUCCESS = 0;
using aclmdlRI = void*;
enum aclmdlRICaptureStatus { ACL_MODEL_RI_CAPTURE_STATUS_NONE, ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE };
static bool capturing = false, fail_pinned_alloc = false, fail_device_alloc = false;
static int streams = 0, pinned = 0, devices = 0;
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

int aclrtGetCurrentContext(aclrtContext* context) {
    *context = reinterpret_cast<void*>(1);
    return 0;
}

const char* aclrtGetSocName() {
    return "Ascend950";
}

int aclrtGetMemInfo(int, size_t* free, size_t* total) {
    *free = low_memory ? 1 : size_t{64} << 30;
    *total = size_t{64} << 30;
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

int aclrtMemcpy(void* dst, size_t capacity, const void* src, size_t bytes, int) {
    assert(bytes <= capacity);
    std::memcpy(dst, src, bytes);
    return 0;
}

int aclrtMemcpyAsync(void* dst, size_t capacity, const void* src, size_t bytes, int kind, aclrtStream stream) {
    assert(stream == reinterpret_cast<void*>(4));
    return aclrtMemcpy(dst, capacity, src, bytes, kind);
}

int aclrtMemset(void*, size_t, int, size_t) {
    return 0;
}

int aclrtSynchronizeStreamWithTimeout(aclrtStream, int) {
    return fail_wait ? 1 : 0;
}

int aclrtSynchronizeDevice() {
    return 0;
}

int HcclCommGetHandleWithName(const char*, HcclComm* comm) {
    *comm = reinterpret_cast<void*>(2);
    return 0;
}

int HcclGetRankId(HcclComm, uint32_t* rank) {
    *rank = 0;
    return 0;
}

int HcclGetRankSize(HcclComm, uint32_t* world) {
    *world = 2;
    return 0;
}

int HcclAllGather(void* send, void* receive, uint64_t count, int, HcclComm, aclrtStream) {
    const auto bytes = count * sizeof(uint16_t);
    std::memcpy(receive, send, bytes);
    std::memcpy(static_cast<char*>(receive) + bytes, send, bytes);
    if (bytes == sizeof(uint16_t))
        static_cast<uint16_t*>(receive)[1] &= remote_capabilities;
    if (bytes == 9 * sizeof(uint64_t)) {
        auto* other = reinterpret_cast<uint64_t*>(static_cast<char*>(receive) + bytes);
        if (reject_peer)
            other[7] = 0;
        if (mismatch_size or mismatch_close)
            ++ other[0];
    }
    if (bytes == 2 * sizeof(uint64_t) and remote_init_failure) {
        auto* other = reinterpret_cast<uint64_t*>(static_cast<char*>(receive) + bytes);
        if (other[0] == 1)
            other[1] = 0;
    }
    return fail_collective ? 1 : 0;
}

namespace c10_npu {

aclrtStream getCurrentNPUStream() {
    return reinterpret_cast<void*>(3);
}

}  // namespace c10_npu

int aclshmemx_init_status() {
    return sdk_status;
}

int aclshmemx_get_uniqueid(aclshmemx_uniqueid_t* uid) {
    uid->bytes[0] = 42;
    return fail_uid ? 1 : 0;
}

int aclshmemx_set_attr_uniqueid_args(int rank, int world, int64_t heap, aclshmemx_uniqueid_t* uid, aclshmemx_init_attr_t*) {
    assert(rank == 0 and world == 2 and uid->bytes[0] == 42);
    configured_heap = heap;
    return fail_attr ? 1 : 0;
}

int aclshmemx_init_attr(int mode, aclshmemx_init_attr_t* attr) {
    assert(mode == ACLSHMEMX_INIT_WITH_UNIQUEID);
    assert(attr->option_attr.data_op_engine_type == ACLSHMEM_DATA_OP_MTE);
    assert(attr->option_attr.shm_init_timeout == 120);
    ++ init_count;
    sdk_status = 1;
    return fail_init ? 1 : 0;
}

void* aclshmem_align(size_t alignment, size_t) {
    assert(alignment == 512);
    ++ allocation_count;
    auto* pointer = std::malloc(1024);
    allocations.push_back(pointer);
    return pointer;
}

void* aclshmem_ptr(void* pointer, int peer) {
    return fail_peer and peer == 1 ? nullptr : static_cast<char*>(pointer) + peer * 64;
}

void aclshmem_free(void* pointer) {
    ++ free_count;
    std::free(pointer);
}

void aclshmem_barrier_all() {
    ++ barrier_count;
}

int aclshmem_finalize() {
    ++ finalize_count;
    sdk_status = 0;
    return fail_finalize ? 1 : 0;
}

#include "../../csrc/comm/shmem.hpp"
#include "../../csrc/comm/decode_agreement.hpp"

using deep_ep::comm::ShmemArena;

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

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string name = argv[1];
    const auto create = [] { return ShmemArena::try_create("group", 0, 2, 4096, 120); };
    if (name == "decode_vote") {
        using deep_ep::comm::DecodeAgreement;
        assert(!DecodeAgreement::is_capturing());
        capturing = true;
        assert(DecodeAgreement::is_capturing());
        for (int failure : {0, 1, 2}) {
            {
                DecodeAgreement agreement;
                fail_pinned_alloc = failure == 1;
                fail_device_alloc = failure == 2;
                assert(agreement.initialize(2) == (failure == 0));
                if (!failure) {
                    auto comm = reinterpret_cast<HcclComm>(2);
                    for (uint16_t proposal : {0, 1, 2, 3, 4, 8}) {
                        remote_capabilities = UINT16_MAX;
                        assert(agreement.matches(comm, 2, 120, proposal) == (proposal != 0));
                        remote_capabilities = 0;
                        assert(!agreement.matches(comm, 2, 120, proposal));
                    }
                }
            }
            assert(streams == 0 && pinned == 0 && devices == 0);
        }
        return 0;
    }
    if (name == "decode_vote_failure") {
        {
            deep_ep::comm::DecodeAgreement agreement;
            assert(agreement.initialize(2));
            fail_wait = true;
            throws([&] { agreement.matches(reinterpret_cast<HcclComm>(2), 2, 120, 1); });
        }
        // An unfinished device request may still reference its buffers.
        assert(streams == 1 && pinned == 1 && devices == 1);
        return 0;
    }
    if (name == "capability" or name == "mixed_capability" or name == "capability_error") {
        auto comm = reinterpret_cast<HcclComm>(2);
        if (name == "capability_error") {
            fail_collective = true;
            throws([&] { ShmemArena::agree_capabilities(comm, 2, 120, 3); });
        } else {
            remote_capabilities = name == "mixed_capability" ? 1 : 3;
            assert(ShmemArena::agree_capabilities(comm, 2, 120, 3) == remote_capabilities);
            assert(ShmemArena::agree_capabilities(comm, 2, 120, 2) == (2 & remote_capabilities));
            assert(ShmemArena::agree_capabilities(comm, 2, 120, 0) == 0);
            assert(ShmemArena::agree_capabilities(comm, 2, 7200, 3) == 0);
        }
        assert(init_count == 0 and allocation_count == 0 and finalize_count == 0);
        std::cout << name << " passed\n";
        return 0;
    }
#ifndef DEEPEP_WITH_SHMEM
    assert(not ShmemArena::compiled() and not create());
    assert(init_count == 0);
#else
    assert(ShmemArena::compiled());
    if (name == "lifecycle") {
        auto first = create();
        auto second = create();
        assert(first and second and init_count == 1 and allocation_count == 2);
        assert(configured_heap == size_t{32} << 30);
        assert(first->peer_ptr(1) == static_cast<char*>(first->data()) + 64);
        first->close();
        assert(finalize_count == 0 and free_count == 1);
        second->close();
        assert(finalize_count == 1 and free_count == 2 and barrier_count == 2);
        auto third = create();
        assert(third and init_count == 2);
        third->close();
    } else if (name == "shared_failure") {
        auto first = create();
        auto second = create();
        first->mark_failed();
        throws([&] { first->check_available(); });
        throws([&] { second->check_available(); });
        throws([&] { second->close(); });
        throws(create);
        assert(init_count == 1 and allocation_count == 2);
        assert(barrier_count == 0 and free_count == 0 and finalize_count == 0);
    } else if (name == "foreign") {
        sdk_status = 1;
        assert(not create());
        assert(init_count == 0 and finalize_count == 0 and sdk_status == 1);
    } else if (name == "group") {
        auto arena = create();
        assert(not ShmemArena::try_create("another", 0, 2, 4096, 120));
        assert(finalize_count == 0);
        arena->close();
    } else if (name == "memory") {
        low_memory = true;
        assert(not create() and init_count == 0);
    } else if (name == "capacity") {
        auto first = ShmemArena::try_create("group", 0, 2, size_t{20} << 30, 120);
        assert(first);
        assert(not ShmemArena::try_create("group", 0, 2, size_t{20} << 30, 120));
        first->close();
    } else if (name == "peer_reject" or name == "size_mismatch") {
        reject_peer = name == "peer_reject";
        mismatch_size = name == "size_mismatch";
        assert(not create() and init_count == 0);
    } else if (name == "uid" or name == "attributes") {
        fail_uid = name == "uid";
        fail_attr = name == "attributes";
        assert(not create() and init_count == 0);
    } else if (name == "init_failure" or name == "mapping_failure" or name == "remote_init_failure") {
        fail_init = name == "init_failure";
        fail_peer = name == "mapping_failure";
        remote_init_failure = name == "remote_init_failure";
        throws(create);
        throws(create);
        assert(init_count == 1 and finalize_count == 0);
    } else if (name == "abandon") {
        auto arena = create();
        arena.reset();
        assert(finalize_count == 0 and free_count == 0);
        throws(create);
    } else if (name == "collective" or name == "wait") {
        fail_collective = name == "collective";
        fail_wait = name == "wait";
        throws(create);
        assert(init_count == 0);
        fail_collective = false;
        fail_wait = false;
        throws(create);
    } else if (name == "close_order") {
        auto arena = create();
        mismatch_close = true;
        throws([&] { arena->close(); });
        assert(free_count == 0 and finalize_count == 0);
    } else if (name == "finalize") {
        auto arena = create();
        fail_finalize = true;
        throws([&] { arena->close(); });
        throws(create);
        assert(finalize_count == 1);
    } else {
        throw std::runtime_error("unknown test case");
    }
#endif
    std::cout << name << " passed\n";
}
