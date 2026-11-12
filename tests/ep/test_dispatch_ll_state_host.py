"""Compile and execute the actual LL State class with host dependency stubs.

This exercises ownership/error/event state transitions, not NPU behavior. Device
submission failures, Torch allocation, SHMEM and events are mocked; no runtime is imported.
"""

import re
import unittest

from _host_test_helpers import ROOT, run_host_cpp

HEADER = ROOT / 'csrc/buffers/low_latency_state.hpp'


def host_test_source():
    prefix = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "csrc/kernels/ep/dispatch_low_latency_policy.hpp"

namespace mock {
std::vector<std::string> events;
int stream = 0, copies = 0, allocation_count = 0;
const char* soc = "Ascend950DT_9582";
int arch = 3510, cores = 64, vec_per_ai = 2, ub_bytes = 216 * 1024;
int fail_allocation = -1, fail_copy = -1;
bool event_failure = false, wait_failure = false;
void note(const std::string& value) { events.push_back(value); }
void reset() {
    soc = "Ascend950DT_9582";
    events.clear(); stream = 0; copies = allocation_count = 0;
    fail_allocation = fail_copy = -1;
    event_failure = wait_failure = false;
    arch = 3510; cores = 64; vec_per_ai = 2; ub_bytes = 216 * 1024;
}
}
const char* aclrtGetSocName() { return mock::soc; }
#define EP_HOST_ASSERT(cond) do { if (!(cond)) throw std::runtime_error(#cond); } while (0)
#define ACL_CHECK(expr) do { if ((expr) != 0) throw std::runtime_error("ACL copy"); } while (0)
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 0;
int aclrtMemcpy(void* dst, size_t cap, const void* src, size_t bytes, int) {
    const int call = ++mock::copies; mock::note("copy");
    if (call == mock::fail_copy) return 1;
    assert(bytes <= cap); std::memcpy(dst, src, bytes); return 0;
}
namespace torch {
enum Type { kByte, kInt, kFloat, kLong, kBFloat16 };
struct TensorOptions {
    Type type = kBFloat16;
    TensorOptions dtype(Type value) const { return {value}; }
};
struct Storage { std::vector<uint8_t> bytes; };
struct Tensor {
    std::shared_ptr<Storage> storage;
    std::vector<int64_t> dimensions;
    TensorOptions opts;
    TensorOptions options() const { return opts; }
    int64_t size(int dim) const { return dimensions.at(dim); }
    void* data_ptr() const {
        if (!storage) throw std::runtime_error("undefined tensor");
        return storage->bytes.data();
    }
};
Tensor empty(std::initializer_list<int64_t> shape, TensorOptions options) {
    const int call = ++mock::allocation_count; mock::note("allocation");
    if (call == mock::fail_allocation) throw std::runtime_error("allocation failure");
    int64_t size = options.type == kByte ? 1 : options.type == kBFloat16 ? 2 : options.type == kLong ? 8 : 4;
    for (auto dimension : shape) { assert(dimension > 0); size *= dimension; }
    auto storage = std::make_shared<Storage>(); storage->bytes.resize(size);
    return {storage, shape, options};
}
}
namespace deep_ep {
struct FakeRuntime {
    int get_npu_arch() const { return mock::arch; }
    int get_num_vec_cores() const { return mock::cores; }
    int get_num_vec_cores_per_ai_core() const { return mock::vec_per_ai; }
    int get_num_ubuf_bytes_per_vec_core() const { return mock::ub_bytes; }
} runtime_value;
auto runtime = &runtime_value;
struct EventHandle {
    int stream;
    EventHandle(): stream(mock::stream) {
        mock::note("record:" + std::to_string(stream));
        if (mock::event_failure) throw std::runtime_error("event record failure");
    }
    void current_stream_wait() const {
        mock::note("wait:" + std::to_string(stream) + "->" + std::to_string(mock::stream));
        if (mock::wait_failure) throw std::runtime_error("event wait failure");
    }
};
namespace comm {
constexpr int64_t kNumSystemCyclesPerSecond = 1000000000LL;
struct HcclContext { void* comm = nullptr; };
struct Context {
    int rank_idx = 3, num_ranks = 8, num_cpu_timeout_secs = 300;
    int64_t num_gpu_timeout_cycles = 100 * kNumSystemCyclesPerSecond;
    std::shared_ptr<HcclContext> hccl_context = std::make_shared<HcclContext>();
};
class DecodeAgreement {
public:
    inline static bool all_match = true, capturing = false;
    inline static int votes = 0;
    bool initialize(int) { return true; }
    static bool is_capturing() { return capturing; }
    bool matches(void*, int, int, uint16_t proposal) {
        ++votes;
        return all_match && proposal != 0;
    }
};
class ShmemArena {
public:
    inline static bool supported = true;
    inline static uint16_t peer_capabilities = 1;
    inline static bool session_failed = false;
    inline static int failure_marks = 0;
    inline static int result = 1, attempts = 0, closes = 0, agreements = 0, peer_reads = 0;
    inline static int bad_peer = -1, peer_misalignment = 0;
    inline static int64_t requested_bytes = 0;
    alignas(32) uint8_t memory[512]{};
    bool closed = false;
    int rank = 0;
    static bool compiled() { return supported; }
    static uint16_t agree_capabilities(void*, int, int, uint16_t local) { ++agreements; return local & peer_capabilities; }
    static std::shared_ptr<ShmemArena> try_create(const std::string& group, int rank, int world, int64_t bytes, int timeout) {
        assert(group == "group" && world == 8 && timeout == 300);
        requested_bytes = bytes; ++attempts; mock::note("arena");
        if (session_failed) throw std::runtime_error("shared session failed");
        if (result < 0) throw std::runtime_error("arena error");
        if (result == 0) return nullptr;
        auto arena = std::make_shared<ShmemArena>(); arena->rank = rank; return arena;
    }
    void* peer_ptr(int peer) {
        ++peer_reads;
        return peer == bad_peer ? nullptr : memory + peer * 32 + peer_misalignment;
    }
    void* data() { return memory + rank * 32; }
    void check_available() {
        if (session_failed) throw std::runtime_error("shared session failed");
        if (closed) throw std::runtime_error("closed arena");
    }
    void mark_failed() noexcept { session_failed = true; ++failure_marks; }
    void close() { check_available(); ++closes; closed = true; mock::note("close"); }
    static void reset() {
        DecodeAgreement::all_match = true; DecodeAgreement::capturing = false; DecodeAgreement::votes = 0;
        supported = true; peer_capabilities = 1; session_failed = false;
        result = 1; attempts = closes = failure_marks = agreements = peer_reads = 0;
        bad_peer = -1; peer_misalignment = 0; requested_bytes = 0;
    }
};
}
}
'''
    production = re.sub(r'^#(?:include|pragma).*\n', '', HEADER.read_text(), flags=re.M)
    harness = r'''
using deep_ep::EPLowLatencyState;
using deep_ep::comm::ShmemArena;
constexpr int64_t buffer_bytes = 1LL << 30;
int cases = 0;
void reset() { mock::reset(); ShmemArena::reset(); }
template<class F> void rejects(F fn, const std::string& message) {
    bool caught = false;
    try { fn(); } catch (const std::runtime_error& error) {
        caught = true; assert(std::string(error.what()).find(message) != std::string::npos);
    }
    assert(caught);
}
auto input() { return torch::empty({1,7168}, {}); }
auto prepare(EPLowLatencyState& state, const deep_ep::comm::Context& context,
             const torch::Tensor& x, bool fp8 = false) {
    return state.prepare_dispatch(context, x, 2, 7168, 8, 128, 64, 216*1024, fp8, false, false, true);
}
auto combine(EPLowLatencyState& state, const deep_ep::comm::Context& ctx, const torch::Tensor& x, bool eligible) {
    bool native_mte = false;
    auto launch = state.prepare_combine(ctx, x, 2, 7168, 8, 128, 64, 221184, 1, eligible, false, native_mte);
    assert(!native_mte);
    return launch;
}
void check_capability_and_cached_refusal() {
    for (int mode=0; mode<6; ++mode) {
        reset();
        if (mode==0) ShmemArena::supported=false;
        if (mode==1) ShmemArena::peer_capabilities=0;
        if (mode==2) ShmemArena::result=0;
        if (mode==3) mock::arch=3500;
        if (mode==4) mock::cores=15;
        if (mode==5) mock::ub_bytes=0;
        EPLowLatencyState state("group",buffer_bytes); deep_ep::comm::Context ctx;
        state.initialize(ctx); auto x=input();
        assert(!prepare(state,ctx,x)); assert(!prepare(state,ctx,x));
        assert(ShmemArena::agreements==1);
        assert(ShmemArena::attempts==(mode==2 ? 1 : 0));
        assert(mock::copies==0 && ShmemArena::peer_reads==0);
        state.record_completion(); state.close(); ++cases;
    }
}
void check_capacity_gate() {
    reset(); deep_ep::comm::Context ctx; auto x=input();
    const auto plan=deep_ep::get_dispatch_low_latency_plan(3510,8,2,7168,8,128,64,216*1024,false);
    assert(plan && plan->buffer_bytes>0);
    EPLowLatencyState short_buffer("group",plan->buffer_bytes-1);
    short_buffer.initialize(ctx);
    assert(!prepare(short_buffer,ctx,x)); assert(ShmemArena::attempts==0);
    EPLowLatencyState exact_buffer("group",plan->buffer_bytes);
    exact_buffer.initialize(ctx); auto launch=prepare(exact_buffer,ctx,x);
    assert(launch && ShmemArena::requested_bytes==plan->buffer_bytes);
    exact_buffer.close(); ++cases;
}
void check_group_capacity_and_transport_switch() {
    deep_ep::comm::Context ctx;
    for (int rank = 0; rank < ctx.num_ranks; ++ rank) {
        reset();
        ctx.rank_idx = rank;
        EPLowLatencyState state("group", buffer_bytes);
        state.initialize(ctx);
        const auto batch = [&](int capacity, bool cached) {
            const auto x = torch::empty({capacity - rank, 7168}, {});
            return state.prepare_dispatch(ctx, x, capacity, 7168, 8, 128, 64, 216 * 1024, false, cached, false, true);
        };
        assert(not batch(17, false));
        assert(ShmemArena::attempts == 0);
        const auto fresh = batch(16, false);
        assert(fresh and ShmemArena::attempts == 1);
        state.begin();
        state.record_completion();

        mock::stream = 1;
        const auto native = batch(16, true);
        assert(not native and mock::events.back() == "wait:0->1");
        state.begin();
        state.record_completion();

        mock::stream = 2;
        const auto smaller = batch(8, false);
        assert(smaller and mock::events.back() == "wait:1->2");
        state.begin();
        state.record_completion();
        const auto cached = batch(8, true);
        assert(cached and cached->arena == fresh->arena and ShmemArena::attempts == 1);
        state.begin();
        state.record_completion();
        state.close();
    }
    ++ cases;
}
void check_pending_and_cross_stream() {
    reset(); EPLowLatencyState state("group",buffer_bytes); deep_ep::comm::Context ctx;
    state.initialize(ctx); auto x=input(); auto launch=prepare(state,ctx,x);
    assert(launch && ShmemArena::requested_bytes==buffer_bytes);
    assert(mock::copies==1 && ShmemArena::peer_reads==ctx.num_ranks);
    assert(launch->buffer()==launch->arena->memory+ctx.rank_idx*32);
    const auto table=launch->peer_buffers.storage;
    assert(table->bytes.size()==ctx.num_ranks*sizeof(uint64_t));
    for(int peer=0;peer<ctx.num_ranks;++peer) {
        uint64_t address;
        std::memcpy(&address,table->bytes.data()+peer*sizeof(uint64_t),sizeof(address));
        assert(address==reinterpret_cast<uint64_t>(launch->arena->memory+peer*32));
    }
    const auto allocations=mock::allocation_count;
    state.begin();
    std::weak_ptr<ShmemArena> weak = launch->arena;
    assert(!weak.expired());
    const auto before_reject=mock::events.size();
    rejects([&] { prepare(state,ctx,x); },"outstanding");
    assert(mock::events.size()==before_reject);
    mock::stream=1; state.record_completion(); assert(!weak.expired());
    assert(mock::events.back()=="record:1");
    mock::events.clear(); mock::stream=2; auto next=prepare(state,ctx,x);
    assert(mock::events.size()==1 && mock::events.front()=="wait:1->2");
    assert(next == launch && next->peer_buffers.storage==table && mock::copies==1 && ShmemArena::peer_reads==8);
    assert(mock::allocation_count==allocations && ShmemArena::attempts==1);
    state.begin(); state.record_completion();
    mock::events.clear(); mock::stream=3;
    assert(!prepare(state,ctx,x,true));
    assert(mock::events.size()==1 && mock::events[0]=="wait:2->3");
    state.begin();
    rejects([&] { prepare(state,ctx,x); },"outstanding");
    state.record_completion(); state.close(); assert(ShmemArena::closes==1);
    state.close(); assert(ShmemArena::closes==1 && weak.expired()); ++cases;
}
void check_combine_then_dispatch() {
    reset(); deep_ep::comm::Context ctx; auto x=input();
    EPLowLatencyState state("group", buffer_bytes);
    state.initialize(ctx);
    auto native = combine(state, ctx, x, false);
    assert(!native && ShmemArena::attempts == 0);
    state.begin();
    rejects([&] { combine(state, ctx, x, true); }, "outstanding");
    rejects([&] { prepare(state, ctx, x); }, "outstanding");
    rejects([&] { state.close(); }, "outstanding");
    assert(!ShmemArena::session_failed && ShmemArena::attempts == 0);
    mock::stream = 1;
    state.record_completion();
    assert(mock::events.back() == "record:1");

    mock::stream = 2;
    mock::events.clear();
    auto first = combine(state, ctx, x, true);
    assert(first && mock::events.front() == "wait:1->2");
    std::weak_ptr<ShmemArena> arena = first->arena;
    std::weak_ptr<torch::Storage> table = first->peer_buffers.storage;
    state.begin();
    assert(!arena.expired() && !table.expired());
    rejects([&] { prepare(state, ctx, x); }, "outstanding");
    rejects([&] { state.close(); }, "outstanding");
    assert(ShmemArena::closes == 0);
    mock::stream = 3;
    state.record_completion();
    assert(!arena.expired() && !table.expired() && mock::events.back() == "record:3");

    mock::stream = 4;
    auto dispatch = prepare(state, ctx, x);
    assert(dispatch && mock::events.back() == "wait:3->4");
    assert(dispatch == first && dispatch->arena == arena.lock() && dispatch->peer_buffers.storage == table.lock());
    state.begin();
    state.record_completion();
    mock::stream = 5;
    native = combine(state, ctx, x, false);
    assert(!native && mock::events.back() == "wait:4->5");
    state.begin();
    state.record_completion();
    mock::stream = 6;
    auto repeated = combine(state, ctx, x, true);
    assert(repeated && mock::events.back() == "wait:5->6");
    assert(repeated == first && repeated->arena == arena.lock() && repeated->peer_buffers.storage == table.lock());
    assert(ShmemArena::agreements == 1 && ShmemArena::attempts == 1 && mock::copies == 1);
    state.begin();
    state.record_completion();
    state.close();
    assert(ShmemArena::closes == 1 && arena.expired() && table.expired());
    ++cases;
}
void check_combine_admission() {
    for (int mode = 0; mode < 7; ++mode) {
        reset(); deep_ep::comm::Context ctx; auto x=input();
        EPLowLatencyState state("group", buffer_bytes);
        if (mode == 0) mock::arch = 3500;
        if (mode == 1) mock::cores = 63;
        if (mode == 2) mock::ub_bytes = 190 * 1024 - 1;
        if (mode == 3) ShmemArena::peer_capabilities = 0;
        if (mode == 4) mock::vec_per_ai = 1;
        if (mode == 5) ShmemArena::result = 0;
        if (mode == 6) mock::ub_bytes = 216 * 1024;
        state.initialize(ctx);
        for (int attempt = 0; attempt < 2; ++attempt) {
            auto launch = combine(state, ctx, x, true);
            assert(bool(launch) == (mode == 6));
            state.begin();
            mock::events.clear();
            state.record_completion();
            if (mode == 0 || mode == 5) assert(mock::events.empty());
        }
        assert(ShmemArena::agreements == 1 && ShmemArena::attempts == (mode >= 5 ? 1 : 0));
        assert(mock::copies == (mode == 6 ? 1 : 0));
        if (mode == 3) {
            assert(!prepare(state, ctx, x));
            assert(ShmemArena::attempts == 0);
        }
        state.close();
        ++cases;
    }
}
void check_group_refusal_and_mixed_combine() {
    using deep_ep::comm::DecodeAgreement;
    for (int world : {8, 16, 32, 64, 128}) {
        for (int rank = 0; rank < world; ++rank) {
            reset();
            deep_ep::comm::Context ctx;
            ctx.num_ranks = world; ctx.rank_idx = rank;
            EPLowLatencyState state("group", buffer_bytes);
            state.initialize(ctx);
            const int m = world == 8 ? 64 : 32;
            const int b = std::max(1, m-rank);
            auto x = input();
            const auto select = [&](int tokens, bool options) {
                bool native_mte = true;
                assert(!state.prepare_combine(ctx,x,m,7168,8,128,64,221184,tokens,options,true,native_mte));
                return native_mte;
            };
            assert(select(b,true) == ((world==8 && rank>=4) || (world==16 && rank==15)));
            assert(!select(b+1,true));
            assert(!select(b,false));
            DecodeAgreement::all_match=false;
            assert(!select(b,true));
            assert(ShmemArena::attempts==0);
            assert(DecodeAgreement::votes == ((world == 8 || world == 16) ? 4 : 0));
            if (world == 8) {
                const int votes=DecodeAgreement::votes;
                assert(!combine(state,ctx,x,true));
                assert(!prepare(state,ctx,x));
                assert(DecodeAgreement::votes==votes+2 && ShmemArena::attempts==0);
                DecodeAgreement::all_match=true; DecodeAgreement::capturing=true;
                assert(!combine(state,ctx,x,true));
                assert(!prepare(state,ctx,x));
                assert(DecodeAgreement::votes==votes+2 && ShmemArena::attempts==0);
            }
            state.close();
        }
    }
    for (int failure = 0; failure < 3; ++failure) {
        reset();
        deep_ep::comm::Context ctx;
        if (failure == 0) --ctx.num_gpu_timeout_cycles;
        if (failure == 1) mock::soc = "Ascend950DT_9599";
        if (failure == 2) mock::soc = nullptr;
        EPLowLatencyState state("group", buffer_bytes);
        state.initialize(ctx);
        auto x = input();
        assert(!prepare(state,ctx,x));
        state.begin();
        assert(!combine(state,ctx,x,true));
        state.record_completion();
        assert(DecodeAgreement::votes == 0 && ShmemArena::attempts == 0);
        state.close();
    }
    ++cases;
}
void check_combine_failures() {
    for (int mode = 0; mode < 4; ++mode) {
        reset(); deep_ep::comm::Context ctx; auto x=input();
        EPLowLatencyState state("group", buffer_bytes);
        state.initialize(ctx);
        const auto launch = combine(state, ctx, x, mode != 0);
        assert(bool(launch) == (mode != 0));
        state.begin();
        if (mode == 2) mock::event_failure = true;
        if (mode == 3) {
            state.record_completion();
            mock::wait_failure = true;
            rejects([&] { combine(state, ctx, x, false); }, "event wait failure");
        } else {
            rejects([&] {
                const auto guard = state.failure_guard();
                if (mode == 2) state.record_completion();
                else throw std::runtime_error("combine launch failed");
            }, mode == 2 ? "event" : "combine");
        }
        rejects([&] { prepare(state, ctx, x); }, "failed");
        rejects([&] { combine(state, ctx, x, false); }, "failed");
        rejects([&] { state.close(); }, "failed");
        assert(ShmemArena::session_failed == (mode != 0));
        ++cases;
    }
}
void check_preparation_failures() {
    for(int mode=0;mode<5;++mode) {
        reset(); EPLowLatencyState state("group",buffer_bytes); deep_ep::comm::Context ctx;
        state.initialize(ctx); auto x=input();
        if(mode==0) ShmemArena::result=-1;
        if(mode==1) mock::fail_allocation=mock::allocation_count+1;
        if(mode==2) mock::fail_copy=1;
        if(mode==3) ShmemArena::bad_peer=5;
        if(mode==4) ShmemArena::peer_misalignment=1;
        rejects([&] { prepare(state,ctx,x); },mode==0 ? "arena" : mode==1 ? "allocation" : mode==2 ? "ACL" : "peers");
        mock::fail_allocation=mock::fail_copy=-1; ShmemArena::result=1;
        ShmemArena::bad_peer=-1; ShmemArena::peer_misalignment=0;
        const auto attempts=ShmemArena::attempts;
        rejects([&] { prepare(state,ctx,x); },"failed");
        assert(ShmemArena::attempts==attempts);
        rejects([&] { state.close(); },"failed");
        ++cases;
    }
}
void check_guard_and_completion_failures() {
    for(int mode=0;mode<2;++mode) {
        reset(); EPLowLatencyState state("group",buffer_bytes); deep_ep::comm::Context ctx;
        state.initialize(ctx); auto x=input(); auto launch=prepare(state,ctx,x); state.begin();
        std::weak_ptr<ShmemArena> weak = launch->arena;
        if(mode==1) mock::event_failure=true;
        rejects([&] {
            const auto guard=state.failure_guard();
            if(mode==0) throw std::runtime_error("external dispatch failure");
            state.record_completion();
        },mode==0 ? "dispatch" : "event");
        assert(!weak.expired());
        rejects([&] { prepare(state,ctx,x); },"failed");
        rejects([&] { state.close(); },"failed"); ++cases;
    }
    reset(); EPLowLatencyState state("group",buffer_bytes); deep_ep::comm::Context ctx;
    state.initialize(ctx); auto x=input(); auto launch=prepare(state,ctx,x);
    assert(launch); state.begin(); state.record_completion(); mock::wait_failure=true;
    rejects([&] { prepare(state,ctx,x); },"event wait failure");
    assert(ShmemArena::session_failed && ShmemArena::failure_marks==1);
    rejects([&] { prepare(state,ctx,x); },"failed"); ++cases;
}
void check_shared_session_failure() {
    for(bool fallback : {false,true}) {
        reset(); deep_ep::comm::Context ctx;
        EPLowLatencyState first("group",buffer_bytes),second("group",buffer_bytes);
        first.initialize(ctx); second.initialize(ctx); auto x=input();
        auto a=prepare(first,ctx,x),b=prepare(second,ctx,x);
        assert(a && b && a->arena!=b->arena && a->peer_buffers.storage!=b->peer_buffers.storage);
        first.begin(); first.record_completion(); second.begin(); second.record_completion();
        mock::events.clear(); first.fail();
        assert(ShmemArena::session_failed && ShmemArena::failure_marks==1);
        rejects([&] { prepare(second,ctx,x,fallback); },"shared session failed");
        assert(mock::events.empty());
        assert(ShmemArena::attempts==2 && ShmemArena::closes==0 && ShmemArena::failure_marks==2);
        rejects([&] { prepare(second,ctx,x); },"failed"); ++cases;
    }
}
int main() {
    check_capability_and_cached_refusal(); check_capacity_gate(); check_pending_and_cross_stream();
    check_group_capacity_and_transport_switch();
    check_combine_then_dispatch(); check_combine_admission(); check_group_refusal_and_mixed_combine(); check_combine_failures();
    check_preparation_failures(); check_guard_and_completion_failures(); check_shared_session_failure();
    std::cout << "PASS " << cases << " CPU State source cases\n";
}
'''
    return prefix + production + harness


class DispatchLLStateHostTests(unittest.TestCase):
    def test_decode_policy_grid(self):
        source = (ROOT / 'tests/ep/decode_policy_test.cpp').read_text()
        run_host_cpp(source, include_dirs=(ROOT, ROOT / 'deep_ep/include'))

    def test_actual_state_transitions(self):
        output = run_host_cpp(host_test_source(), include_dirs=(ROOT, ROOT / 'deep_ep/include'))
        self.assertIn('PASS 32 CPU State source cases', output)


if __name__ == '__main__':
    unittest.main()
