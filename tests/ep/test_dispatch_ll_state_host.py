"""Exercise production decode admission and event ordering with host-only runtime stubs."""

import re
import unittest

from _host_test_helpers import ROOT, run_host_cpp

HEADER = ROOT / 'csrc/buffers/low_latency_state.hpp'
CANN_HEADERS = {'simt_api/device_warp_functions.h': '#pragma once\n',
                'utils/debug/asc_assert.h': '#pragma once\n#include <cassert>\n'}


def host_test_source():
    prefix = r"""
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#define ACL_CHECK(op) do { if (op) throw std::runtime_error("capture"); } while(false)
#define EP_HOST_ASSERT(cond) do { if (!(cond)) throw std::runtime_error(#cond); } while(false)
#include "csrc/comm/decode_topology.hpp"
#include "csrc/kernels/ep/kernel_mode.hpp"
#include "csrc/kernels/ep/dispatch_low_latency_policy.hpp"
using deep_ep::EPKernelMode;

namespace torch {
struct Tensor {
    std::shared_ptr<int> allocation;
    bool defined() const { return allocation != nullptr; }
};
}
namespace mock {
std::vector<std::string> events;
int stream = 0, arch = 3510, cores = 64, vec_per_ai = 2, ub_bytes = 216 * 1024;
const char* soc = "Ascend950DT_9582";
bool event_failure = false, wait_failure = false, capable_peer = true;
int setups = 0;
void note(const std::string& value) { events.push_back(value); }
}
using aclmdlRI = void*;
enum aclmdlRICaptureStatus { ACL_MODEL_RI_CAPTURE_STATUS_NONE, ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE };
aclmdlRICaptureStatus capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
int aclrtCaptureError = 0;
int aclmdlRICaptureGetInfo(int, aclmdlRICaptureStatus* status, aclmdlRI*) {
    *status = capture_status;
    return aclrtCaptureError;
}
const char* aclrtGetSocName() { return mock::soc; }
namespace c10_npu {
using NPUStream = int;
NPUStream getCurrentNPUStream() { return mock::stream; }
}
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
int aclrtMemcpy(void* dst, size_t capacity, const void* src, size_t bytes, int kind) {
    assert(bytes <= capacity && kind == ACL_MEMCPY_HOST_TO_DEVICE);
    std::memcpy(dst, src, bytes);
    return 0;
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
    explicit EventHandle(c10_npu::NPUStream recorded = c10_npu::getCurrentNPUStream()): stream(recorded) {
        mock::note("record:" + std::to_string(stream));
        if (mock::event_failure) throw std::runtime_error("event record failure");
    }
    void current_stream_wait() const {
        mock::note("wait:" + std::to_string(stream) + "->" + std::to_string(mock::stream));
        if (mock::wait_failure) throw std::runtime_error("event wait failure");
    }
};
namespace comm {
struct Context {
    int num_ranks = 8, num_cpu_timeout_secs = 300;
    int64_t num_gpu_timeout_cycles = 100 * kNumSystemCyclesPerSecond;
    void* workspace = nullptr;
    bool finalized = false;
    void check_available() const { EP_HOST_ASSERT(not finalized); }
    void finalize() { finalized = true; }
};
}
}
"""
    production = re.sub(r'^#(?:include|pragma).*\n', '', HEADER.read_text(), flags=re.M)
    dispatch_header = (ROOT / 'csrc/kernels/ep/dispatch.hpp').read_text()
    launcher = dispatch_header[dispatch_header.index('static void launch_ep_dispatch('):
                               dispatch_header.index('static void launch_dispatch(')]
    recording_jit = r"""
struct Config { int num_blocks, num_ubuf_bytes; };
struct TestJit {
    int launches = 0;
    std::string source;
    int compile(const char*, const std::string& code) { source = code; return 1; }
    void launch(int, Config, auto...) { ++launches; }
} test_jit;
auto jit = &test_jit;
"""
    harness = r"""
using deep_ep::comm::Context;
using deep_ep::comm::DecodeTopology;
using deep_ep::EPLowLatencyState;

void initialize(EPLowLatencyState& state, std::shared_ptr<Context> ctx) {
    auto topology = state.local_topology(ctx->num_ranks);
    if (ctx->num_ranks > 1) {
        topology.gather_ports(3000, [&](DecodeTopology::Record local) {
            ++mock::setups;
            std::vector<DecodeTopology::Record> peers(ctx->num_ranks, {3000, mock::capable_peer ? 64 : 0, {}});
            peers[0] = local;
            return peers;
        });
    }
    state.attach_context(ctx, topology);
}
void initialize(EPLowLatencyState& state, const Context& ctx) {
    initialize(state, std::make_shared<Context>(ctx));
}

bool dispatch(EPLowLatencyState& state, const Context& ctx, int m = 2, int sf_words = 0) {
    state.prepare();
    return state.prepare_dispatch(ctx.num_ranks, m, 7168, 8, 128, 64, 216 * 1024, sf_words, sf_words > 0);
}
bool combine(EPLowLatencyState& state, const Context& ctx, int m = 2) {
    state.prepare();
    return state.prepare_combine(ctx.num_ranks, m, 7168, 8, 128, 64, 216 * 1024);
}
template <class F> void rejects(F fn, std::string_view message) {
    try { fn(); } catch (const std::runtime_error& e) {
        assert(std::string_view(e.what()).find(message) != std::string_view::npos);
        return;
    }
    assert(false);
}

void check_admission() {
    for (int mode = 0; mode < 9; ++mode) {
        Context ctx;
        if (mode == 0) mock::soc = nullptr;
        if (mode == 1) mock::arch = 3500;
        if (mode == 2) mock::cores = 63;
        if (mode == 3) mock::vec_per_ai = 1;
        if (mode == 4) mock::ub_bytes = 100;
        if (mode == 5) mock::capable_peer = false;
        if (mode == 6) ctx.num_ranks = 1;
        if (mode == 7) mock::cores = 80;
        EPLowLatencyState state;
        const auto setups = mock::setups;
        initialize(state, ctx);
        for (int i = 0; i < 3; ++i) {
            assert(dispatch(state, ctx) == (mode == 8));
            assert(combine(state, ctx) == (mode == 8));
        }
        assert(mock::setups == setups + (mode != 6));
        mock::soc = "Ascend950DT_9582"; mock::arch = 3510; mock::cores = 64;
        mock::vec_per_ai = 2; mock::ub_bytes = 216 * 1024;
        mock::capable_peer = true;
    }
    for (int world : {2, 4, 8, 16, 32, 64, 128}) {
        Context ctx;
        ctx.num_ranks = world;
        EPLowLatencyState state;
        initialize(state, ctx);
        for (int m = 1; m <= 16; ++m) {
            assert(dispatch(state, ctx, m) && dispatch(state, ctx, m, 112));
            assert(combine(state, ctx, m));
        }
        assert(!dispatch(state, ctx, 2, -1));
        assert(!state.prepare_dispatch(world, 2, 7168, 8, 128, 64, 216 * 1024, 0, true));
        capture_status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
        assert(!dispatch(state, ctx) && !combine(state, ctx));
        capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
        aclrtCaptureError = 1;
        rejects([&] { dispatch(state, ctx); }, "capture");
        aclrtCaptureError = 0;
        assert(dispatch(state, ctx));
    }
}
void check_capacity_boundary() {
    Context ctx;
    EPLowLatencyState state;
    const auto before = mock::events.size();
    initialize(state, ctx);
    const auto setups = mock::setups;
    const torch::Tensor source{std::make_shared<int>(1)}, staging{std::make_shared<int>(2)};
    rejects([&] {
        auto guard = state.submission_guard(source, staging);
        throw std::runtime_error("JIT");
    }, "JIT");
    for (int capacity : {17, 32, 256}) {
        assert(!deep_ep::use_decode_implementation(capacity));
        state.prepare();
        state.submission_guard(source, staging).begin_submission();
        state.record_completion(false);
    }
    assert(mock::events.size() == before && mock::setups == setups);
    assert(deep_ep::use_decode_implementation(16));
    capture_status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    assert(!dispatch(state, ctx, 16) && mock::setups == setups);
    capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    assert(dispatch(state, ctx, 16) && combine(state, ctx, 16));
    assert(mock::setups == setups);
}
void check_events() {
    Context ctx;
    EPLowLatencyState state;
    initialize(state, ctx);
    assert(dispatch(state, ctx));
    state.submission_guard().begin_submission();
    rejects([&] { combine(state, ctx); }, "outstanding");
    mock::stream = 1;
    state.record_completion(true);
    const auto count = mock::events.size();
    assert(mock::events.back() == "record:1");
    assert(combine(state, ctx));
    assert(mock::events.size() == count);
    state.submission_guard().begin_submission();
    state.record_completion(true);
    mock::stream = 2;
    state.prepare();
    assert(mock::events.back() == "wait:1->2");
    torch::Tensor source{std::make_shared<int>(1)}, staging{std::make_shared<int>(2)};
    std::weak_ptr<int> retained = staging.allocation;
    state.submission_guard(source, staging).begin_submission();
    staging = {};
    assert(!retained.expired());
    rejects([&] { dispatch(state, ctx); }, "outstanding");
    state.record_completion(true);
    assert(mock::events.back() == "record:2");
    const auto event_count = mock::events.size();
    assert(state.completion_event().stream == 2 && mock::events.size() == event_count);
    mock::note("unrelated work on stream 2");
    assert(retained.expired());
    mock::stream = 3;
    assert(dispatch(state, ctx));
    assert(mock::events.size() == event_count + 2 && mock::events.back() == "wait:2->3");
    state.submission_guard().begin_submission();
    state.record_completion(false);  // A native call that did not request an event follows main's stream contract.
    ++mock::stream;
    state.prepare();
    assert(mock::events.size() == event_count + 2);
}
void check_deferred_epilogue_retry() {
    EPLowLatencyState state;
    initialize(state, Context{});
    auto submit = [&]() -> std::function<void()> {
        state.prepare();
        state.submission_guard().begin_submission();
        return EPLowLatencyState::make_deferred_epilogue([&] {
            auto guard = state.epilogue_guard(true);
            state.record_completion(true);
        });
    };

    auto first = submit();
    auto retry_first = first;
    rejects([&] {
        first();
        throw std::runtime_error("Python postprocessing failed");
    }, "postprocessing failed");
    auto second = submit();
    rejects(retry_first, "already");
    rejects([&] { state.prepare(); }, "outstanding");
    second();
    state.prepare();
}
void submit_dispatch(EPLowLatencyState& state, EPKernelMode mode, int vec_cores, int64_t timeout_cycles) {
    auto guard = state.submission_guard();
    launch_ep_dispatch(mode, [&] { guard.begin_submission(); }, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       0, 8, 2, 2, 128, 8, 128, false, false, 14336, 0, 0, 0, 216 * 1024, vec_cores, timeout_cycles);
}
void check_invalid_core_count_retry() {
    EPLowLatencyState state;
    Context ctx;
    initialize(state, ctx);
    rejects([&] { submit_dispatch(state, EPKernelMode::DecodeURMA, 8, ctx.num_gpu_timeout_cycles); }, "16 <= num_vec_cores");
    assert(jit->launches == 0);
    assert(dispatch(state, ctx));
    submit_dispatch(state, EPKernelMode::DecodeURMA, 64, ctx.num_gpu_timeout_cycles);
    assert(jit->launches == 1);
    state.record_completion(true);
    assert(combine(state, ctx));
}
void check_gpu_timeouts() {
    for (int seconds : {1, 99, 100, 101, 300}) {
        Context ctx;
        ctx.num_gpu_timeout_cycles = seconds * deep_ep::comm::kNumSystemCyclesPerSecond;
        EPLowLatencyState state;
        initialize(state, ctx);
        assert(dispatch(state, ctx) && combine(state, ctx));
        submit_dispatch(state, EPKernelMode::DecodeMTE, 64, ctx.num_gpu_timeout_cycles);
        assert(jit->source.find(std::format(", {}, false, true, false>", ctx.num_gpu_timeout_cycles)) != std::string::npos);
        state.record_completion(true);
    }
}
void check_failures() {
    Context ctx;
    for (int mode = 0; mode < 3; ++mode) {
        EPLowLatencyState state;
        initialize(state, ctx);
        assert(dispatch(state, ctx));
        rejects([&] { auto guard = state.submission_guard(); throw std::runtime_error("preflight"); }, "preflight");
        assert(dispatch(state, ctx));
        if (mode == 0) {
            rejects([&] { auto guard = state.submission_guard(); guard.begin_submission();
                          throw std::runtime_error("submit"); }, "submit");
        } else if (mode == 1) {
            state.submission_guard().begin_submission();
            mock::event_failure = true;
            rejects([&] { auto guard = state.epilogue_guard(false); state.record_completion(true); }, "record");
            mock::event_failure = false;
        } else {
            state.submission_guard().begin_submission();
            state.record_completion(true);
            ++mock::stream;
            mock::wait_failure = true;
            rejects([&] { combine(state, ctx); }, "wait");
            mock::wait_failure = false;
        }
        rejects([&] { dispatch(state, ctx); }, "failed");
        rejects([&] { state.require_healthy(); }, "failed");
    }
}
void check_resource_lifetime() {
    for (int mode = 0; mode < 4; ++mode) {
        auto ctx = std::make_shared<Context>();
        auto source = torch::Tensor{std::make_shared<int>(1)};
        auto staging = torch::Tensor{std::make_shared<int>(2)};
        std::weak_ptr<Context> window = ctx;
        std::weak_ptr<int> input = source.allocation, send_buffer = staging.allocation;
        {
            EPLowLatencyState state;
            initialize(state, ctx);
            assert(dispatch(state, *ctx));
            ctx.reset();
            try {
                auto submission = state.submission_guard(source, staging);
                source = {};
                staging = {};
                if (mode == 0) throw std::runtime_error("before launch");
                submission.begin_submission();
                if (mode == 1) state.record_completion(true);
                if (mode == 2) throw std::runtime_error("after launch");
                // mode 3 drops a submitted operation without running the epilogue.
            } catch (const std::runtime_error&) {}
            assert(input.expired() == (mode < 2) && send_buffer.expired() == (mode < 2));
            assert(!window.expired() && state.unfinished() == (mode >= 2));
        }
        assert(window.expired() == (mode < 2));
        assert(input.expired() == (mode < 2) && send_buffer.expired() == (mode < 2));
    }
}
int main() {
    check_admission();
    check_events();
    check_deferred_epilogue_retry();
    check_invalid_core_count_retry();
    check_gpu_timeouts();
    check_failures();
    check_resource_lifetime();
    check_capacity_boundary();
}
"""
    return prefix + production + recording_jit + launcher + harness


class DispatchLLStateHostTests(unittest.TestCase):
    def test_decode_policy_grid(self):
        source = (ROOT / 'tests/ep/decode_policy_test.cpp').read_text()
        run_host_cpp(source, include_dirs=(ROOT, ROOT / 'deep_ep/include'), headers=CANN_HEADERS)

    def test_actual_state_transitions(self):
        run_host_cpp(host_test_source(), include_dirs=(ROOT, ROOT / 'deep_ep/include'), headers=CANN_HEADERS)

    def test_deferred_teardown_and_failure(self):
        base = (ROOT / 'csrc/buffers/base.hpp').read_text()
        ep = (ROOT / 'csrc/buffers/ep.hpp').read_text()
        cleanup = base[base.index('    void destroy_on_destruction'):base.index('public:')]
        destructor = ep[ep.index('    ~EPBuffer()'):ep.index('    EventHandle lb_prefetch_weights')]
        stubs = r'''
namespace comm {
int drains = 0;
bool drain_failed = false;
std::weak_ptr<int> send_source;
bool check_send_source = false;
void barrier(Context&, bool) {
    ++drains;
    if (check_send_source) assert(!send_source.expired());
    if (drain_failed) throw std::runtime_error("drain");
}
}
int aclrtHostUnregister(void*) { return 0; }
int aclrtFreeHost(void*) { return 0; }
struct Base {
    bool explicitly_destroy = false, destroyed = false;
    bool in_destructor = false;
    std::shared_ptr<Context> main_context;
    virtual ~Base() noexcept(false) = default;
    virtual void destroy() = 0;
'''
        fixture = r'''
};
struct EPBuffer : Base {
    std::shared_ptr<Context> context = std::make_shared<Context>();
    std::shared_ptr<EPLowLatencyState> low_latency_state = std::make_shared<EPLowLatencyState>();
    void* host_workspace = nullptr;
    torch::Tensor lb_storage;
    EPBuffer() {
        initialize(*low_latency_state, context);
    }
'''
        cases = r'''
};
void submit_send(EPBuffer& buffer) {
    torch::Tensor source{std::make_shared<int>(1)}, staging{std::make_shared<int>(2)};
    comm::send_source = source.allocation;
    comm::check_send_source = true;
    buffer.low_latency_state->submission_guard(source, staging).begin_submission();
}
int main() {
    static_assert(std::is_nothrow_destructible_v<EPBuffer>);
    { auto buffer = std::make_unique<EPBuffer>(); }
    assert(comm::drains == 1);
    {
        auto buffer = std::make_unique<EPBuffer>();
        assert(dispatch(*buffer->low_latency_state, *buffer->context));
        rejects([&] {
            auto guard = buffer->low_latency_state->submission_guard();
            guard.begin_submission();
            throw std::runtime_error("launch");
        }, "launch");
        rejects([&] { buffer->destroy(); }, "failed");
    }
    assert(comm::drains == 1);
    comm::drain_failed = true;
    { auto buffer = std::make_unique<EPBuffer>(); }
    assert(comm::drains == 2);
    comm::drain_failed = false;
    {
        auto buffer = std::make_unique<EPBuffer>();
        buffer->explicitly_destroy = true;
        submit_send(*buffer);
        auto state = buffer->low_latency_state;
        buffer->destroy();
        buffer.reset();
        assert(comm::drains == 3 && comm::send_source.expired());
        assert(!state->unfinished());
        rejects([&] { state->epilogue_guard(true); }, "not finalized");
    }
    {
        auto buffer = std::make_unique<EPBuffer>();
        submit_send(*buffer);
        comm::drain_failed = true;
        rejects([&] { buffer->destroy(); }, "drain");
        assert(!comm::send_source.expired() && buffer->low_latency_state->unfinished());
        rejects([&] { buffer->low_latency_state->epilogue_guard(true); }, "operation failed");
    }
    assert(comm::drains == 4);
}
'''
        source = host_test_source().split('int main() {')[0] + stubs + cleanup + fixture + destructor + cases
        run_host_cpp(source, include_dirs=(ROOT, ROOT / 'deep_ep/include'), headers=CANN_HEADERS)

    def test_non_ep_buffer_destroy_reports_failure_and_retries_locally(self):
        buffers = ROOT / 'csrc/buffers'
        base = (buffers / 'base.hpp').read_text()
        context = (ROOT / 'csrc/kernels/comm/api.hpp').read_text()
        barrier = (ROOT / 'csrc/comm/barrier.hpp').read_text()
        headers = {
            'base.inc': base[base.index('class BufferBase'):base.index('    void print_memory_usage')] + '};',
            'context.inc': context[context.index('    void check_available()'):context.index('\n};')],
            'drain.inc': barrier[barrier.index('inline void Context::drain_for_destroy'):barrier.index('}  // namespace')],
        }
        for name in ('PP', 'Engram', 'Bucket'):
            body = (buffers / f'{name.lower()}.hpp').read_text()
            start = body.index(f'    ~{name}Buffer()')
            end = body.index('\n    }', body.index('    void destroy() override', start)) + len('\n    }')
            headers[f'{name}.inc'] = body[start:end]
        run_host_cpp(r'''
#include <cassert>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>
#define EP_HOST_ASSERT(cond) do { if (!(cond)) throw std::runtime_error(#cond); } while (false)
namespace torch { struct Tensor {}; }
namespace comm {
struct Resource {
    static inline int destructor_errors = 0;
    bool fail = false;
    void close() { if (fail) throw std::runtime_error("device free"); }
    ~Resource() noexcept { try { close(); } catch (...) { ++destructor_errors; } }
};
struct Context {
    std::shared_ptr<Resource> hccl_context = std::make_shared<Resource>();
    std::shared_ptr<Resource> symmetric_memory = std::make_shared<Resource>();
    bool finalized = false;
    int barriers = 0;
#include "context.inc"
};
void barrier(Context& context, bool sync) {
    context.check_available();
    assert(sync);
    ++context.barriers;
}
#include "drain.inc"
}
#include "base.inc"
struct SingleBuffer : BufferBase {
    using BufferBase::BufferBase;
    std::shared_ptr<comm::Context> context = std::make_shared<comm::Context>();
    auto groups() const { return std::vector{context}; }
};
struct PPBuffer : SingleBuffer {
    PPBuffer(bool explicit_destroy) : SingleBuffer(explicit_destroy) {}
    torch::Tensor storage;
#include "PP.inc"
};
struct EngramBuffer : SingleBuffer {
    EngramBuffer(bool explicit_destroy) : SingleBuffer(explicit_destroy) {}
#include "Engram.inc"
};
struct BucketBuffer : BufferBase {
    BucketBuffer(bool explicit_destroy) : BufferBase(explicit_destroy) {}
    std::vector<std::shared_ptr<comm::Context>> contexts{
        std::make_shared<comm::Context>(), std::make_shared<comm::Context>(), std::make_shared<comm::Context>()};
    auto groups() const { return contexts; }
    torch::Tensor storage;
#include "Bucket.inc"
};

template <class Buffer> void check_destroy() {
    Buffer buffer(true);
    auto groups = buffer.groups();
    const auto& failed_group = groups[groups.size() / 2];
    failed_group->symmetric_memory->fail = true;
    try {
        buffer.destroy();
        assert(false);
    } catch (const std::runtime_error& e) {
        assert(std::string_view(e.what()) == "device free");
    }
    for (const auto& group : groups) {
        try { comm::barrier(*group, true); assert(false); } catch (const std::runtime_error&) {}
        assert(group->barriers == 1);
    }
    failed_group->symmetric_memory->fail = false;
    buffer.destroy();
    for (const auto& group : groups)
        assert(group->barriers == 1 && !group->hccl_context && !group->symmetric_memory);

    const auto before = comm::Resource::destructor_errors;
    {
        Buffer automatic(false);
        automatic.groups().front()->symmetric_memory->fail = true;
    }
    assert(comm::Resource::destructor_errors == before + 1);
}
int main() {
    check_destroy<PPBuffer>();
    check_destroy<EngramBuffer>();
    check_destroy<BucketBuffer>();
}
''', headers=headers)

    def test_group_topology_and_transport_limits(self):
        source = (ROOT / 'tests/ep/decode_topology_host_test.cpp').read_text()
        hccl = (ROOT / 'csrc/kernels/comm/hccl.hpp').read_text()
        context = (ROOT / 'csrc/kernels/comm/api.hpp').read_text()
        headers = {'gather_listen_ports.inc': hccl[hccl.index('    std::vector<uint16_t> gather_listen_ports('):
                                                    hccl.index('    void wait_for_channels(')],
                   'finalize.inc': context[context.index('    void finalize()'):context.index('\n};')],
                   'close.inc': hccl[hccl.index('    void close()'):hccl.index('    ~HCCLContext()')]}
        run_host_cpp(source, include_dirs=(ROOT,), headers=headers)


if __name__ == '__main__':
    unittest.main()
