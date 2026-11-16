"""Exercise production decode admission and event ordering with host-only runtime stubs."""

import re
import unittest

from _host_test_helpers import ROOT, run_host_cpp

HEADER = ROOT / 'csrc/buffers/low_latency_state.hpp'


def host_test_source():
    prefix = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "csrc/kernels/ep/dispatch_low_latency_policy.hpp"

namespace mock {
std::vector<std::string> events;
int stream = 0, arch = 3510, cores = 64, vec_per_ai = 2, ub_bytes = 216 * 1024;
const char* soc = "Ascend950DT_9582";
bool event_failure = false, wait_failure = false;
void note(const std::string& value) { events.push_back(value); }
}
const char* aclrtGetSocName() { return mock::soc; }
namespace torch {
struct Tensor {
    int batch;
    int size(int dim) const { assert(dim == 0); return batch; }
};
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
    HcclContext hccl;
    HcclContext* hccl_context = &hccl;
};
struct DecodeAgreement {
    inline static bool resources = true, capable_peer = true, matching_peer = true, capturing = false;
    inline static int initial_votes = 0;
    inline static std::vector<uint16_t> proposals;
    bool initialize(int) { return resources; }
    static bool agree_capability(void*, int, int, bool local) {
        ++initial_votes;
        return local && capable_peer;
    }
    static bool is_capturing() { return capturing; }
    bool matches(void*, int, int, uint16_t proposal) {
        proposals.push_back(proposal);
        return matching_peer && proposal != 0;
    }
};
}
}
"""
    production = re.sub(r'^#(?:include|pragma).*\n', '', HEADER.read_text(), flags=re.M)
    harness = r"""
using deep_ep::comm::Context;
using deep_ep::comm::DecodeAgreement;
using deep_ep::EPLowLatencyState;
constexpr int64_t buffer_bytes = 1LL << 30;

bool dispatch(EPLowLatencyState& state, const Context& ctx, int m = 2, bool cached = false,
              bool prologue = false, bool options = true, int extra_tokens = 0, bool fp8 = false) {
    torch::Tensor x{std::max(1, m - ctx.rank_idx) + extra_tokens};
    return state.prepare_dispatch(ctx, x, m, 7168, 8, 128, 64, 216 * 1024, fp8, cached, prologue, options);
}
bool combine(EPLowLatencyState& state, const Context& ctx, int m = 2, bool options = true,
             int extra_tokens = 0, bool prologue = false) {
    const int b = std::max(1, m - ctx.rank_idx) + extra_tokens;
    return state.prepare_combine(ctx, m, 7168, 8, 128, 64, 216 * 1024, b, options, prologue);
}
template <class F> void rejects(F fn, std::string_view message) {
    try { fn(); } catch (const std::runtime_error& e) {
        assert(std::string_view(e.what()).find(message) != std::string_view::npos);
        return;
    }
    assert(false);
}

void check_admission() {
    for (int mode = 0; mode < 10; ++mode) {
        Context ctx;
        if (mode == 0) mock::soc = nullptr;
        if (mode == 1) mock::arch = 3500;
        if (mode == 2) mock::cores = 63;
        if (mode == 3) mock::vec_per_ai = 1;
        if (mode == 4) mock::ub_bytes = 100;
        if (mode == 5) --ctx.num_gpu_timeout_cycles;
        if (mode == 6) DecodeAgreement::resources = false;
        if (mode == 7) DecodeAgreement::capable_peer = false;
        if (mode == 8) ctx.num_ranks = 64;
        const auto votes = DecodeAgreement::initial_votes;
        DecodeAgreement::proposals.clear();
        EPLowLatencyState state(buffer_bytes);
        state.initialize(ctx);
        assert(DecodeAgreement::initial_votes == votes + (mode != 8));
        assert(dispatch(state, ctx) == (mode == 9));
        assert(combine(state, ctx) == (mode == 9));
        assert(DecodeAgreement::proposals.size() == (mode == 9 ? 2u : 0u));
        state.close();
        mock::soc = "Ascend950DT_9582"; mock::arch = 3510; mock::cores = 64;
        mock::vec_per_ai = 2; mock::ub_bytes = 216 * 1024;
        DecodeAgreement::resources = DecodeAgreement::capable_peer = true;
    }

    Context ctx;
    const auto plan = deep_ep::get_dispatch_low_latency_plan(3510, 8, 2, 7168, 8, 128, 64, 216 * 1024, false);
    for (auto bytes : {plan->buffer_bytes - 1, plan->buffer_bytes}) {
        EPLowLatencyState state(bytes);
        state.initialize(ctx);
        assert(dispatch(state, ctx) == (bytes == plan->buffer_bytes));
    }
}

void check_proposals() {
    Context ctx;
    EPLowLatencyState state(buffer_bytes);
    state.initialize(ctx);
    DecodeAgreement::proposals.clear();
    for (bool cached : {false, true})
        for (bool prologue : {false, true}) {
            assert(dispatch(state, ctx, 3, cached, prologue));
            assert(DecodeAgreement::proposals.back() == 1 + 2 * cached + prologue);
        }
    assert(!dispatch(state, ctx, 3, false, false, false));
    assert(!dispatch(state, ctx, 3, false, false, true, 1));
    assert(!dispatch(state, ctx, 3, false, false, true, 0, true));
    assert(!dispatch(state, ctx, 9, true));
    assert(dispatch(state, ctx, 9, false));
    assert(!combine(state, ctx, 3, false));
    assert(!combine(state, ctx, 3, true, 1));
    assert(combine(state, ctx, 3, true, 0, true));
    assert(DecodeAgreement::proposals.back() == 2);
    DecodeAgreement::matching_peer = false;
    assert(!dispatch(state, ctx) && !combine(state, ctx));
    DecodeAgreement::matching_peer = true;
    const auto count = DecodeAgreement::proposals.size();
    DecodeAgreement::capturing = true;
    assert(!dispatch(state, ctx) && !combine(state, ctx));
    DecodeAgreement::capturing = false;
    assert(DecodeAgreement::proposals.size() == count);
}
void check_mixed_combine() {
    for (int world : {8, 16, 32, 64, 128})
        for (int rank = 0; rank < world; ++rank) {
            Context ctx;
            ctx.num_ranks = world; ctx.rank_idx = rank;
            EPLowLatencyState state(buffer_bytes);
            state.initialize(ctx);
            assert(combine(state, ctx, 3) == (world == 8 || world == 16));
            assert(combine(state, ctx, world == 8 ? 64 : 32) ==
                   ((world == 8 && rank >= 4) || (world == 16 && rank == 15)));
        }
}
void check_events() {
    Context ctx;
    EPLowLatencyState state(buffer_bytes);
    state.initialize(ctx);
    assert(dispatch(state, ctx));
    state.begin();
    rejects([&] { combine(state, ctx); }, "outstanding");
    rejects([&] { state.close(); }, "outstanding");
    mock::stream = 1;
    state.record_completion();
    mock::stream = 2;
    assert(!combine(state, ctx, 2, false));
    assert(mock::events.back() == "wait:1->2");
    state.begin();
    state.record_completion();
    mock::stream = 3;
    assert(dispatch(state, ctx, 2, true));
    assert(mock::events.back() == "wait:2->3");
    state.close();
}
void check_failures() {
    Context ctx;
    for (int mode = 0; mode < 3; ++mode) {
        EPLowLatencyState state(buffer_bytes);
        state.initialize(ctx);
        assert(dispatch(state, ctx));
        state.begin();
        if (mode == 0) {
            rejects([&] { auto guard = state.failure_guard(); throw std::runtime_error("submit"); }, "submit");
        } else if (mode == 1) {
            mock::event_failure = true;
            rejects([&] { auto guard = state.failure_guard(); state.record_completion(); }, "record");
            mock::event_failure = false;
        } else {
            state.record_completion();
            mock::wait_failure = true;
            rejects([&] { combine(state, ctx); }, "wait");
            mock::wait_failure = false;
        }
        rejects([&] { dispatch(state, ctx); }, "failed");
        rejects([&] { state.close(); }, "failed");
    }
}
int main() {
    check_admission();
    check_proposals();
    check_mixed_combine();
    check_events();
    check_failures();
}
"""
    return prefix + production + harness


class DispatchLLStateHostTests(unittest.TestCase):
    def test_decode_policy_grid(self):
        source = (ROOT / 'tests/ep/decode_policy_test.cpp').read_text()
        run_host_cpp(source, include_dirs=(ROOT, ROOT / 'deep_ep/include'))

    def test_actual_state_transitions(self):
        run_host_cpp(host_test_source(), include_dirs=(ROOT, ROOT / 'deep_ep/include'))

    def test_hccl_agreement(self):
        source = (ROOT / 'tests/ep/decode_agreement_host_test.cpp').read_text()
        headers = {name: '#pragma once\n' for name in (
            'acl/acl.h', 'hccl/hccl.h', 'torch_npu/csrc/core/npu/NPUStream.h')}
        headers['deep_ep/common/exception.hpp'] = (
            '#pragma once\n#define ACL_CHECK(op) do { if ((op) != 0) throw std::runtime_error("ACL"); } while (false)\n'
            '#define HCCL_CHECK(op) ACL_CHECK(op)\n')
        run_host_cpp(source, include_dirs=(ROOT,), headers=headers)


if __name__ == '__main__':
    unittest.main()
