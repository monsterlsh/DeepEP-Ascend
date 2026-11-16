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
namespace c10_npu {
using NPUStream = int;
NPUStream getCurrentNPUStream() { return mock::stream; }
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
    explicit EventHandle(c10_npu::NPUStream recorded): stream(recorded) {
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
    static bool is_capturing() { return capturing; }
    bool matches(void*, int, int, uint16_t proposal) {
        proposals.push_back(proposal);
        return matching_peer && proposal != 0;
    }
};
struct DecodeTopology {
    bool group_enabled = false;
    static DecodeTopology discover(const std::string&, int, int, int64_t, int64_t, bool local) {
        ++DecodeAgreement::initial_votes;
        return {local && DecodeAgreement::capable_peer};
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

void initialize(EPLowLatencyState& state, const Context& ctx) {
    state.initialize("ep", ctx.num_ranks, ctx.num_cpu_timeout_secs,
                     int(ctx.num_gpu_timeout_cycles / deep_ep::comm::kNumSystemCyclesPerSecond), buffer_bytes, buffer_bytes);
}

bool dispatch(EPLowLatencyState& state, const Context& ctx, int m = 2, bool cached = false,
              bool prologue = false, bool options = true, int num_sf_packs = 0) {
    return state.prepare_dispatch(ctx, m, 7168, 8, 128, 64, 216 * 1024, num_sf_packs, cached, prologue, options);
}
bool combine(EPLowLatencyState& state, const Context& ctx, int m = 2, bool options = true,
             int b = 1, bool prologue = false) {
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
        if (mode == 8) ctx.num_ranks = 1;
        const auto votes = DecodeAgreement::initial_votes;
        DecodeAgreement::proposals.clear();
        EPLowLatencyState state(buffer_bytes);
        initialize(state, ctx);
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
    for (int sf_words : {0, 56, 112}) {
        const auto plan = deep_ep::get_dispatch_low_latency_plan(3510, 8, 2, 7168, 8, 128, 64, 216 * 1024, sf_words);
        for (auto bytes : {plan->buffer_bytes - 1, plan->buffer_bytes}) {
            EPLowLatencyState state(bytes);
            initialize(state, ctx);
            assert(dispatch(state, ctx, 2, false, false, true, sf_words) == (bytes == plan->buffer_bytes));
        }
    }
}

void check_proposals() {
    Context ctx;
    EPLowLatencyState state(buffer_bytes);
    initialize(state, ctx);
    DecodeAgreement::proposals.clear();
    for (bool cached : {false, true})
        for (bool prologue : {false, true}) {
            assert(dispatch(state, ctx, 3, cached, prologue));
            assert(DecodeAgreement::proposals.back() == 1 + 2 * cached + prologue);
        }
    assert(!dispatch(state, ctx, 3, false, false, false));
    assert(dispatch(state, ctx, 3, false, false, true, 112));
    assert(!dispatch(state, ctx, 3, false, false, true, -1));
    assert(dispatch(state, ctx, 9, true));
    assert(dispatch(state, ctx, 9, false));
    assert(!combine(state, ctx, 3, false));
    assert(!combine(state, ctx, 3, true, 0));
    assert(DecodeAgreement::proposals.back() == 0);
    assert(combine(state, ctx, 3, true, 3, true));
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
void check_all_rank_admission() {
    for (int world : {2, 4, 8, 16, 32, 64, 128})
        for (int rank = 0; rank < world; ++rank) {
            Context ctx;
            ctx.num_ranks = world; ctx.rank_idx = rank;
            EPLowLatencyState state(buffer_bytes);
            initialize(state, ctx);
            for (int m = 1; m <= 256; ++m) {
                for (bool prologue : {false, true}) {
                    assert(dispatch(state, ctx, m, false, prologue));
                    assert(dispatch(state, ctx, m, true, prologue));
                    assert(combine(state, ctx, m, true, m, prologue));
                    assert(combine(state, ctx, m, true, std::max(1, m - rank), prologue));
                }
            }
            const auto votes = DecodeAgreement::proposals.size();
            for (int m : {0, 257}) {
                assert(!dispatch(state, ctx, m) && !combine(state, ctx, m));
            }
            assert(DecodeAgreement::proposals.size() == votes);
        }
}
void check_threshold_override() {
#ifdef _WIN32
    assert(_putenv_s("EP_MTE_BS_THRESHOLD", "16") == 0);
#else
    assert(setenv("EP_MTE_BS_THRESHOLD", "16", 1) == 0);
#endif
    Context ctx;
    EPLowLatencyState state(buffer_bytes);
    initialize(state, ctx);
    assert(combine(state, ctx, 16));
    assert(!combine(state, ctx, 17));
    assert(DecodeAgreement::proposals.back() == 0);
    assert(dispatch(state, ctx, 256));
}
void check_events() {
    Context ctx;
    EPLowLatencyState state(buffer_bytes);
    initialize(state, ctx);
    assert(dispatch(state, ctx));
    state.begin();
    rejects([&] { combine(state, ctx); }, "outstanding");
    rejects([&] { state.close(); }, "outstanding");
    mock::stream = 1;
    state.record_completion();
    const auto count = mock::events.size();
    assert(mock::events.back() == "record:1");
    assert(combine(state, ctx));
    assert(mock::events.size() == count);
    state.begin();
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
        initialize(state, ctx);
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
            ++mock::stream;
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
    check_all_rank_admission();
    check_events();
    check_failures();
    check_threshold_override();
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
            'acl/acl.h', 'hccl/hccl.h', 'hccl/hccl_comm.h', 'torch_npu/csrc/core/npu/NPUStream.h')}
        headers['deep_ep/common/exception.hpp'] = (
            '#pragma once\n#define ACL_CHECK(op) do { if ((op) != 0) throw std::runtime_error("ACL"); } while (false)\n'
            '#define HCCL_CHECK(op) ACL_CHECK(op)\n')
        run_host_cpp(source, include_dirs=(ROOT,), headers=headers)


if __name__ == '__main__':
    unittest.main()
