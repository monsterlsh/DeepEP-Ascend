"""Host checks of the combine JIT launcher and its device destination selection."""

import unittest

from _host_test_helpers import ROOT, run_host_cpp


class CombineNativeHostTests(unittest.TestCase):
    def test_kernel_specialization_and_runtime_arguments(self):
        host = (ROOT / 'csrc/kernels/ep/combine.hpp').read_text()
        launcher = host[host.index('static void* launch_combine('):
                        host.index('static void launch_combine_reduce_epilogue(')]
        run_host_cpp(r'''
#include <cassert>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>

#define EP_HOST_ASSERT(cond) do { if (!(cond)) throw std::invalid_argument(#cond); } while (false)
struct Config { int num_blocks, num_ubuf_bytes; };
struct RecordingJit {
    std::string name, source;
    void *buffer, *send_buffer;
    int rank, launches = 0;
    int compile(std::string kernel, std::string code) {
        name = kernel;
        source = code;
        return 1;
    }
    void launch(int, Config, void*, void*, void*, void*, void* recv, void* send,
                void*, void*, int current_rank, const int* servers, void* scratch) {
        assert((servers != nullptr) == (scratch != nullptr));
        buffer = recv;
        send_buffer = send;
        rank = current_rank;
        ++launches;
    }
} recording_jit;
auto jit = &recording_jit;
''' + launcher + r'''
int main() {
    int recv, send, weights, servers[4]{}, scratch;
    for (bool direct : {false, true})
        for (bool barrier : {false, true})
            for (bool weighted : {false, true})
              for (bool fp16 : {false, true})
                for (bool relay : {false, true}) {
                if (relay && !direct) continue;
                auto staging = direct ? nullptr : &send;
                const int before = jit->launches;
                auto result = launch_combine(nullptr, weighted ? &weights : nullptr, nullptr,
                    nullptr, &recv, staging, nullptr, nullptr, 1, 4, 16, 3, 128,
                    194560, 64, 100, barrier, direct, fp16, relay ? servers : nullptr, relay ? &scratch : nullptr);
                assert(result == &recv && jit->buffer == &recv && jit->send_buffer == staging);
                assert(jit->rank == 1 && jit->launches == before + 1);
                assert(jit->name == (relay ? "combine_mte_relay" : direct ? "combine_mte" : "combine"));
                auto specialization = std::format("combine_impl<4, 3, 128, 16, 64, 100, 194560, {}, {}, {}, {}, {}>",
                                                  barrier, weighted, direct, fp16 ? "half" : "bfloat16_t", relay);
                assert(jit->source.find(specialization) != std::string::npos);
            }
}
''')

    def test_local_and_remote_return_slots(self):
        device = (ROOT / 'deep_ep/include/deep_ep/impls/ep/combine.hpp').read_text()
        addressing = device[device.index('        auto task_recv_buffer ='):
                            device.index('            const auto output_addr =')]
        run_host_cpp(r'''
#include <cassert>
#include <cstdint>

#define __gm__
namespace AscendC { void WriteGmByPassDCache(int* ptr, int value) { *ptr = value; } }
namespace layout {
struct CommonSignals { uint8_t* native[4]; };
uint8_t* get_sym_ptr(const CommonSignals& signals, uint8_t* local, int peer) {
    assert(local == signals.native[1]);
    return signals.native[peer];
}
}

template <bool kLowLatencyTransport, int kNumTopk>
void check_slot(int owner, layout::CommonSignals native,
                uint8_t* combine_recv_buffer, uint8_t* send_buffer, uint8_t* expected) {
    constexpr bool kDomainRelay = false;
    struct { uint8_t* partials; int* masters; } scratch{};
    constexpr int rank_idx = 1, kNumRanks = 4, kNumMaxTokensPerRank = 4;
    constexpr int kNumVecCores = 2, kNumTokenBytes = 32;
    struct { layout::CommonSignals common_signals; } signals{native};
    struct { int rank_idx, rank_slot_idx; } current_task{owner, 1};
    const bool is_local = owner == rank_idx;
    constexpr int stage_idx = 0, num_stage_tokens = 1;
    struct { int metadata[1][1][kNumTopk + 2]{}; } state;
    auto ub = &state;
    ub->metadata[0][0][kNumTopk] = owner * kNumMaxTokensPerRank + 1;
    ub->metadata[0][0][kNumTopk + 1] = kNumTopk - 1;
''' + addressing + r'''
            assert(output == expected);
        }
}

int main() {
    uint8_t native[4][512], send[512];
    layout::CommonSignals signals{};
    for (int i = 0; i < 4; ++i) {
        signals.native[i] = native[i];
    }
    for (int owner = 0; owner < 4; ++owner) {
        // Token 1. K=3 selects top-k lane 2 (slot 9), K=6 selects rank 1 (slot 5).
        check_slot<true, 3>(owner, signals, native[1], nullptr, native[owner] + 9 * 32);
        check_slot<true, 6>(owner, signals, native[1], nullptr, native[owner] + 5 * 32);
        check_slot<false, 3>(owner, signals, native[1], send,
                             owner == 1 ? native[1] + 9 * 32 : send + 32);
    }
}
''')


if __name__ == '__main__':
    unittest.main()
