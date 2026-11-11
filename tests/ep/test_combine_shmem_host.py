"""Host checks of the combine JIT launcher and its device destination selection."""

import unittest

from _host_test_helpers import ROOT, run_host_cpp


class CombineShmemHostTests(unittest.TestCase):
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
    void *buffer, *send_buffer, *peer_buffers;
    int rank, launches = 0;
    int compile(std::string kernel, std::string code) {
        name = kernel;
        source = code;
        return 1;
    }
    void launch(int, Config, void*, void*, void*, void*, void* recv, void* send,
                void*, void*, int current_rank, void* peers) {
        buffer = recv;
        send_buffer = send;
        peer_buffers = peers;
        rank = current_rank;
        ++launches;
    }
} recording_jit;
auto jit = &recording_jit;
''' + launcher + r'''
int main() {
    int recv, send, table, weights;
    for (bool direct : {false, true})
        for (bool shmem : {false, true})
            for (bool barrier : {false, true})
                for (bool weighted : {false, true}) {
                    auto peers = shmem ? &table : nullptr;
                    auto staging = direct ? nullptr : &send;
                    const int before = jit->launches;
                    try {
                        auto result = launch_combine(nullptr, weighted ? &weights : nullptr, nullptr,
                            nullptr, &recv, staging, nullptr, nullptr, 1, 4, 16, 3, 128,
                            194560, 64, 100, barrier, direct, peers);
                        assert(!shmem || direct);
                        assert(result == &recv && jit->buffer == &recv && jit->send_buffer == staging);
                        assert(jit->rank == 1 && jit->peer_buffers == peers && jit->launches == before + 1);
                        assert(jit->name == (shmem ? "combine_shmem" : direct ? "combine_mte" : "combine"));
                        auto specialization = std::format("combine_impl<4, 3, 128, 16, 64, 100, 194560, {}, {}, {}, {}>",
                                                          barrier, weighted, direct, shmem);
                        assert(jit->source.find(specialization) != std::string::npos);
                    } catch (const std::invalid_argument&) {
                        assert(shmem && !direct && before == jit->launches);
                    }
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
namespace layout {
struct CommonSignals { uint8_t* native[4]; };
uint8_t* get_sym_ptr(const CommonSignals& signals, uint8_t* local, int peer) {
    assert(local == signals.native[1]);
    return signals.native[peer];
}
}

template <bool kLowLatencyTransport, bool kShmemTransport, int kNumTopk>
void check_slot(int owner, layout::CommonSignals native, const uint64_t* peer_buffers,
                uint8_t* combine_recv_buffer, uint8_t* send_buffer, uint8_t* expected) {
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
    uint8_t native[4][512], shmem[4][512], send[512];
    layout::CommonSignals signals{};
    uint64_t table[4];
    for (int i = 0; i < 4; ++i) {
        signals.native[i] = native[i];
        table[i] = reinterpret_cast<uint64_t>(shmem[i]);
    }
    for (int owner = 0; owner < 4; ++owner) {
        // Token 1. K=3 selects top-k lane 2 (slot 9), K=6 selects rank 1 (slot 5).
        check_slot<true, true, 3>(owner, signals, table, shmem[1], nullptr, shmem[owner] + 9 * 32);
        check_slot<true, true, 6>(owner, signals, table, shmem[1], nullptr, shmem[owner] + 5 * 32);
        check_slot<true, false, 3>(owner, signals, nullptr, native[1], nullptr, native[owner] + 9 * 32);
        check_slot<false, false, 3>(owner, signals, nullptr, native[1], send,
                                    owner == 1 ? native[1] + 9 * 32 : send + 32);
    }
}
''')


if __name__ == '__main__':
    unittest.main()
