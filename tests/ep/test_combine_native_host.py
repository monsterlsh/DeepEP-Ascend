"""Host checks of the combine JIT launcher and its device destination selection."""

import unittest

from _host_test_helpers import ROOT, run_host_cpp


class CombineNativeHostTests(unittest.TestCase):
    def test_kernel_specialization_and_runtime_arguments(self):
        # Compile through the public include; replace only the device/runtime dependencies.
        headers = {f'csrc/kernels/ep/{name}': (ROOT / 'csrc/kernels/ep' / name).read_text()
                   for name in ('api.hpp', 'dispatch.hpp', 'combine.hpp', 'kernel_mode.hpp')}
        buffer = (ROOT / 'csrc/buffers/ep.hpp').read_text()
        copy_args = buffer.split('launch_dispatch_copy_epilogue(', 1)[1].split(',', 3)
        copy_jetty_argument = copy_args[2].strip()
        headers.update({name: '#pragma once' for name in ('csrc/runtime/jit.hpp', 'deep_ep/layout/ep/token.hpp',
                                                          'deep_ep/common/exception.hpp', 'deep_ep/common/math.hpp')})
        run_host_cpp(r'''
#include <cassert>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <tuple>

#define EP_HOST_ASSERT(cond) do { if (!(cond)) throw std::invalid_argument(#cond); } while (false)
namespace deep_ep {
using sf_pack_t = uint32_t;
constexpr int kNumVectorRegisterBytes = 256;
namespace layout { struct TokenLayout { TokenLayout(int, int, int, bool=true) {} }; }
struct Config { int num_blocks, num_ubuf_bytes; };
struct RecordingJit {
    std::string name, source;
    void *buffer = nullptr, *send_buffer = nullptr, *scratch = nullptr, *copy_jetties = nullptr;
    const int* server_ids = nullptr;
    int rank = -1, launches = 0, argument_count = 0;
    bool reject_geometry = false;
    int compile(std::string kernel, std::string code) {
        if (reject_geometry) throw std::runtime_error("unsupported specialization");
        name = kernel;
        source = code;
        return 1;
    }
    template<class... Args>
    void launch(int, Config, Args... args) {
        auto values = std::tuple{args...};
        argument_count = sizeof...(Args);
        if constexpr (sizeof...(Args) == 9) {
            buffer = std::get<4>(values);
            send_buffer = std::get<5>(values);
            rank = std::get<8>(values);
        }
        if constexpr (sizeof...(Args) == 21) {
            server_ids = std::get<19>(values);
            scratch = std::get<20>(values);
        }
        if constexpr (sizeof...(Args) == 12)
            copy_jetties = std::get<2>(values);
        ++launches;
    }
} recording_jit;
auto jit = &recording_jit;
} // namespace deep_ep
#include "csrc/kernels/ep/api.hpp"

int main() {
    using namespace deep_ep;
    static_assert(get_ep_kernel_mode(false, true) == EPKernelMode::Native);
    static_assert(get_ep_kernel_mode(true, false) == EPKernelMode::DecodeURMA);
    static_assert(get_ep_kernel_mode(true, true) == EPKernelMode::DecodeMTE);
    int recv, send, weights, peers, scratch;
    // Old source calls, including omitted default arguments, still submit main's device ABI.
    launch_dispatch(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, 4, 5, 16,
                    8, 3, 1, false, false, 256, 0, 0, 0, 194560, 64, 100);
    assert(jit->argument_count == 19 && jit->name == "dispatch");
    assert(jit->source.find("#include <deep_ep/impls/ep/dispatch.hpp>") != std::string::npos);
    assert(jit->source.find("&dispatch_impl<4, 8, 3, 1, false, false, 64, 256, 0, 16, 194560, 100, false>")
           != std::string::npos);
    auto result = launch_combine(nullptr, &weights, nullptr, nullptr, &recv, &send, nullptr, nullptr,
                                 1, 4, 16, 3, 128, 194560, 64, 100);
    assert(jit->argument_count == 9 && jit->rank == 1 && result == &recv && jit->buffer == &recv);
    assert(jit->send_buffer == &send && jit->name == "combine");
    assert(jit->source.find("#include <deep_ep/impls/ep/combine.hpp>") != std::string::npos);
    assert(jit->source.find("&combine_impl<4, 3, 128, 16, 64, 100, 194560, false, true>") != std::string::npos);
    launch_combine_reduce_epilogue(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                   4, 5, 16, 8, 3, 128, 194560, 64, 100);
    assert(jit->source.find("&combine_reduce_epilogue_impl<4, 8, 3, 128, 16, 194560, 64, 100, true>")
           != std::string::npos);

    const auto native_launches = jit->launches;
    int submissions = 0;
    auto before_launch = [&] { ++submissions; };
    for (auto mode : {EPKernelMode::Native, EPKernelMode::DecodeURMA, EPKernelMode::DecodeMTE}) {
        jit->reject_geometry = true;
        try {
            launch_ep_combine(mode, before_launch, nullptr, nullptr, false, nullptr, nullptr, &recv, &send,
                              nullptr, nullptr, 1, 4, 16, 3, 128, 194560, 64, 100);
            assert(false);
        } catch (const std::runtime_error&) {
            assert(submissions == 0 && jit->launches == native_launches);
        }
    }
    jit->reject_geometry = false;
    for (bool direct : {false, true}) {
        auto mode = get_ep_kernel_mode(true, direct);
        for (bool relay : {false, true}) {
            if (!direct && relay) continue;
            launch_ep_dispatch(mode, before_launch, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, 4, 5, 16,
                               8, 3, 1, false, false, 256, 0, 0, 0, 194560, 64, 100, false,
                               relay ? &peers : nullptr, relay ? &scratch : nullptr);
            assert(jit->argument_count == 21);
            assert(jit->server_ids == (relay ? &peers : nullptr) && jit->scratch == (relay ? &scratch : nullptr));
            assert(jit->name == (relay ? "dispatch_mte_relay" : direct ? "dispatch_mte" : "dispatch"));
            assert(jit->source.find("#include <deep_ep/impls/ep/decode/dispatch.hpp>") != std::string::npos);
            auto suffix = std::format("100, false, {}, {}>", direct, relay);
            assert(jit->source.find(suffix) != std::string::npos);
        }
        for (bool barrier : {false, true})
            for (bool weighted : {false, true})
                for (bool fp16 : {false, true}) {
                    auto staging = direct ? nullptr : &send;
                    // A defined zero-row weight input may have a null address.
                    auto result = launch_ep_combine(mode, before_launch, nullptr, nullptr, weighted, nullptr,
                        nullptr, &recv, staging, nullptr, nullptr, 1, 4, 16, 3, 128,
                        194560, 64, 100, barrier, fp16);
                    assert(result == &recv && jit->buffer == &recv && jit->send_buffer == staging);
                    assert(jit->rank == 1 && submissions == jit->launches - native_launches);
                    assert(jit->name == (direct ? "combine_mte" : "combine"));
                    auto specialization = std::format("&deep_ep::decode::combine_impl<4, 3, 128, 16, 64, 100, 194560, {}, {}, {}, {}>",
                                                      barrier, weighted, direct, fp16 ? "half" : "bfloat16_t");
                    assert(jit->source.find(specialization) != std::string::npos);
                }
    }
    for (bool direct : {false, true}) {
        launch_ep_combine_reduce_epilogue(get_ep_kernel_mode(true, direct), nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, 4, 5, 16, 8, 3, 128, 194560, 64, 100, true, true);
        assert(jit->source.find("#include <deep_ep/impls/ep/decode/combine_reduce_epilogue.hpp>") != std::string::npos);
        assert(jit->source.find(std::format("100, true, {}, half>", direct)) != std::string::npos);
    }
    struct Comm { void* jetty_device_ptrs; } comm{&peers};
    struct Context { Comm* hccl_context; } ctx{&comm};
    const auto context = &ctx;
    for (bool low_latency_transport : {false, true})
        for (bool barrier : {false, true}) {
            launch_dispatch_copy_epilogue(nullptr, nullptr, ''' + copy_jetty_argument + r''',
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0, 4, 16, 8, 3, 1, 256, 0, 0, 0,
                194560, 64, 100, barrier);
            assert(jit->name == "dispatch_copy_epilogue");
            assert(jit->copy_jetties == (low_latency_transport ? nullptr : &peers));
        }
}
''', headers=headers)

    def test_local_and_remote_return_slots(self):
        device = (ROOT / 'deep_ep/include/deep_ep/impls/ep/decode/combine.hpp').read_text()
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

template <bool kLowLatencyTransport, int kNumTopk>
void check_slot(int owner, layout::CommonSignals native,
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
