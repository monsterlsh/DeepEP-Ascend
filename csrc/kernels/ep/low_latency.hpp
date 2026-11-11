#pragma once

#include <deep_ep/impls/low_latency/dispatch/low_latency/launch.hpp>
#include <deep_ep/impls/low_latency/combine/low_latency/launch.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static void launch_dispatch_low_latency(const deepep::EpServerDedupKernelArgs& args) {
    const auto kernel = jit->compile("dispatch_low_latency", R"(
#include <deep_ep/impls/low_latency/dispatch.hpp>
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&dispatch_server_dedup_kernel<>);
}
)");
    jit->launch(
        kernel, {.num_blocks = 64, .num_ubuf_bytes = 0},
        args.commArgs, args.x, args.expertIds, args.expertScales, args.expandXOut, args.dynamicScalesOut,
        args.expertTokenNumsOut, args.sendCountsOut, args.tokenTypeOut, args.destinationIndexOut, args.relayReadIndexOut,
        args.sourceMaskOut, args.expertScalesOut, args.tilingData, args.reservedMagic, args.inputScales, args.nativeSourceIds);
}

static void launch_combine_low_latency(const deepep::EpServerDedupCombineKernelArgs& args) {
    const auto kernel = jit->compile("combine_low_latency", R"(
#include <deep_ep/impls/low_latency/combine.hpp>
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&combine_server_dedup_kernel<>);
}
)");
    jit->launch(
        kernel, {.num_blocks = 64, .num_ubuf_bytes = 0},
        args.commArgs, args.expertOut, args.destinationIndex, args.sendCounts, args.tokenType,
        args.relayReadIndex, args.sourceMask, args.yOut, args.tilingData);
}

}  // namespace deep_ep
