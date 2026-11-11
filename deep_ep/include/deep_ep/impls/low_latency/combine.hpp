#pragma once

// Copyright (c) 2026, Lu Lu

#include <cstdint>

#include <kernel_operator.h>

#include <deep_ep/impls/low_latency/combine/low_latency/combine_server_dedup_kernel.h>
#include <deep_ep/impls/low_latency/combine/low_latency/ep_memory_server_dedup_combine_tiling.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/device/dispatch_server_dedup_magic.h>

template <int = 0>
__global__ __vector__ void combine_server_dedup_kernel(
    GM_ADDR comm_args, GM_ADDR expert_out, GM_ADDR destination_index, GM_ADDR send_counts, GM_ADDR token_type,
    GM_ADDR relay_read_index, GM_ADDR source_mask, GM_ADDR y_out, GM_ADDR tiling_data) {
    const auto* args = reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(comm_args);
    AscendC::InitSocState();
    uint32_t magic;
    {
        AscendC::TPipe pipe;
        magic = DispatchDedup::RotateDispatchServerDedupMagic(args, AscendC::GetBlockIdx(), &pipe);
    }

    const auto* tiling = reinterpret_cast<const volatile __gm__ CombineDedup::EpServerDedupCombineTilingData*>(tiling_data);
    const bool enable_dedup = DispatchDedup::EpServerDedupEnabled(tiling->numMaxTokensPerRank);
    CombineDedup::CombineMemoryServerDedupKernel kernel;
    kernel.Init(comm_args, expert_out, destination_index, send_counts, token_type, relay_read_index, source_mask, y_out,
                tiling->bs, tiling->h, tiling->topK, tiling->moeExpertNum, tiling->dtype, 0, magic, tiling->recvUbBytes, enable_dedup);
    kernel.Process();
}
