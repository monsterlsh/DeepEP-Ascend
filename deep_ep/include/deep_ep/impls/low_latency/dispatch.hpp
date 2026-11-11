#pragma once

// Copyright (c) 2026, Lu Lu

#include <cstdint>

#include <kernel_operator.h>

#include <deep_ep/impls/low_latency/dispatch/low_latency/dispatch_server_dedup_kernel.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/dispatch_dedup_types.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h>

template <int = 0>
__global__ __vector__ void dispatch_server_dedup_kernel(
    GM_ADDR comm_args, GM_ADDR x, GM_ADDR expert_ids, GM_ADDR expert_scales,
    GM_ADDR expand_x_out, GM_ADDR dynamic_scales_out, GM_ADDR expert_token_nums_out, GM_ADDR send_counts_out,
    GM_ADDR token_type_out, GM_ADDR destination_index_out, GM_ADDR relay_read_index_out,
    GM_ADDR source_mask_out, GM_ADDR expert_scales_out, GM_ADDR tiling_data, int64_t /*reserved_magic*/,
    GM_ADDR input_scales, GM_ADDR native_source_ids = nullptr) {
    AscendC::InitSocState();

    const auto* tiling = reinterpret_cast<const __gm__ DispatchDedup::EpServerDedupTilingData*>(tiling_data);
    const bool enable_dedup = DispatchDedup::EpServerDedupEnabled(tiling->numMaxTokensPerRank);
    const auto input_mode = static_cast<DispatchDedup::EpServerDedupInputMode>(tiling->inputMode);
    const auto quant_mode = tiling->quantMode;
    const auto dtype = tiling->dtype;
    const bool output_e4m3 = tiling->expandXOutDtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3;
    AscendC::TPipe pipe;

#define DISPATCH_SERVER_DEDUP_RUN(XType, WireFp8, E4M3, Prequantized)                                                             \
    DispatchDedup::DispatchMemoryServerDedupKernel<XType, WireFp8, E4M3, Prequantized> kernel;                                    \
    kernel.Init(comm_args, x, expert_ids, expert_scales, expand_x_out, dynamic_scales_out, expert_token_nums_out,                 \
                send_counts_out, token_type_out, destination_index_out, relay_read_index_out, source_mask_out, expert_scales_out, \
                tiling->bs, tiling->h, tiling->topK, tiling->moeExpertNum, tiling->expertTokenNumsType,                           \
                &pipe, input_scales, enable_dedup, native_source_ids);                                                            \
    kernel.Process()

    if (input_mode == DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs and quant_mode == 4 and
        dtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3 and output_e4m3) {
        DISPATCH_SERVER_DEDUP_RUN(uint8_t, true, true, true);
    } else if (dtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16) {
        if (quant_mode == 0) {
            DISPATCH_SERVER_DEDUP_RUN(half, false, false, false);
        } else if (output_e4m3) {
            DISPATCH_SERVER_DEDUP_RUN(half, true, true, false);
        } else {
            DISPATCH_SERVER_DEDUP_RUN(half, true, false, false);
        }
    } else if (dtype == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16) {
        if (quant_mode == 0) {
            DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, false, false, false);
        } else if (output_e4m3) {
            DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, true, true, false);
        } else {
            DISPATCH_SERVER_DEDUP_RUN(bfloat16_t, true, false, false);
        }
    }
#undef DISPATCH_SERVER_DEDUP_RUN
}
