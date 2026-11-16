#pragma once

#include <c_api/asc_simd.h>
#include <c_api/sync/sync.h>
#include <c_api/vector_datamove/vector_datamove.h>

#include <deep_ep/common/exception.hpp>
#include <deep_ep/common/math.hpp>

#include <kernel_operator.h>

namespace deep_ep {

template <int kHidden, int kNumTopk, int kNumStrideElems, int kNumSlots = 0>
__simd_vf__ __forceinline__ void combine_reduce_vf(__ubuf__ bfloat16_t* base_ptr, __ubuf__ bfloat16_t* output_ptr,
                                                    const int count, const int slot_start_idx = 0) {
    static constexpr int kNumElemsPerFP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    EP_STATIC_ASSERT(kHidden % (kNumElemsPerFP32 * 2) == 0);
    EP_STATIC_ASSERT(kNumTopk >= 1);

    vector_bfloat16_t bf16_zero;
    asc_duplicate_scalar(bf16_zero, 0);

    if (count == 0) {
        #pragma unroll 1
        for (int offset = 0; offset < kHidden; offset += kNumElemsPerFP32 * 2)
            asc_storealign(output_ptr + offset, bf16_zero, asc_create_mask_b16(PAT_ALL));
        return;
    } else if (count == 1) {
        __ubuf__ bfloat16_t* ub_input_ptr = base_ptr + slot_start_idx * kNumStrideElems;
        #pragma unroll 1
        for (int offset = 0; offset < kHidden; offset += kNumElemsPerFP32 * 2) {
            vector_bfloat16_t bf16_reg;
            asc_loadalign_postupdate(bf16_reg, ub_input_ptr, kNumElemsPerFP32 * 2);
            asc_storealign(output_ptr + offset, bf16_reg, asc_create_mask_b16(PAT_ALL));
        }
        return;
    }

    if (count == 2 and kNumStrideElems == kHidden) {
        const auto second_slot_idx = kNumSlots != 0 and slot_start_idx + 1 == kNumSlots ? 0 : slot_start_idx + 1;
        auto first_ptr = base_ptr + slot_start_idx * kNumStrideElems;
        auto second_ptr = base_ptr + second_slot_idx * kNumStrideElems;
        const auto mask = asc_create_mask_b16(PAT_ALL);
        #pragma unroll 2
        for (int offset = 0; offset < kHidden; offset += kNumElemsPerFP32 * 2) {
            vector_bfloat16_t first, second;
            asc_loadalign_postupdate(first, first_ptr, kNumElemsPerFP32 * 2);
            asc_loadalign_postupdate(second, second_ptr, kNumElemsPerFP32 * 2);
            asc_add(first, first, second, mask);
            asc_storealign(output_ptr + offset, first, mask);
        }
        return;
    }

    vector_bool u32_mask = asc_create_mask_b32(PAT_ALL);
    const auto num_wrap_count = kNumSlots == 0 ? count : math::min(kNumSlots - slot_start_idx, count);
    #pragma unroll 1
    for (int offset = 0; offset < kHidden; offset += kNumElemsPerFP32 * 2) {
        vector_bfloat16_t bf16_regs[2];
        vector_float acc_regs[2];
        __ubuf__ bfloat16_t* ptr = base_ptr + slot_start_idx * kNumStrideElems + offset;
        asc_loadalign_postupdate(bf16_regs[0], ptr, kNumStrideElems);
        asc_intlv(reinterpret_cast<vector_bfloat16_t&>(acc_regs[0]), reinterpret_cast<vector_bfloat16_t&>(acc_regs[1]), bf16_zero,
                  bf16_regs[0]);
        #pragma unroll 1
        for (int i = 1; i < num_wrap_count; ++ i) {
            vector_float fp32_regs[2];
            asc_loadalign_postupdate(bf16_regs[0], ptr, kNumStrideElems);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(fp32_regs[0]), reinterpret_cast<vector_bfloat16_t&>(fp32_regs[1]), bf16_zero,
                      bf16_regs[0]);
            asc_add(acc_regs[0], acc_regs[0], fp32_regs[0], u32_mask);
            asc_add(acc_regs[1], acc_regs[1], fp32_regs[1], u32_mask);
        }
        if constexpr (kNumSlots != 0) {
            ptr = base_ptr + offset;
            #pragma unroll 1
            for (int i = num_wrap_count; i < count; ++ i) {
                vector_float fp32_regs[2];
                asc_loadalign_postupdate(bf16_regs[0], ptr, kNumStrideElems);
                asc_intlv(reinterpret_cast<vector_bfloat16_t&>(fp32_regs[0]), reinterpret_cast<vector_bfloat16_t&>(fp32_regs[1]),
                          bf16_zero, bf16_regs[0]);
                asc_add(acc_regs[0], acc_regs[0], fp32_regs[0], u32_mask);
                asc_add(acc_regs[1], acc_regs[1], fp32_regs[1], u32_mask);
            }
        }

        asc_float2bfloat16_rn(bf16_regs[0], acc_regs[0], u32_mask);
        asc_float2bfloat16_rn(bf16_regs[1], acc_regs[1], u32_mask);
        asc_deintlv(bf16_regs[0], bf16_regs[1], bf16_regs[0], bf16_regs[1]);
        asc_storealign(output_ptr + offset, bf16_regs[0], asc_create_mask_b16(PAT_ALL));
    }
}

template <int kHidden, int kNumTopk, int kNumStrideElems, int kNumSlots>
__simd_vf__ __forceinline__ void combine_reduce_vf(__ubuf__ half* base_ptr, __ubuf__ half* output_ptr,
                                                   const int count, const int slot_start_idx) {
    constexpr int lanes = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    EP_STATIC_ASSERT(kHidden % (lanes * 2) == 0);
    if (count <= 1) {
        vector_half value;
        asc_duplicate_scalar(value, static_cast<half>(0));
        for (int offset = 0; offset < kHidden; offset += lanes * 2) {
            if (count == 1)
                asc_loadalign(value, base_ptr + slot_start_idx * kNumStrideElems + offset);
            asc_storealign(output_ptr + offset, value, asc_create_mask_b16(PAT_ALL));
        }
        return;
    }

    // Accumulate the selected expert rows in FP32, then round once on store.
    const auto mask = asc_create_mask_b32(PAT_ALL);
    for (int offset = 0; offset < kHidden; offset += lanes) {
        int slot = slot_start_idx;
        vector_float sum;
        vector_half value;
        asc_loadalign_unpack(value, base_ptr + slot * kNumStrideElems + offset);
        asc_half2float(sum, value, mask, ASC_POSITION_EVEN);
        for (int i = 1; i < count; ++i) {
            slot = slot + 1 == kNumSlots ? 0 : slot + 1;
            vector_float contribution;
            asc_loadalign_unpack(value, base_ptr + slot * kNumStrideElems + offset);
            asc_half2float(contribution, value, mask, ASC_POSITION_EVEN);
            asc_add(sum, sum, contribution, mask);
        }
        asc_float2half_rn(value, sum, mask, ASC_POSITION_EVEN);
        asc_storealign_pack(reinterpret_cast<__ubuf__ int32_t*>(output_ptr + offset),
                            reinterpret_cast<vector_int32_t&>(value), mask);
    }
}

template <int kNumTopk, int kNumTokenElems, int kNumSlots, int kNumHiddenElems>
__simd_vf__ __forceinline__ void restore_combine_weights(
    __ubuf__ uint32_t* ub_tokens, __ubuf__ uint32_t* ub_output,
    const int first_token_idx, const int num_tokens) {
    // A weight row is padded to 32 bytes. A full 256-byte vector load can
    // extend past the final UB ring slot at higher hidden/top-k geometries.
    constexpr int kBlockLanes = kNumUbAlignmentBytes / static_cast<int>(sizeof(uint32_t));
    for (int lane = 0; lane < kNumTopk; lane += kBlockLanes) {
        uint32_t remaining = math::min(kBlockLanes, kNumTopk - lane);
        const auto mask = asc_update_mask_b32(remaining);
        vector_uint32_t result;
        asc_duplicate_scalar(result, static_cast<uint32_t>(0), mask);
        int slot = first_token_idx;
        #pragma unroll 1
        for (int i = 0; i < num_tokens; ++i) {
            vector_uint32_t weights;
            auto input = ub_tokens + (slot * kNumTokenElems + kNumHiddenElems) / 2 + lane;
            asc_loadalign_brc_v2(weights, input);
            asc_or(result, result, weights, mask);
            slot = slot + 1 == kNumSlots ? 0 : slot + 1;
        }
        asc_storealign(ub_output + lane, result, mask);
    }
}

template <uint32_t kNumHiddenElems, uint32_t kNumTopk, uint32_t kNumMutex = 32, uint32_t kMutexOffset = 0,
          uint32_t kNumExtraBytes = 0, typename element_t = bfloat16_t>
struct RingReducer {
    static constexpr uint32_t kNumHiddenBytes = kNumHiddenElems * sizeof(element_t);
    static constexpr uint32_t kNumTokenElems =
        kNumHiddenElems + math::ceil_div<uint32_t>(kNumExtraBytes, sizeof(element_t));

    struct Buffer {
        element_t buf[kNumMutex][kNumTokenElems];
    };

    __ubuf__ Buffer* ub;
    uint32_t slot_idx;

    __aicore__ __forceinline__ RingReducer(__ubuf__ Buffer* ub): ub(ub), slot_idx(0) {}

    __aicore__ __forceinline__ void load_next_token(__gm__ element_t* ptr,
                                                    asc_load_l2_cache_mode mode = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                                                    const uint32_t num_extra_bytes = 0) {
        asc_lock(PIPE_MTE2, kMutexOffset + slot_idx);
        asc_copy_gm2ub_align(ub->buf[slot_idx], ptr, 1, kNumHiddenBytes + num_extra_bytes, 0, 0, true, mode, 0, 0);
        asc_unlock(PIPE_MTE2, kMutexOffset + slot_idx);
        slot_idx = slot_idx + 1 == kNumMutex ? 0 : slot_idx + 1;
    }

    __aicore__ __forceinline__ void load_first_2_token(__gm__ element_t* base, int idx_0, int idx_1,
                                                       asc_load_l2_cache_mode mode = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM) {
        slot_idx = slot_idx == kNumMutex - 1 ? 0 : slot_idx;
        const auto min_idx = idx_0 < idx_1 ? idx_0 : idx_1;
        const auto stride = idx_0 < idx_1 ? idx_1 - idx_0 : idx_0 - idx_1;
        asc_lock(PIPE_MTE2, kMutexOffset + slot_idx);
        asc_lock(PIPE_MTE2, kMutexOffset + slot_idx + 1);
        asc_copy_gm2ub_align(ub->buf[slot_idx], base + static_cast<int64_t>(min_idx) * kNumTokenElems,
                            2, kNumHiddenBytes + kNumExtraBytes, 0, 0, true, mode,
                            static_cast<uint64_t>(stride) * (kNumHiddenBytes + kNumExtraBytes),
                            kNumHiddenBytes + kNumExtraBytes);
        asc_unlock(PIPE_MTE2, kMutexOffset + slot_idx);
        asc_unlock(PIPE_MTE2, kMutexOffset + slot_idx + 1);
        slot_idx = slot_idx + 2 == kNumMutex ? 0 : slot_idx + 2;
    }

    __aicore__ __forceinline__ void store_last_token(__gm__ element_t* output_addr,
                                                     asc_store_l2_cache_mode mode = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                                                     __gm__ uint8_t* extra_output_addr = nullptr,
                                                     const uint32_t num_extra_output_bytes = kNumExtraBytes) {
        const auto output_slot_idx = slot_idx == 0 ? kNumMutex - 1 : slot_idx - 1;
        asc_lock(PIPE_MTE3, kMutexOffset + output_slot_idx);
        asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(output_addr),
                            reinterpret_cast<__ubuf__ uint8_t*>(ub->buf[output_slot_idx]),
                            1, kNumHiddenBytes, mode, 0, 0);
        if constexpr (kNumExtraBytes > 0) {
            if (extra_output_addr != nullptr) {
                asc_copy_ub2gm_align(extra_output_addr,
                    reinterpret_cast<__ubuf__ uint8_t*>(ub->buf[output_slot_idx]) + kNumHiddenBytes,
                    num_extra_output_bytes);
            }
        }
        asc_unlock(PIPE_MTE3, kMutexOffset + output_slot_idx);
    }

    __aicore__ __forceinline__ void reduce(const uint32_t num_slots) {
        const auto start_slot_idx = slot_idx < num_slots ? kNumMutex - (num_slots - slot_idx) : slot_idx - num_slots;
        const auto output_slot_idx = num_slots > 0 ? (slot_idx == 0 ? kNumMutex - 1 : slot_idx - 1) : slot_idx;
        const auto num_locked_slots = math::max(1U, num_slots);
        for (uint32_t i = 0, input_slot_idx = start_slot_idx; i < num_locked_slots; ++ i) {
            asc_lock(PIPE_V, kMutexOffset + input_slot_idx);
            input_slot_idx = input_slot_idx + 1 == kNumMutex ? 0 : input_slot_idx + 1;
        }
        combine_reduce_vf<kNumHiddenElems, kNumTopk, kNumTokenElems, kNumMutex>(
            ub->buf[0], ub->buf[output_slot_idx], num_slots, start_slot_idx);
        for (uint32_t i = 0, input_slot_idx = start_slot_idx; i < num_locked_slots; ++ i) {
            asc_unlock(PIPE_V, kMutexOffset + input_slot_idx);
            input_slot_idx = input_slot_idx + 1 == kNumMutex ? 0 : input_slot_idx + 1;
        }
        if (num_slots == 0)
            slot_idx = slot_idx + 1 == kNumMutex ? 0 : slot_idx + 1;
    }
};

}  // namespace deep_ep
