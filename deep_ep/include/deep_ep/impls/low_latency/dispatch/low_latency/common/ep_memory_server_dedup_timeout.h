#pragma once

#include <cstdint>

#ifdef __CCE__
#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <utils/debug/asc_time.h>
#include <deep_ep/common/exception.hpp>
#define DISPATCH_DEDUP_TIMEOUT_INLINE __aicore__ __forceinline__
#else
#define DISPATCH_DEDUP_TIMEOUT_INLINE inline
#endif

namespace DispatchDedup {

DISPATCH_DEDUP_TIMEOUT_INLINE bool DispatchWaitExpired(int64_t timeout_cycles, uint64_t start, uint64_t now) {
    return timeout_cycles > 0 && now - start >= static_cast<uint64_t>(timeout_cycles);
}

#ifdef __CCE__
class DispatchWaitDeadline {
    int64_t timeout_cycles_;
    uint64_t start_;

public:
    __aicore__ __forceinline__ explicit DispatchWaitDeadline(int64_t timeout_cycles):
        timeout_cycles_(timeout_cycles), start_(timeout_cycles > 0 ? static_cast<uint64_t>(asc_get_system_cycle()) : 0) {}

    __aicore__ __forceinline__ void Check() const {
        if (timeout_cycles_ > 0 && DispatchWaitExpired(timeout_cycles_, start_, static_cast<uint64_t>(asc_get_system_cycle()))) {
            EP_DEVICE_ASSERT(false && "LL dispatch wait timed out");
            trap();
        }
    }
};
#endif

}  // namespace DispatchDedup

#undef DISPATCH_DEDUP_TIMEOUT_INLINE
