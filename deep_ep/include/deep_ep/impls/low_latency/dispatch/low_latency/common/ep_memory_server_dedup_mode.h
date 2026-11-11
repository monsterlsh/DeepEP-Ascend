#pragma once

// Copyright (c) 2026, Lu Lu

#include <cstdint>

namespace DispatchDedup {

// Select the execution path from the per-rank token capacity.
constexpr uint32_t kEpServerDedupDedupThreshold = 32U;
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#endif
constexpr bool EpServerDedupEnabled(uint32_t numMaxTokensPerRank) {
    return numMaxTokensPerRank > kEpServerDedupDedupThreshold;
}

}  // namespace DispatchDedup
