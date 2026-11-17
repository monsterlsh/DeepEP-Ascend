#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace deep_ep::comm {

// AscendCL stream waits accept a signed 32-bit timeout in milliseconds.
constexpr int32_t acl_timeout_ms(int timeout_secs) {
    const int64_t millis = int64_t{timeout_secs} * 1000;
    return static_cast<int32_t>(std::min<int64_t>(millis, std::numeric_limits<int32_t>::max()));
}

}  // namespace deep_ep::comm
