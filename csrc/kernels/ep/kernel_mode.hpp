#pragma once

namespace deep_ep {

enum class EPKernelMode { Native, DecodeURMA, DecodeMTE };

constexpr EPKernelMode get_ep_kernel_mode(bool decode, bool mte) {
    if (not decode)
        return EPKernelMode::Native;
    return mte ? EPKernelMode::DecodeMTE : EPKernelMode::DecodeURMA;
}

}  // namespace deep_ep
