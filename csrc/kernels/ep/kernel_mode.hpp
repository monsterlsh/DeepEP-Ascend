#pragma once

#include <optional>
#include <utility>

namespace deep_ep {

enum class EPKernelMode { Native, DecodeURMA, DecodeMTE };

constexpr EPKernelMode get_ep_kernel_mode(bool decode, bool mte) {
    if (not decode)
        return EPKernelMode::Native;
    return mte ? EPKernelMode::DecodeMTE : EPKernelMode::DecodeURMA;
}

// A decode specialization can reject the selected UB geometry. Resolve it before
// submitting the collective. Native epilogues keep their original JIT timing.
static auto make_epilogue_launcher(EPKernelMode mode, auto prepare) {
    auto prepared = mode == EPKernelMode::Native ? std::nullopt : std::optional{prepare()};
    return [prepared = std::move(prepared), prepare = std::move(prepare)](auto... args) {
        if (prepared)
            (*prepared)(args...);
        else
            prepare()(args...);
    };
}

}  // namespace deep_ep
