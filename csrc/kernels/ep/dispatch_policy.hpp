#pragma once

namespace deep_ep {

constexpr int dispatch_mte_capacity_limit(int num_ranks, bool cached_mode) {
    switch (num_ranks) {
        case 8:
        case 16: return 64;
        case 32: return cached_mode ? 32 : 48;
        case 64: return cached_mode ? 16 : 32;
        default: return 0;
    }
}

constexpr bool prefer_mte_dispatch(int arch, int num_ranks,
                                   int num_tokens, int num_max_tokens_per_rank,
                                   int hidden, int num_topk, int num_experts,
                                   int num_vec_cores, int num_ub_bytes,
                                   bool use_fp8, bool do_cpu_sync, bool cached_mode) {
    return arch == 3510 and
           hidden == 7168 and num_topk == 8 and num_experts == 128 and
           num_vec_cores == 64 and num_ub_bytes >= 190 * 1024 and
           not use_fp8 and not do_cpu_sync and
           num_tokens > 0 and num_tokens <= num_max_tokens_per_rank and
           num_max_tokens_per_rank <= dispatch_mte_capacity_limit(num_ranks, cached_mode) and
           num_max_tokens_per_rank <= 256;
}

}  // namespace deep_ep
