#include <cassert>
#include <initializer_list>
#include <tuple>

#include "../../csrc/kernels/ep/dispatch_policy.hpp"

struct Config {
    int arch = 3510, ranks = 64, tokens = 1, capacity = 1;
    int hidden = 7168, topk = 8, experts = 128, vec_cores = 64, ub_bytes = 190 * 1024;
    bool fp8 = false, cpu_sync = false, cached = false;
};

static bool select(const Config& c) {
    return deep_ep::prefer_mte_dispatch(c.arch, c.ranks, c.tokens, c.capacity,
        c.hidden, c.topk, c.experts, c.vec_cores, c.ub_bytes, c.fp8, c.cpu_sync, c.cached);
}

int main() {
    static_assert(deep_ep::prefer_mte_dispatch(3510, 64, 32, 32, 7168, 8, 128, 64, 190 * 1024, false, false, false));
    static_assert(not deep_ep::prefer_mte_dispatch(3510, 64, 1, 33, 7168, 8, 128, 64, 190 * 1024, false, false, false));
    static_assert(deep_ep::prefer_mte_dispatch(3510, 64, 16, 16, 7168, 8, 128, 64, 190 * 1024, false, false, true));
    static_assert(not deep_ep::prefer_mte_dispatch(3510, 64, 1, 17, 7168, 8, 128, 64, 190 * 1024, false, false, true));
    const Config base;
    assert(select(base));
    for (const auto& [ranks, fresh_limit, cached_limit]: {std::tuple{8, 64, 64}, {16, 64, 64}, {32, 48, 32}, {64, 32, 16}}) {
        for (bool cached: {false, true}) {
            const int limit = cached ? cached_limit : fresh_limit;
            assert(deep_ep::dispatch_mte_capacity_limit(ranks, cached) == limit);
            for (int capacity: {1, limit - 1, limit, limit + 1, 256, 257}) {
                for (int tokens = 1; tokens <= capacity; ++ tokens) {
                    auto c = base;
                    c.ranks = ranks;
                    c.tokens = tokens;
                    c.capacity = capacity;
                    c.cached = cached;
                    assert(select(c) == (capacity <= limit));
                }
            }
            for (int tokens: {-1, 0, limit + 1, 256, 4096}) {
                auto c = base;
                c.ranks = ranks;
                c.tokens = tokens;
                c.capacity = limit;
                c.cached = cached;
                assert(not select(c));
            }
        }
        auto c = base;
        c.ranks = ranks;
        c.capacity = fresh_limit;
        assert(select(c));
        c.cached = true;
        assert(select(c) == (fresh_limit <= cached_limit));
        c.capacity = cached_limit;
        c.cached = false;
        c.cpu_sync = true;
        assert(not select(c));
        c.cached = true;
        c.cpu_sync = false;
        assert(select(c));
    }
    for (int capacity: {-1, 0, 257, 4096}) {
        auto c = base;
        c.capacity = capacity;
        assert(not select(c));
    }
    for (int ranks: {1, 2, 4, 7, 12, 128, 256}) {
        auto c = base;
        c.ranks = ranks;
        assert(not select(c));
    }
    assert(not select(Config{.arch = 3500}));
    assert(not select(Config{.hidden = 5120}));
    assert(not select(Config{.topk = 6}));
    assert(not select(Config{.experts = 256}));
    assert(not select(Config{.vec_cores = 32}));
    assert(not select(Config{.vec_cores = 128}));
    assert(not select(Config{.ub_bytes = 190 * 1024 - 1}));
    assert(select(Config{.ub_bytes = 190 * 1024 + 1}));
    assert(not select(Config{.fp8 = true}));
    assert(not select(Config{.cpu_sync = true}));
}
