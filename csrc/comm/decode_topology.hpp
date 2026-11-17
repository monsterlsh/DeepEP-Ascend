#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace deep_ep::comm {

// Exchange capability and optional server IDs in communicator rank order together
// with the EP buffer's listen ports. Non-EP buffers exchange only their ports.
struct DecodeTopology {
    struct Record {
        int32_t port, server_id, vec_cores;
    };
    static_assert(sizeof(Record) == 3 * sizeof(int32_t));
    int vec_cores;
    bool mte_enabled = false;
    std::vector<int> server_ids;

    explicit DecodeTopology(int cores): vec_cores(cores) {}

    static int local_server_id() {
        if (const char* value = std::getenv("EP_DECODE_SERVER_ID"); value and *value) {
            const auto end = value + std::strlen(value);
            int parsed;
            const auto result = std::from_chars(value, end, parsed);
            if (result.ec == std::errc{} and result.ptr == end and parsed >= 0)
                return parsed;
        }
        return -1;
    }

    std::vector<uint16_t> gather_ports(uint16_t port, auto&& gather) {
        const auto peers = gather(Record{port, local_server_id(), vec_cores});
        std::vector<uint16_t> ports;
        ports.reserve(peers.size());
        for (const auto& peer: peers)
            ports.push_back(static_cast<uint16_t>(peer.port));

        // Public defaults use the local core count; heterogeneous ranks must agree
        // to use URMA before any of them can enter an MTE kernel.
        mte_enabled = vec_cores > 0 and std::all_of(peers.begin(), peers.end(),
            [&](const auto& p) { return p.vec_cores == vec_cores; });
        if (mte_enabled and std::all_of(peers.begin(), peers.end(), [](const auto& p) { return p.server_id >= 0; })) {
            for (const auto& p: peers)
                server_ids.push_back(p.server_id);
            auto distinct = server_ids;
            std::sort(distinct.begin(), distinct.end());
            const auto end = std::unique(distinct.begin(), distinct.end());
            if (end == distinct.begin() + 1 or end == distinct.end())
                server_ids.clear();
        }
        return ports;
    }
};

}  // namespace deep_ep::comm
