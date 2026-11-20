#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace deep_ep::comm {

// Exchange capability and HCCL endpoint locations in communicator rank order together
// with the EP buffer's listen ports. Non-EP buffers exchange only their ports.
struct DecodeTopology {
    struct ServerLocation {
        uint32_t server_idx = UINT32_MAX;
        uint32_t super_pod_idx = UINT32_MAX;

        bool operator==(const ServerLocation&) const = default;
    };

    struct Record {
        int32_t port, vec_cores;
        ServerLocation location;
    };
    static_assert(sizeof(Record) == 4 * sizeof(uint32_t));
    int vec_cores;
    ServerLocation local_server;
    bool mte_enabled = false;
    std::vector<int> server_ids;

    explicit DecodeTopology(int cores): vec_cores(cores) {}

    template <typename Location>
    static ServerLocation get_server_location(const Location& loc) {
        using LocType = decltype(loc.locType);
        if constexpr (requires { LocType::ENDPOINT_LOC_TYPE_DEVICE; loc.device.serverIdx; loc.device.superPodIdx; }) {
            if (loc.locType == LocType::ENDPOINT_LOC_TYPE_DEVICE)
                return {loc.device.serverIdx, loc.device.superPodIdx};
        }
        return {};
    }

    template <typename Gather>
    std::vector<uint16_t> gather_ports(uint16_t port, Gather&& gather) {
        const auto peers = gather(Record{port, vec_cores, local_server});
        std::vector<uint16_t> ports;
        ports.reserve(peers.size());
        for (const auto& peer: peers)
            ports.push_back(static_cast<uint16_t>(peer.port));

        // All ranks must agree on MTE eligibility before entering the collective.
        // The default launch geometry uses the local vector core count on each rank.
        mte_enabled = vec_cores > 0 and std::all_of(peers.begin(), peers.end(),
            [&](const Record& peer) { return peer.vec_cores == vec_cores; });
        server_ids.clear();
        // Some HCCL versions leave device location fields at their 0xFF defaults.
        // Relay only if every rank has an identified server in the rank graph.
        if (mte_enabled and std::all_of(peers.begin(), peers.end(), [](const Record& peer) {
            return peer.location.server_idx != UINT32_MAX and peer.location.super_pod_idx != UINT32_MAX;
        })) {
            std::vector<ServerLocation> distinct;
            for (const auto& peer: peers) {
                const auto it = std::find(distinct.begin(), distinct.end(), peer.location);
                server_ids.push_back(static_cast<int>(it - distinct.begin()));
                if (it == distinct.end())
                    distinct.push_back(peer.location);
            }
            if (distinct.size() == 1 or distinct.size() == peers.size())
                server_ids.clear();
        }
        return ports;
    }
};

}  // namespace deep_ep::comm
