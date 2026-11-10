import unittest

from low_latency_reference import expert_entries, native_dispatch_payload_sources, native_dispatch_wire_routes


class NativeDispatchPayloadReferenceTests(unittest.TestCase):
    def test_empty_rank_padding_and_masked_routes(self):
        routes = [[], [[0, -1, 3], [-1, -1, -1]], [[2, 5, -1]], []]
        wire = native_dispatch_wire_routes(routes, 4, 3)
        self.assertTrue(all(len(rows) == 4 for rows in wire))
        self.assertEqual(wire[0], [[-1, -1, -1]] * 4)
        self.assertEqual(native_dispatch_payload_sources(routes, 8, 0, 4, 3), [[4], []])
        self.assertEqual(native_dispatch_payload_sources(routes, 8, 1, 4, 3), [[8], [4]])
        self.assertEqual(native_dispatch_payload_sources(routes, 8, 2, 4, 3), [[], [8]])
        self.assertEqual(native_dispatch_payload_sources(routes, 8, 3, 4, 3), [[], []])

    def test_duplicate_payloads_preserve_native_expansion_and_weights(self):
        routes = [[[0, 0, 2, -1], [3, 0, 3, 0]], [], [[1, 1, 2, 2]], []]
        capacity = 3
        wire = native_dispatch_wire_routes(routes, capacity, 4)
        self.assertEqual(wire[0][0], [0, -1, 2, -1])
        self.assertEqual(wire[0][1], [3, 0, -1, -1])
        self.assertEqual(wire[2][0], [1, -1, 2, -1])
        for rank in range(4):
            payloads = native_dispatch_payload_sources(routes, 8, rank, capacity, 4)
            native = expert_entries(routes, 8, 4, rank)
            for local_expert, native_rows in enumerate(native):
                source_ids = payloads[local_expert]
                self.assertEqual(len(source_ids), len(set(source_ids)))
                restored = [(source, token) for source, token in native_rows
                            if source * capacity + token in source_ids]
                self.assertEqual(restored, native_rows)
            # Native metadata owns one weight per original route, even when one
            # payload row serves repeated experts or multiple local experts.
            expected_weights = []
            restored_weights = []
            for source, rows in enumerate(routes):
                for token, selected in enumerate(rows):
                    for slot, expert in enumerate(selected):
                        if rank * 2 <= expert < rank * 2 + 2:
                            weight = (source + 1, token + 1, slot + 1)
                            expected_weights.append(weight)
                            if source * capacity + token in payloads[expert - rank * 2]:
                                restored_weights.append(weight)
            self.assertEqual(restored_weights, expected_weights)

    def test_all_empty_and_common_capacity_source_ids(self):
        self.assertEqual(native_dispatch_payload_sources([[], [], [], []], 8, 1, 8, 2), [[], []])
        routes = [[], [], [[3, -1]], []]
        self.assertEqual(native_dispatch_payload_sources(routes, 8, 1, 8, 2), [[], [16]])

    def test_capacity_and_topk_contract(self):
        with self.assertRaises(ValueError):
            native_dispatch_wire_routes([[[0], [1]]], 1, 1)
        with self.assertRaises(ValueError):
            native_dispatch_wire_routes([[[0, 1]]], 2, 1)


if __name__ == '__main__':
    unittest.main()
