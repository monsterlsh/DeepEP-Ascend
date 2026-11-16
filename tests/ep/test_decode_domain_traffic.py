import unittest

from decode_domain_traffic import analyze, count_rows


class DomainTrafficTests(unittest.TestCase):
    def test_fanout_masks_and_intra_server_traffic(self):
        routes = [[[1, 2, 3, 4, 6, -1], [-1]], [], [], [[4, 0, 1, 7]]]
        count = count_rows(routes, ['a', 'b', 'a', 'b'], 8, 2)
        self.assertEqual(count, dict(tokens=3, direct_cross_server_rows=4, relay_cross_server_rows=2,
                                     direct_within_server_rows=1, dispatch_relay_within_server_rows=3,
                                     combine_relay_within_server_rows=5))
        self.assertEqual(count_rows(routes, ['a'] * 4, 8, 2)['direct_cross_server_rows'], 0)
        self.assertEqual(count_rows(routes, list(range(4)), 8, 2)['relay_cross_server_rows'], 5)

    def test_actual_benchmark_load_and_hidden_bytes(self):
        config = dict(world=4, num_topk=8, num_experts=8, rank_load='decode', hidden=512, bs=[2, 4])
        rows = analyze(config, ['a', 'b', 'a', 'b'])['rows']
        for row, tokens in zip(rows, [5, 10], strict=True):
            self.assertEqual(row['tokens'], tokens)
            self.assertEqual(row['direct_cross_server_hidden_bytes'], 2 * tokens * 1024)
            self.assertEqual(row['relay_cross_server_hidden_bytes'], tokens * 1024)
        with self.assertRaisesRegex(ValueError, 'one server label'):
            analyze(config, ['a', 'b'])
        with self.assertRaisesRegex(ValueError, 'no server IDs'):
            analyze(config)
        config['server_ids'] = [2, 5, 2, 5]
        self.assertEqual(analyze(config)['mapping_source'], 'benchmark_configuration')
        alternate = analyze(config, [0, 0, 0, 0])
        self.assertEqual(alternate['mapping_source'], 'counterfactual_override')
        self.assertEqual(alternate['rows'][0]['direct_cross_server_rows'], 0)
