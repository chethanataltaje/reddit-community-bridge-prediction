import unittest
import networkx as nx
import pandas as pd
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.community_detection import (
    detect_communities,
    generate_community_mapping_dataframe,
    get_community_statistics,
    generate_community_summary_dataframe
)

from src.bridge_analysis import (
    annotate_community_bridges,
    generate_bridge_summary
)


class TestCommunity(unittest.TestCase):

    def setUp(self):
        # Create a small synthetic graph with 2 clear cliques (communities)
        self.G = nx.Graph()
        self.G.add_edges_from([
            ("A", "B"), ("B", "C"), ("A", "C"),
            ("X", "Y"), ("Y", "Z"), ("X", "Z"),
            ("C", "X")
        ])

    def test_every_node_receives_exactly_one_community_id(self):
        comm_map = detect_communities(self.G, seed=42)
        nodes = list(self.G.nodes())

        self.assertEqual(len(comm_map), len(nodes))
        for n in nodes:
            self.assertIn(str(n), comm_map)
            self.assertIsInstance(comm_map[str(n)], int)

    def test_community_ids_are_valid(self):
        comm_map = detect_communities(self.G, seed=42)
        for node, comm_id in comm_map.items():
            self.assertGreaterEqual(comm_id, 0)

    def test_same_and_different_community_edges(self):
        comm_map = {"A": 0, "B": 0, "X": 1, "Y": 1}

        df_preds = pd.DataFrame([
            {"algorithm": "CN", "rank": 1, "source_subreddit": "A", "target_subreddit": "B", "score": 5.0, "ground_truth": 1},
            {"algorithm": "CN", "rank": 2, "source_subreddit": "A", "target_subreddit": "X", "score": 2.0, "ground_truth": 1}
        ])

        df_ann = annotate_community_bridges(df_preds, comm_map)

        row_same = df_ann[df_ann["source_subreddit"] == "A"][df_ann["target_subreddit"] == "B"].iloc[0]
        self.assertFalse(row_same["cross_community"])
        self.assertEqual(row_same["source_community"], 0)
        self.assertEqual(row_same["target_community"], 0)

        row_cross = df_ann[df_ann["source_subreddit"] == "A"][df_ann["target_subreddit"] == "X"].iloc[0]
        self.assertTrue(row_cross["cross_community"])
        self.assertEqual(row_cross["source_community"], 0)
        self.assertEqual(row_cross["target_community"], 1)

    def test_missing_community_node_handled_safely(self):
        comm_map = {"A": 0, "B": 0}

        df_preds = pd.DataFrame([
            {"algorithm": "JC", "rank": 1, "source_subreddit": "A", "target_subreddit": "UNKNOWN", "score": 0.5, "ground_truth": 0}
        ])

        df_ann = annotate_community_bridges(df_preds, comm_map)
        row = df_ann.iloc[0]

        self.assertEqual(row["source_community"], 0)
        self.assertEqual(row["target_community"], -1)
        self.assertTrue(row["cross_community"])

    def test_bridge_counts_and_cross_community_proportion(self):
        comm_map = {"A": 0, "B": 0, "C": 0, "X": 1, "Y": 1, "Z": 1}

        df_preds = pd.DataFrame([
            {"algorithm": "TestAlgo", "k": 4, "rank": 1, "source_subreddit": "A", "target_subreddit": "B", "score": 4.0},
            {"algorithm": "TestAlgo", "k": 4, "rank": 2, "source_subreddit": "A", "target_subreddit": "X", "score": 3.0},
            {"algorithm": "TestAlgo", "k": 4, "rank": 3, "source_subreddit": "B", "target_subreddit": "Y", "score": 2.0},
            {"algorithm": "TestAlgo", "k": 4, "rank": 4, "source_subreddit": "C", "target_subreddit": "Z", "score": 1.0}
        ])

        df_ann = annotate_community_bridges(df_preds, comm_map)
        df_summary = generate_bridge_summary(df_ann)

        self.assertEqual(len(df_summary), 1)
        row = df_summary.iloc[0]

        self.assertEqual(row["algorithm"], "TestAlgo")
        self.assertEqual(row["k"], 4)
        self.assertEqual(row["total_candidates"], 4)
        self.assertEqual(row["cross_community_count"], 3)
        self.assertEqual(row["within_community_count"], 1)
        self.assertAlmostEqual(row["cross_community_proportion"], 0.75)

    def test_top_k_bridge_summary_generation(self):
        comm_map = {"A": 0, "B": 1}
        df_preds = pd.DataFrame([
            {"algorithm": "CN", "k": 10, "rank": 1, "source_subreddit": "A", "target_subreddit": "B", "score": 5.0},
            {"algorithm": "CN", "k": 20, "rank": 1, "source_subreddit": "A", "target_subreddit": "B", "score": 5.0}
        ])
        df_ann = annotate_community_bridges(df_preds, comm_map)
        df_sum = generate_bridge_summary(df_ann)
        self.assertEqual(len(df_sum), 2)


if __name__ == "__main__":
    unittest.main()
