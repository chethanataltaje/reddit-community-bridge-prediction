import unittest
import pandas as pd
import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from person2.predictions.predictions import (
    extract_top_k_predictions,
    generate_top_k_dataframe,
    summarize_top_k_precision
)


class TestPerson2Predictions(unittest.TestCase):

    def test_descending_score_ordering_and_ranks(self):
        scores = [
            ("subA", "subB", 2.5),
            ("subC", "subD", 10.0),
            ("subE", "subF", 5.0)
        ]
        results = extract_top_k_predictions(scores, "Common Neighbors", k=3)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["source_subreddit"], "subC")
        self.assertEqual(results[0]["score"], 10.0)

        self.assertEqual(results[1]["rank"], 2)
        self.assertEqual(results[1]["source_subreddit"], "subE")
        self.assertEqual(results[1]["score"], 5.0)

        self.assertEqual(results[2]["rank"], 3)
        self.assertEqual(results[2]["source_subreddit"], "subA")
        self.assertEqual(results[2]["score"], 2.5)

    def test_duplicate_undirected_edges_handling(self):
        # (subA, subB) and (subB, subA) represent the exact same undirected edge
        scores = [
            ("subA", "subB", 8.0),
            ("subB", "subA", 8.0),  # Duplicate undirected pair
            ("subX", "subY", 3.0)
        ]
        results = extract_top_k_predictions(scores, "Jaccard", k=10)

        # Should deduplicate (subB, subA) and return only 2 unique pairs
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["source_subreddit"], "subA")
        self.assertEqual(results[0]["target_subreddit"], "subB")
        self.assertEqual(results[1]["source_subreddit"], "subX")
        self.assertEqual(results[1]["target_subreddit"], "subY")

    def test_stable_ranking(self):
        # Tied scores should preserve original order stably
        scores = [
            ("subA", "subB", 5.0),
            ("subC", "subD", 5.0),
            ("subE", "subF", 5.0)
        ]
        results = extract_top_k_predictions(scores, "Adamic-Adar", k=3)

        self.assertEqual(results[0]["source_subreddit"], "subA")
        self.assertEqual(results[1]["source_subreddit"], "subC")
        self.assertEqual(results[2]["source_subreddit"], "subE")

    def test_fewer_than_k_candidates(self):
        scores = [
            ("sub1", "sub2", 1.0),
            ("sub3", "sub4", 0.5)
        ]
        results = extract_top_k_predictions(scores, "Preferential Attachment", k=10)
        self.assertEqual(len(results), 2)

    def test_correct_algorithm_labels_and_k_extraction(self):
        scores_dict = {
            "CN": [("A", "B", 10.0), ("C", "D", 8.0)],
            "JC": [("X", "Y", 0.9), ("Z", "W", 0.7)]
        }
        df = generate_top_k_dataframe(scores_dict, k=1)
        self.assertEqual(len(df), 2)
        self.assertEqual(df[df["algorithm"] == "CN"].iloc[0]["source_subreddit"], "A")
        self.assertEqual(df[df["algorithm"] == "JC"].iloc[0]["source_subreddit"], "X")

    def test_ground_truth_annotation(self):
        scores = [
            ("subA", "subB", 5.0),  # Positive
            ("subX", "subY", 4.0)   # Negative
        ]
        test_pos = [("subA", "subB")]
        results = extract_top_k_predictions(scores, "CN", k=2, test_edges_pos=test_pos)

        self.assertEqual(results[0]["ground_truth"], 1)
        self.assertEqual(results[1]["ground_truth"], 0)


if __name__ == "__main__":
    unittest.main()
