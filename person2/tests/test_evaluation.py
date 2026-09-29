import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from person2.evaluation.evaluation import (
    extract_labels_and_scores,
    compute_roc_auc,
    compute_classification_metrics,
    compute_precision_at_k,
    compute_recall_at_k,
    generate_comparison_table
)


class TestPerson2Evaluation(unittest.TestCase):

    def test_label_alignment(self):
        # Synthetic positive edges
        test_edges_pos = [("A", "B"), ("C", "D")]
        
        # Synthetic scores list with reverse ordering on (B, A) to test bidirectional tuple matching
        scores_list = [
            ("B", "A", 0.9),  # Ground truth positive (1)
            ("X", "Y", 0.1),  # Ground truth negative (0)
            ("C", "D", 0.8),  # Ground truth positive (1)
            ("D", "Z", 0.3)   # Ground truth negative (0)
        ]

        y_true, y_score = extract_labels_and_scores(scores_list, test_edges_pos)
        
        expected_y_true = np.array([1, 0, 1, 0])
        expected_y_score = np.array([0.9, 0.1, 0.8, 0.3])

        np.testing.assert_array_equal(y_true, expected_y_true)
        np.testing.assert_array_almost_equal(y_score, expected_y_score)

    def test_precision_at_k(self):
        # y_true has positives at scores: 0.9 (pos), 0.8 (pos), 0.7 (neg), 0.6 (neg), 0.5 (pos)
        y_true = np.array([1, 1, 0, 0, 1])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        # Top-2 scores: 0.9, 0.8 -> both positive -> Precision@2 = 2/2 = 1.0
        p2 = compute_precision_at_k(y_true, y_score, k=2)
        self.assertAlmostEqual(p2, 1.0)

        # Top-4 scores: 0.9, 0.8, 0.7, 0.6 -> 2 positives out of 4 -> Precision@4 = 2/4 = 0.5
        p4 = compute_precision_at_k(y_true, y_score, k=4)
        self.assertAlmostEqual(p4, 0.5)

    def test_recall_at_k(self):
        y_true = np.array([1, 1, 0, 0, 1])  # Total positives = 3
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        # Top-2 scores: 0.9, 0.8 -> 2 positives out of 3 total -> Recall@2 = 2/3
        r2 = compute_recall_at_k(y_true, y_score, k=2)
        self.assertAlmostEqual(r2, 2.0 / 3.0)

        # Top-5 scores: all 3 positives retrieved -> Recall@5 = 3/3 = 1.0
        r5 = compute_recall_at_k(y_true, y_score, k=5)
        self.assertAlmostEqual(r5, 1.0)

    def test_classification_metrics_recall_f1(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.3, 0.8, 0.1])

        metrics = compute_classification_metrics(y_true, y_score, threshold=0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_roc_auc(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        auc = compute_roc_auc(y_true, y_score)
        self.assertAlmostEqual(auc, 1.0)

    def test_empty_input_handling(self):
        y_true_empty = np.array([], dtype=int)
        y_score_empty = np.array([], dtype=float)

        self.assertEqual(compute_precision_at_k(y_true_empty, y_score_empty, k=10), 0.0)
        self.assertEqual(compute_recall_at_k(y_true_empty, y_score_empty, k=10), 0.0)
        self.assertEqual(compute_roc_auc(y_true_empty, y_score_empty), 0.0)
        metrics = compute_classification_metrics(y_true_empty, y_score_empty)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

        y_t, y_s = extract_labels_and_scores([], [])
        self.assertEqual(len(y_t), 0)
        self.assertEqual(len(y_s), 0)

    def test_k_larger_than_candidate_count(self):
        y_true = np.array([1, 0, 1])
        y_score = np.array([0.9, 0.8, 0.7])

        p10 = compute_precision_at_k(y_true, y_score, k=10)
        self.assertAlmostEqual(p10, 0.2)

        r10 = compute_recall_at_k(y_true, y_score, k=10)
        self.assertAlmostEqual(r10, 1.0)

    def test_comparison_table_generation(self):
        results_dict = {
            "Common Neighbors": {
                "precision@10": 0.8,
                "precision@20": 0.7,
                "recall@10": 0.08,
                "recall@20": 0.14,
                "recall@0.5": 0.5,
                "f1@0.5": 0.6,
                "roc_auc": 0.85
            }
        }
        df = generate_comparison_table(results_dict)
        self.assertIsInstance(df, pd.DataFrame)
        expected_cols = [
            "Algorithm",
            "Precision@10",
            "Precision@20",
            "Recall@10",
            "Recall@20",
            "Recall@0.5",
            "F1@0.5",
            "ROC-AUC"
        ]
        self.assertListEqual(list(df.columns), expected_cols)
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
