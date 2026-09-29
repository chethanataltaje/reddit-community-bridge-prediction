import unittest
import os
import tempfile
import pandas as pd
import networkx as nx
import numpy as np
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from person2.cytoscape.cytoscape_export import (
    select_cytoscape_subgraph_nodes,
    export_cytoscape_nodes,
    export_cytoscape_existing_edges,
    export_cytoscape_predicted_edges
)

from person2.visualizations.generate_plots import (
    plot_algorithm_precision_roc_comparison,
    plot_algorithm_recall_comparison,
    plot_cross_community_bridge_proportion,
    plot_community_size_distribution,
    plot_roc_curves
)

from person2.evaluation.evaluation import extract_labels_and_scores, compute_roc_auc


class TestPerson2CytoscapeVisualizations(unittest.TestCase):

    def setUp(self):
        self.G = nx.Graph()
        self.G.add_edges_from([("A", "B"), ("B", "C"), ("X", "Y")])
        self.comm_map = {"A": 0, "B": 0, "C": 0, "X": 1, "Y": 1}

        self.df_preds = pd.DataFrame([
            {
                "algorithm": "CN", "k": 10, "rank": 1,
                "source_subreddit": "A", "target_subreddit": "X",
                "score": 5.0, "source_community": 0, "target_community": 1,
                "cross_community": True, "ground_truth": 1
            }
        ])

    def test_node_uniqueness_and_columns(self):
        selected_nodes = select_cytoscape_subgraph_nodes(self.G, self.df_preds)
        df_nodes = export_cytoscape_nodes(self.G, self.comm_map, selected_nodes)

        self.assertEqual(len(df_nodes), len(df_nodes["id"].unique()))
        self.assertListEqual(list(df_nodes.columns), ["id", "subreddit", "degree", "community_id"])
        self.assertGreater(len(df_nodes), 0)

    def test_edge_columns_and_export(self):
        selected_nodes = select_cytoscape_subgraph_nodes(self.G, self.df_preds)
        df_edges = export_cytoscape_existing_edges(self.G, selected_nodes)

        self.assertListEqual(list(df_edges.columns), ["source", "target", "edge_type"])
        self.assertGreater(len(df_edges), 0)

    def test_predicted_edge_columns(self):
        df_pred_edges = export_cytoscape_predicted_edges(self.df_preds)
        expected_cols = [
            "source", "target", "algorithm", "score",
            "source_community", "target_community", "cross_community", "rank"
        ]
        for col in expected_cols:
            self.assertIn(col, df_pred_edges.columns)

    def test_visualization_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1a = os.path.join(tmpdir, "precision_roc_comp.png")
            p1b = os.path.join(tmpdir, "recall_comp.png")
            p2 = os.path.join(tmpdir, "bridge_prop.png")
            p3 = os.path.join(tmpdir, "comm_dist.png")
            p4 = os.path.join(tmpdir, "roc.png")

            df_eval = pd.DataFrame([
                {"Algorithm": "CN", "Precision@10": 1.0, "Precision@20": 1.0, "Recall@10": 0.000227, "Recall@20": 0.000454, "ROC-AUC": 0.860476}
            ])
            plot_algorithm_precision_roc_comparison(df_eval, p1a)
            self.assertTrue(os.path.exists(p1a) and os.path.getsize(p1a) > 0)

            plot_algorithm_recall_comparison(df_eval, p1b)
            self.assertTrue(os.path.exists(p1b) and os.path.getsize(p1b) > 0)

            df_bridge_sum = pd.DataFrame([
                {"algorithm": "CN", "k": 10, "cross_community_proportion": 1.0},
                {"algorithm": "CN", "k": 20, "cross_community_proportion": 0.85}
            ])
            plot_cross_community_bridge_proportion(df_bridge_sum, p2)
            self.assertTrue(os.path.exists(p2) and os.path.getsize(p2) > 0)

            plot_community_size_distribution(self.comm_map, p3)
            self.assertTrue(os.path.exists(p3) and os.path.getsize(p3) > 0)

            scores_dict = {"CN": [("A", "B", 5.0), ("A", "X", 1.0)]}
            test_pos = [("A", "B")]
            plot_roc_curves(scores_dict, test_pos, p4)
            self.assertTrue(os.path.exists(p4) and os.path.getsize(p4) > 0)

    def test_roc_auc_consistency_with_evaluation(self):
        # Verify that compute_roc_auc matches expected exact values
        scores_list = [("A", "B", 5.0), ("X", "Y", 1.0), ("C", "D", 4.0), ("W", "Z", 2.0)]
        test_pos = [("A", "B"), ("C", "D")]
        
        y_true, y_score = extract_labels_and_scores(scores_list, test_pos)
        auc_val = compute_roc_auc(y_true, y_score)
        self.assertAlmostEqual(auc_val, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
