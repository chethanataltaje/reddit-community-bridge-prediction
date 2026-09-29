"""
Unified Master Pipeline Runner

Executes all 4 pipeline stages sequentially:
1. Evaluation Metrics Calculation -> results/evaluation_results.csv
2. Top-K Prediction Candidate Extraction -> results/top10_predictions.csv, results/top20_predictions.csv
3. Louvain Community Detection & Bridge Analysis -> results/community_results.csv, results/bridge_analysis.csv, results/bridge_summary.csv
4. Cytoscape Network Exports & Matplotlib Visualizations -> cytoscape/*.csv, visualizations/*.png
"""

import sys
import os
import time
import pickle
import pandas as pd

# Ensure src/ is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import data_preprocessing as dp
import network_analysis as na
import link_prediction as lp
import evaluation as eval_mod
import predictions as pred_mod
import community_detection as comm_mod
import bridge_analysis as bridge_mod
import visualization as viz_mod
import cytoscape_export as cyto_mod


def main():
    print("==================================================")
    print("RUNNING UNIFIED REDDIT BRIDGE PREDICTION PIPELINE")
    print("==================================================")
    
    start_time = time.time()
    
    # Ensure output directories exist
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    viz_dir = os.path.join(PROJECT_ROOT, 'visualizations')
    cyto_dir = os.path.join(PROJECT_ROOT, 'cytoscape')
    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(cyto_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Load Dataset & Construct Graph
    dataset_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'soc-redditHyperlinks-title.tsv')
    print(f"[1/6] Loading raw dataset from {dataset_path}...")
    df = dp.load_dataset(dataset_path)
    
    print("[2/6] Constructing undirected graph & performing train/test split...")
    G = na.construct_undirected_graph(df)
    G_train, train_edges_pos, test_edges_pos = lp.train_test_split_edges(G, test_fraction=0.2, seed=42)
    
    num_neg_samples = len(test_edges_pos)
    test_edges_neg = lp.generate_negative_samples(G, num_neg_samples, seed=42)
    
    dp.save_processed_data(G_train, train_edges_pos, test_edges_pos, test_edges_neg, save_dir=processed_dir)
    candidate_edges = test_edges_pos + test_edges_neg
    
    # 2. Heuristic Link Prediction Scoring
    print("[3/6] Scoring candidate edges via topological heuristics...")
    scores_dict = {
        "Common Neighbors": lp.predict_common_neighbors(G_train, candidate_edges),
        "Jaccard": lp.predict_jaccard(G_train, candidate_edges),
        "Adamic-Adar": lp.predict_adamic_adar(G_train, candidate_edges),
        "Preferential Attachment": lp.predict_preferential_attachment(G_train, candidate_edges)
    }
    
    # 3. Evaluation Metrics
    print("[4/6] Computing evaluation metrics...")
    eval_results = {}
    for name, scores in scores_dict.items():
        y_true, y_score = eval_mod.extract_labels_and_scores(scores, test_edges_pos)
        eval_results[name] = {
            "precision@10": eval_mod.compute_precision_at_k(y_true, y_score, k=10),
            "precision@20": eval_mod.compute_precision_at_k(y_true, y_score, k=20),
            "recall@10": eval_mod.compute_recall_at_k(y_true, y_score, k=10),
            "recall@20": eval_mod.compute_recall_at_k(y_true, y_score, k=20),
            "recall@0.5": eval_mod.compute_classification_metrics(y_true, y_score, threshold=0.5)["recall"],
            "f1@0.5": eval_mod.compute_classification_metrics(y_true, y_score, threshold=0.5)["f1"],
            "roc_auc": eval_mod.compute_roc_auc(y_true, y_score)
        }
    df_eval = eval_mod.generate_comparison_table(eval_results)
    eval_csv = os.path.join(results_dir, 'evaluation_results.csv')
    df_eval.to_csv(eval_csv, index=False)
    
    # 4. Top-K Extraction & Community Analysis
    print("[5/6] Extracting Top-K predictions & running Louvain community detection...")
    df_top10 = pred_mod.generate_top_k_dataframe(scores_dict, k=10, test_edges_pos=test_edges_pos)
    df_top20 = pred_mod.generate_top_k_dataframe(scores_dict, k=20, test_edges_pos=test_edges_pos)
    
    df_top10.to_csv(os.path.join(results_dir, 'top10_predictions.csv'), index=False)
    df_top20.to_csv(os.path.join(results_dir, 'top20_predictions.csv'), index=False)
    
    community_map = comm_mod.detect_communities(G_train, seed=42)
    df_comm = comm_mod.generate_community_mapping_dataframe(community_map)
    df_comm.to_csv(os.path.join(results_dir, 'community_results.csv'), index=False)
    
    df_top10["k"] = 10
    df_top20["k"] = 20
    df_combined_preds = pd.concat([df_top10, df_top20], ignore_index=True)
    
    df_bridge = bridge_mod.annotate_community_bridges(df_combined_preds, community_map)
    df_bridge_sum = bridge_mod.generate_bridge_summary(df_bridge)
    
    df_bridge.to_csv(os.path.join(results_dir, 'bridge_analysis.csv'), index=False)
    df_bridge_sum.to_csv(os.path.join(results_dir, 'bridge_summary.csv'), index=False)
    
    # 5. Cytoscape Exports & Visualizations
    print("[6/6] Exporting Cytoscape CSVs & generating Matplotlib figures...")
    selected_nodes = cyto_mod.select_cytoscape_subgraph_nodes(G_train, df_bridge, max_neighbor_degree=100)
    
    cyto_mod.export_cytoscape_nodes(G_train, community_map, selected_nodes).to_csv(os.path.join(cyto_dir, 'nodes.csv'), index=False)
    cyto_mod.export_cytoscape_existing_edges(G_train, selected_nodes).to_csv(os.path.join(cyto_dir, 'edges.csv'), index=False)
    cyto_mod.export_cytoscape_predicted_edges(df_bridge).to_csv(os.path.join(cyto_dir, 'predicted_edges.csv'), index=False)
    
    viz_mod.plot_algorithm_precision_roc_comparison(df_eval, os.path.join(viz_dir, 'algorithm_precision_roc_comparison.png'))
    viz_mod.plot_algorithm_recall_comparison(df_eval, os.path.join(viz_dir, 'algorithm_recall_comparison.png'))
    viz_mod.plot_cross_community_bridge_proportion(df_bridge_sum, os.path.join(viz_dir, 'cross_community_bridge_proportion.png'))
    viz_mod.plot_community_size_distribution(community_map, os.path.join(viz_dir, 'community_size_distribution.png'))
    viz_mod.plot_roc_curves(scores_dict, test_edges_pos, os.path.join(viz_dir, 'roc_curves.png'))
    
    print("\n==================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
    print("==================================================")

if __name__ == "__main__":
    main()
