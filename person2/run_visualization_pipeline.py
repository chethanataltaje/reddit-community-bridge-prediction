import sys
import os
import time
import pickle
import pandas as pd

# Add src/ and project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

import data_preprocessing as dp
import network_analysis as na
import link_prediction as lp

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


def main():
    print("==================================================")
    print("PHASE 6: Cytoscape Export & Visualizations Pipeline")
    print("==================================================")
    
    start_time = time.time()
    
    # 1. Load cached G_train graph
    processed_g_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'G_train.pkl')
    print(f"[1/4] Loading G_train graph from {processed_g_path}...")
    with open(processed_g_path, 'rb') as f:
        G_train = pickle.load(f)
        
    # Load community mapping
    comm_csv = os.path.join(PROJECT_ROOT, 'person2', 'community', 'community_results.csv')
    print(f"[1/4] Loading community results from {comm_csv}...")
    df_comm = pd.read_csv(comm_csv)
    community_map = dict(zip(df_comm["subreddit"].astype(str), df_comm["community_id"].astype(int)))
    
    # Load bridge analysis results
    bridge_csv = os.path.join(PROJECT_ROOT, 'person2', 'community', 'bridge_analysis.csv')
    print(f"[1/4] Loading bridge analysis from {bridge_csv}...")
    df_bridge = pd.read_csv(bridge_csv)
    
    # Load evaluation comparison table
    eval_csv = os.path.join(PROJECT_ROOT, 'person2', 'evaluation_results.csv')
    print(f"[1/4] Loading evaluation table from {eval_csv}...")
    df_eval = pd.read_csv(eval_csv)
    
    # Load bridge summary
    bridge_sum_csv = os.path.join(PROJECT_ROOT, 'person2', 'community', 'bridge_summary.csv')
    print(f"[1/4] Loading bridge summary from {bridge_sum_csv}...")
    df_bridge_sum = pd.read_csv(bridge_sum_csv)
    
    # Load positive test edges for ROC curve calculation
    test_pos_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'test_edges_pos.csv')
    test_neg_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'test_edges_neg.csv')
    df_pos = pd.read_csv(test_pos_csv)
    df_neg = pd.read_csv(test_neg_csv)
    
    test_edges_pos = [tuple(x) for x in df_pos[['source', 'target']].to_numpy()]
    test_edges_neg = [tuple(x) for x in df_neg[['source', 'target']].to_numpy()]
    candidate_edges = test_edges_pos + test_edges_neg
    
    # 2. Phase 6A: Cytoscape Export
    print("[2/4] Generating Cytoscape CSV exports...")
    cyto_dir = os.path.join(PROJECT_ROOT, 'person2', 'cytoscape')
    os.makedirs(cyto_dir, exist_ok=True)
    
    selected_nodes = select_cytoscape_subgraph_nodes(G_train, df_bridge, max_neighbor_degree=100)
    print(f"      Subgraph selection rule extracted {len(selected_nodes)} nodes (Top-20 candidates + 1-hop neighborhood).")
    
    df_nodes = export_cytoscape_nodes(G_train, community_map, selected_nodes)
    df_existing_edges = export_cytoscape_existing_edges(G_train, selected_nodes)
    df_predicted_edges = export_cytoscape_predicted_edges(df_bridge)
    
    nodes_csv_path = os.path.join(cyto_dir, 'nodes.csv')
    edges_csv_path = os.path.join(cyto_dir, 'edges.csv')
    pred_edges_csv_path = os.path.join(cyto_dir, 'predicted_edges.csv')
    
    df_nodes.to_csv(nodes_csv_path, index=False)
    df_existing_edges.to_csv(edges_csv_path, index=False)
    df_predicted_edges.to_csv(pred_edges_csv_path, index=False)
    
    print(f"      Saved Cytoscape nodes ({len(df_nodes)} rows) to: {nodes_csv_path}")
    print(f"      Saved Cytoscape existing edges ({len(df_existing_edges)} rows) to: {edges_csv_path}")
    print(f"      Saved Cytoscape predicted edges ({len(df_predicted_edges)} rows) to: {pred_edges_csv_path}")
    
    # 3. Phase 6B: Visualizations
    print("[3/4] Generating publication-quality matplotlib figures...")
    viz_dir = os.path.join(PROJECT_ROOT, 'person2', 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    fig1a_path = os.path.join(viz_dir, 'algorithm_precision_roc_comparison.png')
    plot_algorithm_precision_roc_comparison(df_eval, fig1a_path)
    print(f"      Saved Precision & ROC Comparison Plot: {fig1a_path}")
    
    fig1b_path = os.path.join(viz_dir, 'algorithm_recall_comparison.png')
    plot_algorithm_recall_comparison(df_eval, fig1b_path)
    print(f"      Saved Recall Comparison Plot: {fig1b_path}")
    
    fig2_path = os.path.join(viz_dir, 'cross_community_bridge_proportion.png')
    plot_cross_community_bridge_proportion(df_bridge_sum, fig2_path)
    print(f"      Saved Bridge Proportion Plot: {fig2_path}")
    
    fig3_path = os.path.join(viz_dir, 'community_size_distribution.png')
    plot_community_size_distribution(community_map, fig3_path)
    print(f"      Saved Community Distribution Plot: {fig3_path}")
    
    # Calculate heuristic scores for exact ROC curves matching evaluation module
    print("[3/4] Scoring candidates for exact ROC curves...")
    scores_dict = {
        "Common Neighbors": lp.predict_common_neighbors(G_train, candidate_edges),
        "Jaccard": lp.predict_jaccard(G_train, candidate_edges),
        "Adamic-Adar": lp.predict_adamic_adar(G_train, candidate_edges),
        "Preferential Attachment": lp.predict_preferential_attachment(G_train, candidate_edges)
    }
    
    fig4_path = os.path.join(viz_dir, 'roc_curves.png')
    plot_roc_curves(scores_dict, test_edges_pos, fig4_path)
    print(f"      Saved ROC Curves Plot: {fig4_path}")
    
    print("\n==================================================")
    print("PHASE 6 EXECUTION SUMMARY")
    print("==================================================")
    print(f"Exported Nodes:            {len(df_nodes)}")
    print(f"Exported Existing Edges:   {len(df_existing_edges)}")
    print(f"Exported Predicted Edges:  {len(df_predicted_edges)}")
    print(f"Visualizations Generated:  5 PNG figures")
    print(f"Total time elapsed:        {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
