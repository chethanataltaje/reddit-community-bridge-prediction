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

from person2.community.community_detection import (
    detect_communities,
    generate_community_mapping_dataframe,
    get_community_statistics
)

from person2.community.bridge_analysis import (
    annotate_community_bridges,
    generate_bridge_summary
)


def main():
    print("==================================================")
    print("PHASE 5: Community Detection & Bridge Analysis")
    print("==================================================")
    
    start_time = time.time()
    
    # 1. Load G_train (from processed pickle if available, else run Person 1 pipeline)
    processed_g_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'G_train.pkl')
    
    if os.path.exists(processed_g_path):
        print(f"[1/4] Loading cached G_train from {processed_g_path}...")
        with open(processed_g_path, 'rb') as f:
            G_train = pickle.load(f)
    else:
        print("[1/4] G_train cache not found. Re-constructing via Person 1 modules...")
        dataset_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'soc-redditHyperlinks-title.tsv')
        df = dp.load_dataset(dataset_path)
        G = na.construct_undirected_graph(df)
        G_train, _, _ = lp.train_test_split_edges(G, test_fraction=0.2, seed=42)
        
    print(f"      G_train has {G_train.number_of_nodes()} nodes and {G_train.number_of_edges()} edges.")
    
    # 2. Phase 5A: Louvain Community Detection on G_train ONLY
    print("[2/4] Executing Louvain community detection on G_train (seed=42)...")
    comm_map = detect_communities(G_train, seed=42)
    comm_stats = get_community_statistics(comm_map)
    
    print("\n--------------------------------------------------")
    print("COMMUNITY DETECTION STATISTICS")
    print("--------------------------------------------------")
    print(f"Total Nodes Assigned:       {comm_stats['total_nodes']}")
    print(f"Total Communities Detected: {comm_stats['num_communities']}")
    print(f"Largest Community Size:     {comm_stats['largest_size']} subreddits")
    print(f"Smallest Community Size:    {comm_stats['smallest_size']} subreddit(s)")
    print("--------------------------------------------------\n")
    
    comm_output_dir = os.path.join(PROJECT_ROOT, 'person2', 'community')
    os.makedirs(comm_output_dir, exist_ok=True)
    
    df_comm_map = generate_community_mapping_dataframe(comm_map)
    comm_csv = os.path.join(comm_output_dir, 'community_results.csv')
    df_comm_map.to_csv(comm_csv, index=False)
    print(f"Saved community mapping ({len(df_comm_map)} rows) to: {comm_csv}")
    
    # 3. Phase 5B & 5C: Load Top-10 / Top-20 Predictions and Annotate Bridges
    print("[3/4] Annotating Top-10 and Top-20 predictions with community structures...")
    top10_csv = os.path.join(PROJECT_ROOT, 'person2', 'predictions', 'top10_predictions.csv')
    top20_csv = os.path.join(PROJECT_ROOT, 'person2', 'predictions', 'top20_predictions.csv')
    
    if not os.path.exists(top10_csv) or not os.path.exists(top20_csv):
        raise FileNotFoundError("Top-10/Top-20 prediction files not found. Run Phase 4 pipeline first.")
        
    df_top10 = pd.read_csv(top10_csv)
    df_top10["k"] = 10
    
    df_top20 = pd.read_csv(top20_csv)
    df_top20["k"] = 20
    
    # Combine predictions
    df_combined_preds = pd.concat([df_top10, df_top20], ignore_index=True)
    
    # Annotate with source/target communities and cross_community flag
    df_bridge_annotated = annotate_community_bridges(df_combined_preds, comm_map)
    
    bridge_analysis_csv = os.path.join(comm_output_dir, 'bridge_analysis.csv')
    df_bridge_annotated.to_csv(bridge_analysis_csv, index=False)
    print(f"Saved bridge analysis ({len(df_bridge_annotated)} rows) to: {bridge_analysis_csv}")
    
    # 4. Phase 5D: Generate Bridge Summary
    print("[4/4] Generating Top-K Community Bridge Summary...")
    df_bridge_summary = generate_bridge_summary(df_bridge_annotated)
    
    bridge_summary_csv = os.path.join(comm_output_dir, 'bridge_summary.csv')
    df_bridge_summary.to_csv(bridge_summary_csv, index=False)
    print(f"Saved bridge summary ({len(df_bridge_summary)} rows) to: {bridge_summary_csv}")
    
    print("\n==================================================")
    print("COMMUNITY BRIDGE SUMMARY TABLE")
    print("==================================================")
    print(df_bridge_summary.to_string(index=False))
    print(f"\nTotal time elapsed: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
