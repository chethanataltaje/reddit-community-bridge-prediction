import sys
import os
import time
import pandas as pd

# Add src/ and project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

import data_preprocessing as dp
import network_analysis as na
import link_prediction as lp

from person2.predictions.predictions import (
    generate_top_k_dataframe,
    summarize_top_k_precision
)


def main():
    print("==================================================")
    print("PHASE 4: Extracting Top-10 and Top-20 Predictions")
    print("==================================================")
    
    start_time = time.time()
    
    # 1. Load Dataset via Person 1 module
    dataset_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'soc-redditHyperlinks-title.tsv')
    print(f"[1/4] Loading raw dataset from {dataset_path}...")
    df = dp.load_dataset(dataset_path)
    
    # 2. Network Construction & Train/Test Split via Person 1 module
    print("[2/4] Constructing graph & Train/Test split...")
    G = na.construct_undirected_graph(df)
    G_train, train_edges_pos, test_edges_pos = lp.train_test_split_edges(G, test_fraction=0.2, seed=42)
    num_neg_samples = len(test_edges_pos)
    test_edges_neg = lp.generate_negative_samples(G, num_neg_samples, seed=42)
    
    candidate_edges = test_edges_pos + test_edges_neg
    print(f"      Candidate set size: {len(candidate_edges)}")
    
    # 3. Link Prediction Heuristics via Person 1 module
    print("[3/4] Scoring candidates via Person 1 heuristics...")
    scores_dict = {
        "Common Neighbors": lp.predict_common_neighbors(G_train, candidate_edges),
        "Jaccard": lp.predict_jaccard(G_train, candidate_edges),
        "Adamic-Adar": lp.predict_adamic_adar(G_train, candidate_edges),
        "Preferential Attachment": lp.predict_preferential_attachment(G_train, candidate_edges)
    }
    
    # 4. Generate Top-10 and Top-20 DataFrames
    print("[4/4] Extracting Top-10 and Top-20 predictions...")
    output_dir = os.path.join(PROJECT_ROOT, 'person2', 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    df_top10 = generate_top_k_dataframe(scores_dict, k=10, test_edges_pos=test_edges_pos)
    df_top20 = generate_top_k_dataframe(scores_dict, k=20, test_edges_pos=test_edges_pos)
    
    top10_file = os.path.join(output_dir, 'top10_predictions.csv')
    top20_file = os.path.join(output_dir, 'top20_predictions.csv')
    
    df_top10.to_csv(top10_file, index=False)
    df_top20.to_csv(top20_file, index=False)
    
    print(f"\nSaved Top-10 predictions ({len(df_top10)} rows) to: {top10_file}")
    print(f"Saved Top-20 predictions ({len(df_top20)} rows) to: {top20_file}")
    
    # Machine-readable summary validation
    df_summary = summarize_top_k_precision(df_top10, df_top20)
    print("\n==================================================")
    print("TOP-K PRECISION SUMMARY")
    print("==================================================")
    print(df_summary.to_string(index=False))
    print(f"\nTotal time elapsed: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
