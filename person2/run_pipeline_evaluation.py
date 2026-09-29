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

from person2.evaluation.evaluation import (
    extract_labels_and_scores,
    compute_roc_auc,
    compute_classification_metrics,
    compute_precision_at_k,
    compute_recall_at_k,
    generate_comparison_table
)

def main():
    print("==================================================")
    print("PHASE 3: Running Person 2 Evaluation on Pipeline")
    print("==================================================")
    
    start_time = time.time()
    
    # 1. Load Dataset via Person 1 module
    dataset_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'soc-redditHyperlinks-title.tsv')
    print(f"[1/5] Loading raw dataset from {dataset_path}...")
    df = dp.load_dataset(dataset_path)
    print(f"      Loaded DataFrame with shape: {df.shape}")
    
    # 2. Network Construction via Person 1 module
    print("[2/5] Constructing undirected graph...")
    G = na.construct_undirected_graph(df)
    print(f"      Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # 3. Train/Test Split & Negative Sampling via Person 1 module
    print("[3/5] Performing Train/Test split (80/20, seed=42)...")
    G_train, train_edges_pos, test_edges_pos = lp.train_test_split_edges(G, test_fraction=0.2, seed=42)
    
    num_neg_samples = len(test_edges_pos)
    print(f"[3/5] Generating {num_neg_samples} negative samples...")
    test_edges_neg = lp.generate_negative_samples(G, num_neg_samples, seed=42)
    
    # Save processed artifacts via Person 1 function
    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    dp.save_processed_data(G_train, train_edges_pos, test_edges_pos, test_edges_neg, save_dir=processed_dir)
    
    # Candidate evaluation edges
    candidate_edges = test_edges_pos + test_edges_neg
    print(f"      Total candidate edges for evaluation: {len(candidate_edges)}")
    
    # 4. Link Prediction Scoring via Person 1 module
    print("[4/5] Computing Link Prediction heuristic scores...")
    print("      -> Common Neighbors...")
    cn_scores = lp.predict_common_neighbors(G_train, candidate_edges)
    
    print("      -> Jaccard Coefficient...")
    jc_scores = lp.predict_jaccard(G_train, candidate_edges)
    
    print("      -> Adamic-Adar Index...")
    aa_scores = lp.predict_adamic_adar(G_train, candidate_edges)
    
    print("      -> Preferential Attachment...")
    pa_scores = lp.predict_preferential_attachment(G_train, candidate_edges)
    
    # 5. Person 2 Evaluation
    print("[5/5] Executing Person 2 Evaluation metrics...")
    
    algorithms = [
        ("Common Neighbors", cn_scores),
        ("Jaccard", jc_scores),
        ("Adamic-Adar", aa_scores),
        ("Preferential Attachment", pa_scores)
    ]
    
    results_dict = {}
    
    for name, scores in algorithms:
        y_true, y_score = extract_labels_and_scores(scores, test_edges_pos)
        
        p10 = compute_precision_at_k(y_true, y_score, k=10)
        p20 = compute_precision_at_k(y_true, y_score, k=20)
        r10 = compute_recall_at_k(y_true, y_score, k=10)
        r20 = compute_recall_at_k(y_true, y_score, k=20)
        class_metrics = compute_classification_metrics(y_true, y_score, threshold=0.5)
        roc_auc = compute_roc_auc(y_true, y_score)
        
        results_dict[name] = {
            "precision@10": p10,
            "precision@20": p20,
            "recall@10": r10,
            "recall@20": r20,
            "recall@0.5": class_metrics["recall"],
            "f1@0.5": class_metrics["f1"],
            "roc_auc": roc_auc
        }
        
    df_table = generate_comparison_table(results_dict)
    
    print("\n==================================================")
    print("FINAL EVALUATION RESULTS TABLE")
    print("==================================================")
    print(df_table.to_string(index=False))
    
    output_csv = os.path.join(PROJECT_ROOT, 'person2', 'evaluation_results.csv')
    df_table.to_csv(output_csv, index=False)
    print(f"\nSaved evaluation results to {output_csv}")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
