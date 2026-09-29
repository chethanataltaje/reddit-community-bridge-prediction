"""
Person 2 Top-K Predictions Module

Terminological & Methodological Note:
- These Top-10 and Top-20 outputs represent high-scoring predicted candidate links (potential emerging connections)
  extracted from the held-out test candidate set.
- They are candidate topological link predictions and are NOT guaranteed future Reddit hyperlinks.
- Ground-truth annotations ('ground_truth' column: 1 for held-out positive test edge, 0 for negative test candidate)
  are included purely for post-hoc analysis/evaluation and DO NOT influence ranking.
"""

import pandas as pd
import numpy as np


def extract_top_k_predictions(
    scores_list: list,
    algorithm_name: str,
    k: int,
    test_edges_pos: list = None
) -> list:
    """
    Extracts Top-K highest-scoring candidate edge predictions for a single algorithm.

    Args:
        scores_list (list): List of (u, v, score) candidate prediction tuples.
        algorithm_name (str): Name of the link prediction algorithm.
        k (int): Number of top candidate predictions to retrieve.
        test_edges_pos (list, optional): Ground-truth positive test edges for analysis annotations.

    Returns:
        list of dict: Top-K prediction records.
    """
    if not scores_list or k <= 0:
        return []

    # Prepare ground-truth set if provided (for post-hoc analysis only)
    test_pos_set = None
    if test_edges_pos is not None:
        test_pos_set = {tuple(sorted((str(u), str(v)))) for u, v in test_edges_pos}

    # Stable sort descending by numerical prediction score
    sorted_candidates = sorted(scores_list, key=lambda item: float(item[2]), reverse=True)

    seen_undirected_pairs = set()
    top_k_records = []

    for u, v, score in sorted_candidates:
        u_str, v_str = str(u), str(v)
        pair_key = tuple(sorted((u_str, v_str)))

        # Skip duplicate undirected node pairs
        if pair_key in seen_undirected_pairs:
            continue

        seen_undirected_pairs.add(pair_key)
        rank = len(top_k_records) + 1

        record = {
            "rank": rank,
            "source_subreddit": u_str,
            "target_subreddit": v_str,
            "algorithm": algorithm_name,
            "score": float(score)
        }

        if test_pos_set is not None:
            record["ground_truth"] = 1 if pair_key in test_pos_set else 0

        top_k_records.append(record)

        if len(top_k_records) == k:
            break

    return top_k_records


def generate_top_k_dataframe(
    all_algorithm_scores: dict,
    k: int,
    test_edges_pos: list = None
) -> pd.DataFrame:
    """
    Generates a combined DataFrame of Top-K prediction candidates across multiple algorithms.

    Args:
        all_algorithm_scores (dict): Dictionary mapping algorithm_name -> list of (u, v, score).
        k (int): Top-K cut-off.
        test_edges_pos (list, optional): Ground-truth positive test edges.

    Returns:
        pd.DataFrame: DataFrame containing combined Top-K predictions.
    """
    all_records = []
    for algo_name, scores in all_algorithm_scores.items():
        algo_records = extract_top_k_predictions(
            scores_list=scores,
            algorithm_name=algo_name,
            k=k,
            test_edges_pos=test_edges_pos
        )
        all_records.extend(algo_records)

    df = pd.DataFrame(all_records)
    if df.empty:
        return pd.DataFrame(columns=["rank", "source_subreddit", "target_subreddit", "algorithm", "score", "ground_truth"])

    cols = ["rank", "source_subreddit", "target_subreddit", "algorithm", "score"]
    if "ground_truth" in df.columns:
        cols.append("ground_truth")

    return df[cols]


def summarize_top_k_precision(df_top10: pd.DataFrame, df_top20: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a machine-readable summary table comparing Top-10 and Top-20 positive counts and precision.

    Args:
        df_top10 (pd.DataFrame): DataFrame of Top-10 predictions.
        df_top20 (pd.DataFrame): DataFrame of Top-20 predictions.

    Returns:
        pd.DataFrame: Machine-readable summary DataFrame.
    """
    summary_rows = []
    algorithms = df_top10["algorithm"].unique() if not df_top10.empty else []

    for algo in algorithms:
        sub_10 = df_top10[df_top10["algorithm"] == algo]
        sub_20 = df_top20[df_top20["algorithm"] == algo]

        pos_10 = int(sub_10["ground_truth"].sum()) if "ground_truth" in sub_10.columns else 0
        pos_20 = int(sub_20["ground_truth"].sum()) if "ground_truth" in sub_20.columns else 0

        p10 = float(pos_10 / len(sub_10)) if len(sub_10) > 0 else 0.0
        p20 = float(pos_20 / len(sub_20)) if len(sub_20) > 0 else 0.0

        summary_rows.append({
            "Algorithm": algo,
            "Top-10 Positive Count": pos_10,
            "Top-20 Positive Count": pos_20,
            "Top-10 Precision": p10,
            "Top-20 Precision": p20
        })

    return pd.DataFrame(summary_rows)
