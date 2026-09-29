"""
Person 2 Evaluation Module

Methodology Note:
- Precision@K and Recall@K are ranking-based metrics that evaluate the quality of top-ranked link predictions.
- ROC-AUC measures ranking quality across all candidate pairs using raw continuous prediction scores.
- Recall@0.5 and F1@0.5 are threshold-based classification metrics evaluated at fixed decision boundary threshold 0.5.
- Fixed decision boundary threshold (0.5) is not scale-equivalent across heuristic scoring functions 
  (Common Neighbors count vs Jaccard ratio vs Adamic-Adar logarithmic sum vs Preferential Attachment degree product).
- Therefore, ranking-based metrics (Precision@K, Recall@K, ROC-AUC) are emphasized for comparative baseline evaluation.
  No single heuristic is claimed to be universally 'best' across all operational requirements.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, f1_score


def extract_labels_and_scores(scores_list: list, test_edges_pos: list) -> tuple:
    """
    Extracts binary y_true ground truth labels and y_score prediction arrays.

    Args:
        scores_list (list): List of (u, v, score) tuples.
        test_edges_pos (list): List of (u, v) positive test edge tuples.

    Returns:
        tuple: (y_true as np.ndarray[int], y_score as np.ndarray[float])
    """
    if not scores_list:
        return np.array([], dtype=int), np.array([], dtype=float)

    # Normalize positive test edges for bidirectional matching
    test_pos_set = {tuple(sorted((str(u), str(v)))) for u, v in test_edges_pos}

    y_true = []
    y_score = []

    for u, v, score in scores_list:
        edge_key = tuple(sorted((str(u), str(v))))
        label = 1 if edge_key in test_pos_set else 0
        y_true.append(label)
        y_score.append(float(score))

    return np.array(y_true, dtype=int), np.array(y_score, dtype=float)


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes Area Under ROC Curve (ROC-AUC) using raw continuous prediction scores.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_score (np.ndarray): Predicted continuous scores.

    Returns:
        float: ROC-AUC score (0.0 if empty or single-class ground truth).
    """
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def compute_classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Computes Recall and F1 metrics for binary classification at a given decision boundary threshold.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_score (np.ndarray): Predicted continuous scores.
        threshold (float): Decision boundary threshold (default=0.5).

    Returns:
        dict: {"recall": float, "f1": float}
    """
    if len(y_true) == 0:
        return {"recall": 0.0, "f1": 0.0}

    y_pred = (y_score >= threshold).astype(int)
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {"recall": rec, "f1": f1}


def compute_precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Computes Precision@K: (True positive predictions among top K) / K.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_score (np.ndarray): Predicted scores.
        k (int): Number of top predictions to evaluate.

    Returns:
        float: Precision@K value.
    """
    if len(y_true) == 0 or k <= 0:
        return 0.0

    # Sort descending by score using stable sort
    sort_indices = np.argsort(-y_score, kind='stable')
    top_k_indices = sort_indices[:k]
    num_positives_in_top_k = np.sum(y_true[top_k_indices])

    return float(num_positives_in_top_k / k)


def compute_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Computes Recall@K: (True positive predictions among top K) / (Total number of positive test edges).

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_score (np.ndarray): Predicted scores.
        k (int): Number of top predictions to evaluate.

    Returns:
        float: Recall@K value.
    """
    total_positives = np.sum(y_true)
    if total_positives == 0 or len(y_true) == 0 or k <= 0:
        return 0.0

    sort_indices = np.argsort(-y_score, kind='stable')
    top_k_indices = sort_indices[:k]
    num_positives_in_top_k = np.sum(y_true[top_k_indices])

    return float(num_positives_in_top_k / total_positives)


def generate_comparison_table(results_dict: dict) -> pd.DataFrame:
    """
    Generates structured comparison DataFrame distinguishing ranking vs threshold metrics.

    Args:
        results_dict (dict): Dictionary mapping algorithm name to metrics dictionary.

    Returns:
        pd.DataFrame: DataFrame with exact columns:
            Algorithm | Precision@10 | Precision@20 | Recall@10 | Recall@20 | Recall@0.5 | F1@0.5 | ROC-AUC
    """
    rows = []
    for algo_name, metrics in results_dict.items():
        rows.append({
            "Algorithm": algo_name,
            "Precision@10": metrics.get("precision@10", 0.0),
            "Precision@20": metrics.get("precision@20", 0.0),
            "Recall@10": metrics.get("recall@10", 0.0),
            "Recall@20": metrics.get("recall@20", 0.0),
            "Recall@0.5": metrics.get("recall@0.5", 0.0),
            "F1@0.5": metrics.get("f1@0.5", 0.0),
            "ROC-AUC": metrics.get("roc_auc", 0.0)
        })

    df = pd.DataFrame(rows)
    cols = [
        "Algorithm",
        "Precision@10",
        "Precision@20",
        "Recall@10",
        "Recall@20",
        "Recall@0.5",
        "F1@0.5",
        "ROC-AUC"
    ]
    return df[cols]
