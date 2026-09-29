"""
Person 2 Result Visualizations Module

Generates publication-quality charts using Matplotlib:
1. algorithm_precision_roc_comparison.png: Comparison of Precision@10, Precision@20, and ROC-AUC.
2. algorithm_recall_comparison.png: Comparison of Recall@10 and Recall@20 (ranking recall).
3. cross_community_bridge_proportion.png: Comparison of Top-10 and Top-20 cross-community proportions with explicit bar value labels.
4. community_size_distribution.png: Linear histogram and log-log rank distribution of detected Louvain community sizes.
5. roc_curves.png: Exact ROC curves for all four baseline link prediction heuristics matching evaluation_results.csv.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from person2.evaluation.evaluation import extract_labels_and_scores, compute_roc_auc


def plot_algorithm_precision_roc_comparison(df_eval: pd.DataFrame, output_path: str):
    """
    Plots comparative bar chart of Precision@10, Precision@20, and ROC-AUC across algorithms.

    Args:
        df_eval (pd.DataFrame): Evaluation results DataFrame.
        output_path (str): Destination file path for PNG image.
    """
    metrics = ["Precision@10", "Precision@20", "ROC-AUC"]
    existing_metrics = [m for m in metrics if m in df_eval.columns]

    x = np.arange(len(df_eval["Algorithm"]))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, metric in enumerate(existing_metrics):
        offset = (i - len(existing_metrics) / 2) * width + width / 2
        rects = ax.bar(
            x + offset,
            df_eval[metric],
            width,
            label=metric,
            color=colors[i % len(colors)]
        )
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight='bold'
            )

    ax.set_ylabel("Metric Score", fontsize=12)
    ax.set_title("Link Prediction Algorithm Performance (Precision & ROC-AUC)", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_eval["Algorithm"], fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close('all')


def plot_algorithm_recall_comparison(df_eval: pd.DataFrame, output_path: str):
    """
    Plots comparative bar chart of Recall@10 and Recall@20 across algorithms.

    Args:
        df_eval (pd.DataFrame): Evaluation results DataFrame.
        output_path (str): Destination file path for PNG image.
    """
    metrics = ["Recall@10", "Recall@20"]
    existing_metrics = [m for m in metrics if m in df_eval.columns]

    x = np.arange(len(df_eval["Algorithm"]))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#9467bd', '#8c564b']

    max_val = max(df_eval[m].max() for m in existing_metrics) if existing_metrics else 0.001

    for i, metric in enumerate(existing_metrics):
        offset = (i - len(existing_metrics) / 2) * width + width / 2
        rects = ax.bar(
            x + offset,
            df_eval[metric],
            width,
            label=metric,
            color=colors[i % len(colors)]
        )
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.6f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight='bold'
            )

    ax.set_ylabel("Recall Score (Rank-based)", fontsize=12)
    ax.set_title("Link Prediction Algorithm Performance (Recall@K)", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_eval["Algorithm"], fontsize=11)
    ax.set_ylim(0, max_val * 1.3 if max_val > 0 else 0.001)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close('all')


def plot_cross_community_bridge_proportion(df_bridge_summary: pd.DataFrame, output_path: str):
    """
    Plots bar chart comparing Top-10 and Top-20 cross-community proportions with bar labels.

    Args:
        df_bridge_summary (pd.DataFrame): Bridge summary DataFrame.
        output_path (str): Destination file path for PNG image.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    algorithms = df_bridge_summary["algorithm"].unique()
    x = np.arange(len(algorithms))
    width = 0.35

    top10_props = []
    top20_props = []

    for algo in algorithms:
        sub = df_bridge_summary[df_bridge_summary["algorithm"] == algo]
        p10 = sub[sub["k"] == 10]["cross_community_proportion"].values
        p20 = sub[sub["k"] == 20]["cross_community_proportion"].values

        top10_props.append(p10[0] if len(p10) > 0 else 0.0)
        top20_props.append(p20[0] if len(p20) > 0 else 0.0)

    rects1 = ax.bar(x - width/2, top10_props, width, label='Top-10 Predictions', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, top20_props, width, label='Top-20 Predictions', color='#e07a5f')

    # Add bar value labels above every bar
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )

    ax.set_ylabel("Cross-Community Proportion", fontsize=12)
    ax.set_title("Proportion of Potential Emerging Community Bridges by Algorithm", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close('all')


def plot_community_size_distribution(community_map: dict, output_path: str):
    """
    Plots log-log rank distribution and linear histogram of Louvain community sizes.

    Args:
        community_map (dict): Dict of subreddit -> community_id.
        output_path (str): Destination file path for PNG image.
    """
    if not community_map:
        return

    sizes = pd.Series(list(community_map.values())).value_counts().values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Linear Histogram
    ax1.hist(sizes, bins=30, color='#4682b4', edgecolor='black', alpha=0.7)
    ax1.set_title("Community Size Distribution (Linear Scale)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Community Size (Subreddits)", fontsize=11)
    ax1.set_ylabel("Frequency (Count of Communities)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Log-Log Rank Distribution
    sorted_sizes = np.sort(sizes)[::-1]
    ranks = np.arange(1, len(sorted_sizes) + 1)

    ax2.loglog(ranks, sorted_sizes, color='#d95f02', linewidth=2)
    ax2.set_title("Community Size Rank Distribution (Log-Log)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Community Rank", fontsize=11)
    ax2.set_ylabel("Community Size", fontsize=11)
    ax2.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close('all')


def plot_roc_curves(scores_dict: dict, test_edges_pos: list, output_path: str):
    """
    Plots ROC curves for all four link prediction heuristics using exact aligned labels and scores.

    Args:
        scores_dict (dict): Dict mapping algorithm_name -> list of (u, v, score).
        test_edges_pos (list): Positive test edge list.
        output_path (str): Destination file path for PNG image.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (algo_name, scores) in enumerate(scores_dict.items()):
        y_true, y_score = extract_labels_and_scores(scores, test_edges_pos)
        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc_val = compute_roc_auc(y_true, y_score)
            ax.plot(
                fpr, tpr,
                color=colors[i % len(colors)],
                lw=2,
                label=f"{algo_name} (AUC = {roc_auc_val:.6f})"
            )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500000)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("ROC Curves for Link Prediction Heuristics", fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close('all')
