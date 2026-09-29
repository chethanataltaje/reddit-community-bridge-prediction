"""
Person 2 Community Bridge Analysis Module

Analyzes predicted link candidates to identify potential emerging community bridges (cross-community candidates).

Methodological & Terminological Note:
- Cross-community candidates (source_community != target_community) represent potential community bridges.
- They are topological candidate link predictions from the held-out test candidate set and DO NOT constitute
  confirmed future connections or proof of information diffusion.
- Ground-truth annotations ('ground_truth' column) are included solely for post-hoc analysis and DO NOT influence ranking.
"""

import pandas as pd


def annotate_community_bridges(df_predictions: pd.DataFrame, community_map: dict) -> pd.DataFrame:
    """
    Annotates candidate predictions with source/target community IDs and cross-community boolean flags.

    Args:
        df_predictions (pd.DataFrame): DataFrame containing candidate predictions.
        community_map (dict): Dict of subreddit -> community_id.

    Returns:
        pd.DataFrame: Annotated DataFrame with community bridge columns.
    """
    if df_predictions.empty:
        cols = [
            "algorithm", "k", "rank", "source_subreddit", "target_subreddit",
            "score", "source_community", "target_community", "cross_community"
        ]
        if "ground_truth" in df_predictions.columns:
            cols.append("ground_truth")
        return pd.DataFrame(columns=cols)

    df_annotated = df_predictions.copy()

    source_comms = []
    target_comms = []
    cross_flags = []

    for _, row in df_annotated.iterrows():
        u = str(row["source_subreddit"])
        v = str(row["target_subreddit"])

        src_c = community_map.get(u, -1)
        tgt_c = community_map.get(v, -1)

        # Cross-community if source and target belong to different communities
        is_cross = bool(src_c != tgt_c)

        source_comms.append(src_c)
        target_comms.append(tgt_c)
        cross_flags.append(is_cross)

    df_annotated["source_community"] = source_comms
    df_annotated["target_community"] = target_comms
    df_annotated["cross_community"] = cross_flags

    # Order columns cleanly
    base_cols = ["algorithm", "rank", "source_subreddit", "target_subreddit", "score"]
    if "k" in df_annotated.columns:
        base_cols.insert(1, "k")
    comm_cols = ["source_community", "target_community", "cross_community"]
    if "ground_truth" in df_annotated.columns:
        comm_cols.append("ground_truth")

    final_cols = [c for c in base_cols + comm_cols if c in df_annotated.columns]
    return df_annotated[final_cols]


def generate_bridge_summary(df_annotated: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary of cross-community candidate counts and proportions for each algorithm and Top-K cutoff.

    Args:
        df_annotated (pd.DataFrame): Annotated predictions DataFrame with 'cross_community' and 'algorithm' columns.

    Returns:
        pd.DataFrame: Summary DataFrame with columns:
            algorithm | k | total_candidates | cross_community_count | within_community_count | cross_community_proportion
    """
    if df_annotated.empty:
        return pd.DataFrame(columns=[
            "algorithm", "k", "total_candidates", "cross_community_count",
            "within_community_count", "cross_community_proportion"
        ])

    rows = []
    algorithms = df_annotated["algorithm"].unique()

    # Determine K cutoffs present or compute per algorithm
    for algo in algorithms:
        sub_algo = df_annotated[df_annotated["algorithm"] == algo]

        # If 'k' column exists, group by k; otherwise use length
        k_values = sub_algo["k"].unique() if "k" in sub_algo.columns else [len(sub_algo)]

        for k_val in sorted(k_values):
            if "k" in sub_algo.columns:
                sub_k = sub_algo[sub_algo["k"] == k_val]
            else:
                sub_k = sub_algo.head(k_val)

            total = len(sub_k)
            cross_count = int(sub_k["cross_community"].sum())
            within_count = total - cross_count
            prop = float(cross_count / total) if total > 0 else 0.0

            rows.append({
                "algorithm": algo,
                "k": int(k_val),
                "total_candidates": total,
                "cross_community_count": cross_count,
                "within_community_count": within_count,
                "cross_community_proportion": round(prop, 4)
            })

    df_summary = pd.DataFrame(rows)
    cols = [
        "algorithm", "k", "total_candidates", "cross_community_count",
        "within_community_count", "cross_community_proportion"
    ]
    return df_summary[cols]
