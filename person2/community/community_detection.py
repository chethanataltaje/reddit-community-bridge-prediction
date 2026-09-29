"""
Person 2 Community Detection Module

Executes Louvain community detection on G_train strictly to partition nodes into discrete modular communities.
Does NOT use held-out test edges or negative test candidates during community partitioning.
"""

import networkx as nx
import pandas as pd


def detect_communities(G: nx.Graph, seed: int = 42) -> dict:
    """
    Detects communities on the training graph using NetworkX Louvain algorithm.

    Args:
        G (nx.Graph): Training graph G_train.
        seed (int): Random seed for reproducibility (default=42).

    Returns:
        dict: Mapping of subreddit node string -> community_id (int).
    """
    if G.number_of_nodes() == 0:
        return {}

    # Execute NetworkX Louvain communities algorithm
    communities = nx.community.louvain_communities(G, seed=seed)

    community_map = {}
    for comm_id, node_set in enumerate(communities):
        for node in node_set:
            community_map[str(node)] = comm_id

    return community_map


def generate_community_mapping_dataframe(community_map: dict) -> pd.DataFrame:
    """
    Generates a DataFrame mapping subreddits to community IDs.

    Args:
        community_map (dict): Dict of subreddit -> community_id.

    Returns:
        pd.DataFrame: DataFrame with columns ['subreddit', 'community_id'].
    """
    rows = [{"subreddit": k, "community_id": v} for k, v in community_map.items()]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["subreddit", "community_id"])
    return df.sort_values(by=["community_id", "subreddit"]).reset_index(drop=True)


def get_community_statistics(community_map: dict) -> dict:
    """
    Calculates macro community detection statistics.

    Args:
        community_map (dict): Dict of subreddit -> community_id.

    Returns:
        dict: Summary statistics including count, largest/smallest size, and total nodes.
    """
    if not community_map:
        return {
            "num_communities": 0,
            "total_nodes": 0,
            "largest_size": 0,
            "smallest_size": 0,
            "community_sizes": {}
        }

    df = generate_community_mapping_dataframe(community_map)
    size_series = df["community_id"].value_counts()

    return {
        "num_communities": len(size_series),
        "total_nodes": len(community_map),
        "largest_size": int(size_series.max()),
        "smallest_size": int(size_series.min()),
        "community_sizes": size_series.to_dict()
    }


def generate_community_summary_dataframe(community_map: dict) -> pd.DataFrame:
    """
    Generates a DataFrame summarizing community IDs and their node counts.

    Args:
        community_map (dict): Dict of subreddit -> community_id.

    Returns:
        pd.DataFrame: DataFrame with columns ['community_id', 'community_size'].
    """
    if not community_map:
        return pd.DataFrame(columns=["community_id", "community_size"])

    df = generate_community_mapping_dataframe(community_map)
    summary = df["community_id"].value_counts().reset_index()
    summary.columns = ["community_id", "community_size"]
    return summary.sort_values(by="community_id").reset_index(drop=True)
