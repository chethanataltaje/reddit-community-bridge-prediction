"""
Person 2 Cytoscape Export Module

Exports NetworkX G_train subgraph and top predicted link candidates for visualization in Cytoscape.

Subgraph Selection Rule:
1. Target Nodes: Collect all unique subreddits involved in the Top-20 predicted candidate links across all 4 algorithms.
2. Induced Neighborhood Subgraph: Extract these target nodes and their 1-hop neighbors in G_train (excluding ultra-high degree hubs > 100 to maintain clean Cytoscape layout).
3. Node Export (nodes.csv): Unique nodes with id, subreddit, G_train degree, and Louvain community_id. Each node appears exactly once.
4. Existing Edges Export (edges.csv): Training edges in G_train within the subgraph with edge_type='existing_edge'.
5. Predicted Edges Export (predicted_edges.csv): Top candidate prediction links with algorithm, score, source/target communities, cross_community flag, and rank.
"""

import os
import pandas as pd
import networkx as nx


def select_cytoscape_subgraph_nodes(G_train: nx.Graph, df_predictions: pd.DataFrame, max_neighbor_degree: int = 100) -> set:
    """
    Selects a readable, relevant node subset for Cytoscape network visualization.

    Args:
        G_train (nx.Graph): Training graph.
        df_predictions (pd.DataFrame): Predictions DataFrame containing Top-K candidates.
        max_neighbor_degree (int): Maximum degree threshold for 1-hop neighbor inclusion.

    Returns:
        set: Set of subreddit node IDs to include in Cytoscape export.
    """
    if df_predictions.empty:
        return set()

    # Core target nodes from top predictions
    target_nodes = set(df_predictions["source_subreddit"].astype(str)).union(
        set(df_predictions["target_subreddit"].astype(str))
    )

    subgraph_nodes = set(target_nodes)

    for node in target_nodes:
        if G_train.has_node(node):
            neighbors = list(G_train.neighbors(node))
            # Include 1-hop neighbors if degree is within readable limits
            for nbr in neighbors:
                if G_train.degree(nbr) <= max_neighbor_degree or nbr in target_nodes:
                    subgraph_nodes.add(str(nbr))

    return subgraph_nodes


def export_cytoscape_nodes(G_train: nx.Graph, community_map: dict, selected_nodes: set) -> pd.DataFrame:
    """
    Exports unique node records for Cytoscape.

    Args:
        G_train (nx.Graph): Training graph.
        community_map (dict): Dict of subreddit -> community_id.
        selected_nodes (set): Set of selected node IDs.

    Returns:
        pd.DataFrame: DataFrame with columns ['id', 'subreddit', 'degree', 'community_id'].
    """
    records = []
    for node in sorted(selected_nodes):
        node_str = str(node)
        degree = G_train.degree(node) if G_train.has_node(node) else 0
        comm_id = community_map.get(node_str, -1)

        records.append({
            "id": node_str,
            "subreddit": node_str,
            "degree": int(degree),
            "community_id": int(comm_id)
        })

    df_nodes = pd.DataFrame(records)
    if df_nodes.empty:
        return pd.DataFrame(columns=["id", "subreddit", "degree", "community_id"])

    # Guarantee node uniqueness
    df_nodes = df_nodes.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df_nodes


def export_cytoscape_existing_edges(G_train: nx.Graph, selected_nodes: set) -> pd.DataFrame:
    """
    Exports existing training edges within the selected subgraph.

    Args:
        G_train (nx.Graph): Training graph.
        selected_nodes (set): Set of selected node IDs.

    Returns:
        pd.DataFrame: DataFrame with columns ['source', 'target', 'edge_type'].
    """
    # Subgraph induced by selected nodes
    subgraph = G_train.subgraph([n for n in selected_nodes if G_train.has_node(n)])

    records = []
    for u, v in subgraph.edges():
        records.append({
            "source": str(u),
            "target": str(v),
            "edge_type": "existing_edge"
        })

    df_edges = pd.DataFrame(records)
    if df_edges.empty:
        return pd.DataFrame(columns=["source", "target", "edge_type"])
    return df_edges


def export_cytoscape_predicted_edges(df_bridge_annotated: pd.DataFrame) -> pd.DataFrame:
    """
    Exports predicted candidate links formatted for Cytoscape overlay.

    Args:
        df_bridge_annotated (pd.DataFrame): Annotated bridge predictions DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns:
            ['source', 'target', 'algorithm', 'score', 'source_community', 'target_community', 'cross_community', 'rank']
    """
    if df_bridge_annotated.empty:
        return pd.DataFrame(columns=[
            "source", "target", "algorithm", "score",
            "source_community", "target_community", "cross_community", "rank"
        ])

    df = df_bridge_annotated.copy()
    df["source"] = df["source_subreddit"].astype(str)
    df["target"] = df["target_subreddit"].astype(str)

    cols = [
        "source", "target", "algorithm", "score",
        "source_community", "target_community", "cross_community", "rank"
    ]
    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols].reset_index(drop=True)
