import networkx as nx
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def construct_undirected_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Constructs an undirected unweighted simple graph from the Reddit Hyperlinks dataset.
    
    Transformations applied:
    1. Extracts 'SOURCE_SUBREDDIT' and 'TARGET_SUBREDDIT'.
    2. Drops duplicate edges to collapse multi-edges into single unweighted edges.
    3. Removes self-loops (where source == target).
    4. Creates a NetworkX Graph (undirected).
    """
    logger.info("Starting graph construction...")
    
    if 'SOURCE_SUBREDDIT' not in df.columns or 'TARGET_SUBREDDIT' not in df.columns:
        raise ValueError("DataFrame must contain 'SOURCE_SUBREDDIT' and 'TARGET_SUBREDDIT' columns")
        
    edge_df = df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].copy()
    logger.info(f"Initial raw edges: {len(edge_df)}")
    
    # Remove self-loops
    edge_df = edge_df[edge_df['SOURCE_SUBREDDIT'] != edge_df['TARGET_SUBREDDIT']]
    logger.info(f"Edges after removing self-loops: {len(edge_df)}")
    
    # Collapse duplicates for undirected projection
    # Sort source and target so (A, B) and (B, A) become identical
    sorted_pairs = np.sort(edge_df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].values, axis=1)
    edge_df_undirected = pd.DataFrame(sorted_pairs, columns=['source', 'target'])
    
    edge_df_undirected = edge_df_undirected.drop_duplicates()
    logger.info(f"Edges after collapsing duplicates to undirected: {len(edge_df_undirected)}")
    
    # Construct NetworkX graph
    G = nx.from_pandas_edgelist(edge_df_undirected, 'source', 'target', create_using=nx.Graph())
    
    logger.info(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def get_network_statistics(G: nx.Graph) -> dict:
    """
    Calculates basic SNA statistics for the constructed graph.
    Note: Does not create an all-node visualization as requested.
    """
    logger.info("Calculating network statistics...")
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
    density = nx.density(G)
    
    # Connected components
    components = list(nx.connected_components(G))
    num_components = len(components)
    
    # Largest connected component
    if num_components > 0:
        lcc = max(components, key=len)
        lcc_size = len(lcc)
        lcc_percent = (lcc_size / num_nodes) * 100
    else:
        lcc_size = 0
        lcc_percent = 0
        
    # Top degree nodes
    degree_dict = dict(G.degree())
    top_degree_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        
    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "average_degree": avg_degree,
        "density": density,
        "num_connected_components": num_components,
        "lcc_size": lcc_size,
        "lcc_percent": lcc_percent,
        "top_degree_nodes": top_degree_nodes
    }
    
    return stats
