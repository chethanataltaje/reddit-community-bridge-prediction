import networkx as nx
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_test_split_edges(G: nx.Graph, test_fraction: float = 0.2, seed: int = 42):
    """
    Splits the graph edges into a training graph and held-out test edges.
    Removes the test edges from the training graph to prevent data leakage.
    
    Args:
        G (nx.Graph): Original graph
        test_fraction (float): Proportion of edges to use for testing
        seed (int): Random seed for reproducibility
        
    Returns:
        G_train (nx.Graph): Graph with test edges removed
        train_edges_pos (list): List of training edges
        test_edges_pos (list): List of test edges
    """
    logger.info(f"Starting train/test split (test_fraction={test_fraction}, seed={seed})...")
    random.seed(seed)
    
    edges = list(G.edges())
    num_edges = len(edges)
    num_test = int(num_edges * test_fraction)
    
    # Shuffle and split
    random.shuffle(edges)
    test_edges_pos = edges[:num_test]
    train_edges_pos = edges[num_test:]
    
    # Create training graph by removing test edges
    G_train = G.copy()
    G_train.remove_edges_from(test_edges_pos)
    
    logger.info(f"Total edges: {num_edges}")
    logger.info(f"Training edges: {len(train_edges_pos)}")
    logger.info(f"Test edges: {len(test_edges_pos)}")
    
    return G_train, train_edges_pos, test_edges_pos

def generate_negative_samples(G: nx.Graph, num_samples: int, seed: int = 42) -> list:
    """
    Generates negative samples (non-edges) for the dataset.
    Avoids self-loops and existing positive edges.
    
    Args:
        G (nx.Graph): The original graph containing all true edges
        num_samples (int): Number of negative edges to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        list of tuples: Negative edges (u, v)
    """
    logger.info(f"Generating {num_samples} negative samples (seed={seed})...")
    random.seed(seed)
    
    nodes = list(G.nodes())
    negative_edges = set()
    
    while len(negative_edges) < num_samples:
        u = random.choice(nodes)
        v = random.choice(nodes)
        
        # Avoid self-loops
        if u == v:
            continue
            
        # Ensure undirected consistency (u, v) == (v, u)
        edge = tuple(sorted((u, v)))
        
        # Check if it's already a positive edge or already sampled
        if not G.has_edge(u, v) and edge not in negative_edges:
            negative_edges.add(edge)
            
    logger.info(f"Generated {len(negative_edges)} negative edges.")
    return list(negative_edges)

# --- Link Prediction Algorithms ---

def predict_common_neighbors(G: nx.Graph, ebunch: list) -> list:
    """
    Computes Common Neighbors for a list of candidate edges.
    Returns list of (u, v, score)
    """
    scores = []
    for u, v in ebunch:
        if G.has_node(u) and G.has_node(v):
            score = len(list(nx.common_neighbors(G, u, v)))
        else:
            score = 0
        scores.append((u, v, score))
    return scores

def predict_jaccard(G: nx.Graph, ebunch: list) -> list:
    """
    Computes Jaccard Coefficient for a list of candidate edges.
    Returns list of (u, v, score)
    """
    # Filter ebunch to only include nodes present in G
    valid_ebunch = [(u, v) for u, v in ebunch if G.has_node(u) and G.has_node(v)]
    invalid_ebunch = [(u, v) for u, v in ebunch if not (G.has_node(u) and G.has_node(v))]
    
    preds = list(nx.jaccard_coefficient(G, valid_ebunch))
    preds.extend([(u, v, 0.0) for u, v in invalid_ebunch])
    return preds

def predict_adamic_adar(G: nx.Graph, ebunch: list) -> list:
    """
    Computes Adamic-Adar index for a list of candidate edges.
    Returns list of (u, v, score)
    """
    valid_ebunch = [(u, v) for u, v in ebunch if G.has_node(u) and G.has_node(v)]
    invalid_ebunch = [(u, v) for u, v in ebunch if not (G.has_node(u) and G.has_node(v))]
    
    preds = list(nx.adamic_adar_index(G, valid_ebunch))
    preds.extend([(u, v, 0.0) for u, v in invalid_ebunch])
    return preds

def predict_preferential_attachment(G: nx.Graph, ebunch: list) -> list:
    """
    Computes Preferential Attachment score for a list of candidate edges.
    Returns list of (u, v, score)
    """
    valid_ebunch = [(u, v) for u, v in ebunch if G.has_node(u) and G.has_node(v)]
    invalid_ebunch = [(u, v) for u, v in ebunch if not (G.has_node(u) and G.has_node(v))]
    
    preds = list(nx.preferential_attachment(G, valid_ebunch))
    preds.extend([(u, v, 0.0) for u, v in invalid_ebunch])
    return preds
